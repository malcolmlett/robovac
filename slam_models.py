# NN model for SLAM
#
# Network architecture:
# - UNet with skip-connections
# - Siamese - two independent input pipelines (don't share weights)
# - Main output is an image
# - Additionally branches an output from the bottom of the U to infer:
#    - prob that the agent is actually located within the provided input map
#    - delta to current estimate of location/orientation
#
# Input map semantic map image:
# - A fixed-sized square window snapshot from the known map, or blank if no map known yet
# - Has shape (H, W, C)
# - Always pixel-aligned and in the orientation of the map (no orientation / subpixel sampling
#   will be done when taking windows out of the map)
# - Always represented in probability form (class labels always sum to 1.0 and are in range 0.0 to 1.0).
#   In practice, it is either strict one-hot-encoded from known or simulated ground-truth state,
#   or it is the softmax output from a prior iteration of the network.
#
# Input LDS image:
# - A fixed-sized square image generated from LDS.
# - Has shape (H, W)
# - Exact same size as for input map image.
# - Always oriented to align to map if (estimated) location/orientation known, or otherwise
#   generated assuming centre as location and 0-degrees orientation (positive x-axis).
# - Always pixel aligned to map. This means that the agent's exact estimated location may not be
#   represented by the exact centre of the image, but will be to within the width of a pixel.
# - When the agent's location is not known, the LDS image centre will have no bearing on
#   the agent's location in the map.
#
# Output semantic map update image:
# - A fixed-sized square image representing the generated map and/or update the provided map based on
#   the LDS data.
# - Has shape (H, W, C)
# - Exact same size as for input map image.
# - Output as logits-based semantic map, but always has softmax applied before using / recording.
# - Depending on process being executed and on the accept/reject output, the revised map
#   will be used to update the corresponding section of the full map. A weighted sum is applied
#   based on a mask that accepts 100% of the new map for the middle 3/5-ths diameter, scaling down to 0%
#   at the outer edge diameter, and 0% for all corners outside of the circular region that fits within the square.
#
# Output accept/reject:
# - Probability that the agent is located within the input map.
# - Should always be 1.0 when a blank input map is provided.
# - When performing Localisation fom unknown estimate, the 'accept' output is used to identify
#   which of several window positions the agent is located within - the highest probability window
#   location winning out.
# - When performing map update with known estimate, the 'accept' output is used to identify when
#   the agent has been unexpectedly moved.
# - when outputting raw logits, the value needs to have the following expression applied: sigmoid(o)
#
# Output delta location:
# - The delta update to the current estimate of the agent location within the source map
# - delta x,y as percentage of window size, each in range -0.5 .. +0.5
# - Always outputs in scaled form by applying the following expression: tanh(o) * 0.5
#
# Output delta orientation:
# - The delta update to the current estimate of the agent orientation within the source map
# - delta angle as a multiple of pi, in range -1.0 .. +1.0
# - Always outputs in scaled form by applying the following expression: tanh(o)
# - note: under this definition there is no concern over which order the delta location or orientation are applied,
#   as they are applied to the estimate of the agent's coordinates, not to the maps.

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add

# constants
__CLASSES__ = 3
__FLOOR_IDX__ = 0
__OBSTRUCTION_IDX__ = 1
__UNKNOWN_IDX__ = 2


def slam_model(map_shape, conv_filters=32, adlo_units=256, **kwargs):
    """
    Constructs a basic UNet model.

    Has 4 2x2 downsampling blocks on both input arms, matched up with 4 2x2 upsampling blocks.
    Automatically adds padding layers to ensure that the input has dimensions
    that are multiples of 2**4 = 16.

    Args:
      map_shape: tuple of int, (height, width, channels)
        Size and shape of window used for input and output map representation.
        Implies LDS inputs have shape (batch, height, width).
      conv_filters: int
        Base number of filters for the top-most convolutional layers.
        Deeper layers have powers-of-2 multiplies of this number.
      adlo_units: int
        Number of units at each fully-connected layer within ADLO block (Accept, delta Location/Orientation)

    Keyword args:
      merge_mode: str, default: 'concat'
        One of: 'concat' or 'add'.
        How skip connection should be combined with up-scaled input.
      output_logits: bool, default: True
        Whether to output logits where appropriate, or scaled otherwise.
        The values for delta x/y/orientation are always in scaled form.
      dlo_encoding: one of 'tanh/log-cosh', 'linear/linear', 'linear/importance', default: linear/importance.
        Controls activation function and loss function.
      compile: bool, default: False
        Whether to also compile the model with standard optimizer, loss, and metrics.
      verbose_history: bool, default: False
        Whether to include extra metrics that break-down for the invividual components of the output.

    Returns:
      model -- tf.keras.Model
    """

    merge_mode = kwargs.get('merge_mode', 'concat')
    output_logits = kwargs.get('output_logits', True)
    do_compile = kwargs.get('compile', False)
    dlo_encoding = kwargs.get('dlo_encoding', 'linear/importance')

    # Sanity check
    if np.size(map_shape) != 3:
        raise ValueError(f"Map shape must have 3 dims, found {np.size(map_shape)}")

    # Prepare map input
    # (pad so it's a multiple of our down/up-scaling blocks)
    map_input = Input(shape=map_shape, name='map_input')
    map_down, pad_w, pad_h = pad_block(map_input, map_shape)
    n_classes = map_shape[2]

    # Prepare LDS input
    # (convert from (B,H,W) to (B,H,W,1) to make later work easier)
    # (pad so it's a multiple of our down/up-scaling blocks)
    lds_shape = (map_shape[0], map_shape[1], 1)
    lds_input = Input(shape=(map_shape[0], map_shape[1]), name='lds_input')  # raw input omits channels axis
    lds_down = tf.keras.layers.Reshape(target_shape=lds_shape)(lds_input)
    lds_down, _, _ = pad_block(lds_down, lds_shape)

    # Map downsampling input arm
    # (each block here returns two outputs (downsampled, convolved-only),
    #  the latter is used for skip-connections)
    map_down, map_skip1 = slam_down_block(map_down, conv_filters)
    map_down, map_skip2 = slam_down_block(map_down, conv_filters * 2)
    map_down, map_skip3 = slam_down_block(map_down, conv_filters * 4)
    map_down, map_skip4 = slam_down_block(map_down, conv_filters * 8, dropout_prob=0.3)

    # LDS downsampling input arm
    lds_down, lds_skip1 = slam_down_block(lds_down, conv_filters)
    lds_down, lds_skip2 = slam_down_block(lds_down, conv_filters * 2)
    lds_down, lds_skip3 = slam_down_block(lds_down, conv_filters * 4)
    lds_down, lds_skip4 = slam_down_block(lds_down, conv_filters * 8, dropout_prob=0.3)

    # Bottom layer
    # (combine both input arms, apply some final convolutions, leave at same scale)
    bottom = Concatenate(axis=3)([map_down, lds_down])
    bottom, _ = slam_down_block(bottom, conv_filters * 16, dropout_prob=0.3, max_pooling=False)

    # Upsampling output arm
    up = slam_up_block(bottom, map_skip4, lds_skip4, conv_filters * 8, **kwargs)
    up = slam_up_block(up, map_skip3, lds_skip3, conv_filters * 4, **kwargs)
    up = slam_up_block(up, map_skip2, lds_skip2, conv_filters * 2, **kwargs)
    up = slam_up_block(up, map_skip1, lds_skip1, conv_filters, **kwargs)

    # Final map output
    # (one last convolve, collapse channels down to desired number of output classes, output either logits or softmax,
    #  and remove padding)
    final_activation = None if output_logits else 'softmax'
    final_conv_name = 'map_output' if pad_h == 0 and pad_w == 0 else None
    map_out = Conv2D(conv_filters,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(up)
    map_out = Conv2D(filters=n_classes,
                     kernel_size=(1, 1),
                     padding='same',
                     activation=final_activation,
                     name=final_conv_name)(map_out)
    if pad_h > 0 or pad_w > 0:
        print(f"Added final cropping layer: w={pad_w}, h={pad_h}")
        map_out = Cropping2D(cropping=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)),
                             name='map_output')(map_out)

    # Accept and Delta location/orientation output
    adlo_out = adlo_block(bottom, adlo_units, output_logits, dlo_encoding)

    model = tf.keras.Model(inputs=[map_input, lds_input], outputs=[map_out, adlo_out])

    if do_compile:
        compile_model(model, **kwargs)

    print(f"Prepared SLAM model")
    print(f"  Map shape:        {map_shape} + padding ({pad_h}, {pad_w}, 0)")
    print(f"  Skip-connections: {merge_mode}")
    print(f"  Output scaling:   {'logits' if output_logits else 'scaled'}")
    print(f"  DLO encoding:     {dlo_encoding}")
    print(f"  Layers:           {len(model.layers)}")
    print(f"  Inputs:           {model.inputs}")
    print(f"  Outputs:          {model.outputs}")
    print(f"  Output names:     {model.output_names}")
    print(f"  Compiled:         {do_compile}")
    return model


def compile_model(model, **kwargs):
    """
    Compiles the model according to configuration that has proven to work well.
    Args:
      model:
        A model returned by slam_model()
    Keyword args:
      output_logits: bool, default: True
        Whether outputting logits where appropriate, or scaled otherwise.
      verbose_history: bool, default: False
        Whether to include extra metrics that break-down for the individual components of the output.
    """
    output_logits = kwargs.get('output_logits', True)
    verbose_history = kwargs.get('verbose_history', False)
    dlo_encoding = kwargs.get('dlo_encoding', 'linear/importance')

    # Keras 3.5.0 workaround for dictionary-based loss functions on multi-output models
    # https://github.com/keras-team/keras/issues/20596
    keras350_workaround = (tf.keras.__version__ >= '3.5.0') and (tf.keras.__version__ < '3.7.0')

    if keras350_workaround:
        print(f"WARNING: Keras 3.5.0 workaround activated (current version {tf.keras.__version__}) - "
              f"skipping use of dictionory for multi-output model loss functions")
        loss = (
            MapLoss(from_logits=output_logits),
            ADLOLoss(from_logits=output_logits, dlo_encoding=dlo_encoding)
        )
    else:
        loss = {
            'map_output': MapLoss(from_logits=output_logits),
            'adlo_output': ADLOLoss(from_logits=output_logits, dlo_encoding=dlo_encoding)
        }

    if verbose_history:
        metrics = {
            'map_output': [MapLoss(from_logits=output_logits), MapAccuracy(), ObstructionAccuracy()],
            'adlo_output': [ADLOLoss(from_logits=output_logits, dlo_encoding=dlo_encoding),
                            AcceptAccuracy(), LocationError(), OrientationError()]
        }
    else:
        metrics = None

    model.compile(optimizer='adam', loss=loss, metrics=metrics)


def slam_down_block(inputs, n_filters, dropout_prob=0.0, max_pooling=True):
    """
    Convolutional downsampling block within one input arm
    of the UNet-with-skip-connections.

    Args:
      inputs:
        Input tensor
      n_filters: int
        Number of filters for the convolutional layers
      dropout_prob: float, default: 0.0
        Dropout probability
      max_pooling: bool, default True
        Use MaxPooling2D to reduce the spatial dimensions of the output volume

    Returns:
      down_output, skip_connection -- Main downsampled output and skip connection output
    """

    # core convolutional processing at input resolution
    conv = Conv2D(filters=n_filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(inputs)
    conv = Conv2D(filters=n_filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  # set 'kernel_initializer' same as above
                  kernel_initializer='he_normal')(conv)

    # dropout for some layers
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)

    # skip connection needs to be taken before downsampling as it'll be
    # used on the way up after upscaling
    skip_connection = conv

    # max-pool downsampling step
    if max_pooling:
        conv = MaxPooling2D(pool_size=(2, 2))(conv)

    return conv, skip_connection


def slam_up_block(up_input, skip_input1, skip_input2, n_filters, **kwargs):
    """
    Convolutional upsampling block within output arm of the UNet.
    Takes skip connections from both input arms.

    Args:
      up_input: Input tensor from lower level in UNet
      skip_input1: Input tensor from previous skip layer on first arm
      skip_input2: Input tensor from previous skip layer on second arm
      n_filters: Number of filters for the convolutional layers

    Keyword args:
      merge_mode: str, default: 'concat'
        One of: 'concat' or 'add'.
        How skip connection should be combined with up-scaled input.

    Returns:
      conv - Tensor output
    """

    merge_mode = kwargs.get('merge_mode', 'concat')

    # Upsample
    up = Conv2DTranspose(
                 filters=n_filters,
                 kernel_size=(3, 3),
                 strides=2,
                 padding='same')(up_input)

    # Merge the previous output and the skip_inputs
    if merge_mode == 'concat':
        merge = Concatenate(axis=3)([up, skip_input1, skip_input2])
    elif merge_mode == 'add':
        merge = Add()([up, skip_input1, skip_input2])
    else:
        raise ValueError(f"Unknown merge mode: {merge_mode}")

    conv = Conv2D(n_filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(conv)

    return conv


def adlo_block(input, n_units, output_logits, dlo_encoding):
    """
    Accept and Delta Location/Orientation block.

    Generates an output tensor in the form of a vector (per batch row):
    - accept: logit/sigmoid - likelihood that the agent is present within the map, based on the LDS data
    - delta x: -0.5 .. +0.5 - percentage of window width to add to current estimated x-location
    - delta y: -0.5 .. +0.5 - percentage of window width to add to current estimated y-location
    - delta angle: -1.0 .. +1.0 - fraction of pi (+/-) to add to current estimated orientation

    :param input: Input tensor
    :param n_units: number of units in hidden layers
    :param output_logits: whether to output accept as logits or scaled. delta x/y/angle always scaled
    :return: adlo output (B,4)
    """

    # Reduce the level of dimensionality and flatten
    adlo = Conv2D(16,
                  kernel_size=(3, 3),
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal')(input)
    adlo = Flatten()(adlo)

    # Apply some fully-connected layers
    adlo = Dense(units=n_units, activation='relu')(adlo)
    adlo = Dense(units=n_units, activation='relu')(adlo)

    # Squish down into our output shape and apply final activations
    adlo = Dense(units=4)(adlo)
    output = ADLOActivation(output_logits=output_logits, dlo_encoding=dlo_encoding, name='adlo_output')(adlo)

    return output


def pad_block(x, input_size):
    """
    Works out if padding is required and applies it if needed.
    Args:
      x: image or map tensor (B,H,W,...)
      input_size: size of input (H,W,...)
    Returns:
      padded tensor (B,H+pad_h,W+pad_w,...), total padded width, total padded height
    """
    pad_h = 16 - input_size[0] % 16
    pad_w = 16 - input_size[1] % 16
    if pad_h > 0 or pad_w > 0:
        padded = ZeroPadding2D(padding=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(x)
    else:
        padded = x
    return padded, pad_w, pad_h


class ADLOActivation(tf.keras.layers.Layer):
    """
    Applies activation functions against each component of the ADLO output tensor.
    """

    def __init__(self, output_logits=True, dlo_encoding='linear/importance', name=None, **kwargs):
        """
        Args:
          output_logits: bool
            Whether to output raw logits for the 'accept' component, or to applied sigmoid activation function.
            Ignored for DLO components.
        """
        super(ADLOActivation, self).__init__(name=name, **kwargs)
        self._output_logits = output_logits
        self._dlo_encoding = dlo_encoding

    def call(self, inputs, *args, **kwargs):
        if self._output_logits:
            accept = inputs[:, 0]
        else:
            accept = tf.nn.sigmoid(inputs[:, 0])  # accept prob in range 0.0 .. 1.0

        if self._dlo_encoding.startswith('tanh/'):
            delta_x = tf.nn.tanh(inputs[:, 2]) * 0.5  # delta inputs in range -0.5 .. +0.5 (fraction of window size)
            delta_y = tf.nn.tanh(inputs[:, 3]) * 0.5  # delta y in range -0.5 .. +0.5 (fraction of window size)
            delta_angle = tf.nn.tanh(inputs[:, 1])    # delta angle in range -1.0 .. 1.0 (multiples of pi)
        elif self._dlo_encoding.startswith('linear/'):
            delta_x = inputs[:, 2] * 0.5  # delta inputs in range -0.5 .. +0.5 (fraction of window size)
            delta_y = inputs[:, 3] * 0.5  # delta y in range -0.5 .. +0.5 (fraction of window size)
            delta_angle = inputs[:, 1]  # delta angle in range -1.0 .. 1.0 (multiples of pi)
        else:
            raise ValueError(f"Unknown dlo encoding: {self._dlo_encoding}")

        return tf.stack([accept, delta_x, delta_y, delta_angle], axis=1)


# Note: don't add @tf.function to custom loss classes/functions. Auto-graphing is implied for these
# and adding @tf.function seems to double up the graphing and makes everything run 100x slower.
class MapLoss(tf.keras.losses.Loss):
    """
    Custom variant of a CategoricalCrossEntropyLoss that omits samples from the loss calculation
    where there is no ground-truth map output.
    """
    def __init__(self, name="map_loss", from_logits=True, reduction="sum_over_batch_size"):
        """
        Args:
          from_logits: bool, must be supplied the same as when constructing the model.
          reduction:
        """
        super(MapLoss, self).__init__(name=name, reduction=reduction)
        self._from_logits = from_logits

    def call(self, y_true, y_pred):
        # Compute mask - ignore samples where ground-truth output map is blank
        unknown_true = y_true[..., __UNKNOWN_IDX__]  # shape: (B,H,W)
        unknown_min = tf.reduce_min(unknown_true, axis=(1, 2))  # shape: (B,)
        mask = tf.cast(tf.not_equal(unknown_min, 1.0), y_pred.dtype)  # shape: (B,)

        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=self._from_logits)
        loss = tf.reduce_mean(loss, axis=(1, 2))  # shape: (B,)
        loss = loss * mask

        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)  # shape: scalar

    def get_config(self):
        config = super(MapLoss, self).get_config()
        config.update({
            "from_logits": self._from_logits
        })
        return config


class MapAccuracy(tf.keras.metrics.Metric):
    """
    Metric function against the output map.

    Computes the percentage of correct category predictions after scaling
    and discretizing.

    Assumes:
      y_true: (B,H,W,3), probs
      y_pred: (B,H,W,3), logits or probs
      sample_weight: (B,) or None
    Returns:
      metric scalar 0.0 .. 1.0
    """
    def __init__(self, name="map_accuracy", **kwargs):
        super(MapAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")  # sum over (B,)
        self.correct = self.add_weight(name="correct", initializer="zeros")  # sum over (B,)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute mask - ignore samples where ground-truth output map is blank
        unknown_true = y_true[..., __UNKNOWN_IDX__]  # shape: (B,H,W)
        unknown_min = tf.reduce_min(unknown_true, axis=(1, 2))  # shape: (B,)
        mask = tf.cast(tf.not_equal(unknown_min, 1.0), y_pred.dtype)  # shape: (B,)

        y_true_categories = tf.argmax(y_true, axis=-1)  # (B,H,W) x int
        y_pred_categories = tf.argmax(y_pred, axis=-1)  # (B,H,W) x int
        matches = tf.equal(y_true_categories, y_pred_categories)  # (B,H,W) x bool
        matches = tf.cast(matches, self.dtype)
        accuracies = tf.reduce_mean(matches, axis=(1, 2))  # (B,) x 0..1

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            accuracies *= sample_weight

        self.correct.assign_add(tf.reduce_sum(accuracies * mask))
        self.total.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.total.assign(0.0)
        self.correct.assign(0.0)


class ObstructionAccuracy(tf.keras.metrics.Metric):
    """
    Metric function against the output map that focuses on
    obstruction detection.
    This is because only a few pixels of any given output map are obstructions,
    so MapAccuracy() alone can ignore significant problems.

    This metric measures the accuracy across all pixels whether _either_
    the true state is an obstruction or the predicted state is an obstruction.
    In effect, this calculates the accuracy primarily based on False Negative and False Positive.

    Assumes:
      y_true: (B,H,W,3), probs
      y_pred: (B,H,W,3), logits or probs
      sample_weight: (B,) or None
    Returns:
      metric scalar 0.0 .. 1.0
    """
    def __init__(self, name="obstruction_accuracy", **kwargs):
        super(ObstructionAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")  # sum over (B,)
        self.correct = self.add_weight(name="correct", initializer="zeros")  # sum over (B,)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Sample mask - ignore samples where ground-truth output map is blank
        unknown_true = y_true[..., __UNKNOWN_IDX__]  # shape: (B,H,W)
        unknown_min = tf.reduce_min(unknown_true, axis=(1, 2))  # shape: (B,)
        mask = tf.cast(tf.not_equal(unknown_min, 1.0), y_pred.dtype)  # shape: (B,)

        # Pixel mask - include only pixels whether EITHER y_true or y_pred have on obstruction
        y_true_categories = tf.argmax(y_true, axis=-1)  # (B,H,W) x int
        y_pred_categories = tf.argmax(y_pred, axis=-1)  # (B,H,W) x int
        pixel_mask = tf.cast(tf.logical_or(
            tf.equal(y_true_categories, __OBSTRUCTION_IDX__),
            tf.equal(y_pred_categories, __OBSTRUCTION_IDX__)
        ), self.dtype)  # shape: (B,H,W) x float of 0.0 or 1.0

        matches = tf.equal(y_true_categories, y_pred_categories)  # (B,H,W) x bool
        matches = tf.cast(matches, self.dtype)  # (B,H,W) x float
        matches *= pixel_mask

        _correct = tf.reduce_sum(matches, axis=(1, 2)) * mask
        _total = tf.reduce_sum(pixel_mask, axis=(1, 2)) * mask

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            _correct *= sample_weight
            _total *= sample_weight

        self.correct.assign_add(tf.reduce_sum(_correct))
        self.total.assign_add(tf.reduce_sum(_total))

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.correct.assign(0.0)


# Note: don't add @tf.function to custom loss classes/functions. Auto-graphing is implied for these
# and adding @tf.function seems to double up the graphing and makes everything run 100x slower.
class ADLOLoss(tf.keras.losses.Loss):
    """
    Custom loss for ADLO output.
    By default, assumes output_logits=True in model.

    Assumes:
      y_true: (B,4), scaled
      y_pred: (B,4), accept part logit or scaled, and DLO parts scaled always
    """
    def __init__(self, name="adlo_loss", from_logits=True, dlo_encoding='linear/importance', reduction=None):
        """
        Args:
          from_logits: bool, must be supplied the same as for ADLOActivation.
          reduction: ignored. Required for deserialization.
        """
        super(ADLOLoss, self).__init__(name=name)
        self._from_logits = from_logits
        self._dlo_encoding = dlo_encoding

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        # binary cross-entropy loss for accept
        accept_true = y_true[:, 0]  # shape: (B,)
        accept_pred = y_pred[:, 0]  # shape: (B,)
        accept_losses = tf.keras.losses.binary_crossentropy(accept_true, accept_pred, from_logits=self._from_logits)

        # DLO mask - simply: include if accept_true, exclude otherwise
        mask = accept_true

        # loss for delta x, y, and orientation
        dlo_true = y_true[:, 1:4]  # shape: (B,3)
        dlo_pred = y_pred[:, 1:4]  # shape: (B,3)
        if self._dlo_encoding.endswith('/log-cosh'):
            dlo_losses = tf.math.log(tf.cosh(dlo_pred - dlo_true))
        elif self._dlo_encoding.endswith('/linear'):
            dlo_losses = tf.math.abs(dlo_pred - dlo_true)
        elif self._dlo_encoding.endswith('/importance'):
            dlo_losses = tf.math.abs(dlo_pred - dlo_true)

            # For importance calculation we first convert dlo_true's value ranges as follows:
            #   delta-x:     abs(-0.5..+0.5) * 2 -> 0..1
            #   delta-y:     abs(-0.5..+0.5) * 2 -> 0..1
            #   delta-angle: abs(-1.0..+1.0)     -> 0..1
            # As a computationally efficient approximation of a L2 distance that is biased towards
            # giving higher importance, we simply take the max of delta-x and delta-y (ie: the L-inf norm),
            # and combine back with the unchanged third column.
            # This gives everything in range 0.0 to 1.0. Finally, we apply an importance scaling
            # s.t.
            #   y_true ~= 0   -> 4.0x loss   <-- more importance when very close to zero
            #   y_true ~= 0.2 -> 2.5x loss   <-- still high importance when near zero
            #   y_true ~= 1.0 -> 1.0x loss   <-- drops to normal loss at edges
            scaled_dist_true = tf.reduce_max(tf.abs(dlo_true[:, 0:2]), axis=-1, keepdims=True) * 2  # (B,1) x 0..1
            scaled_dlo_true = tf.concat([scaled_dist_true, scaled_dist_true, tf.abs(dlo_true[:, 2:])],
                                        axis=-1)  # (B,3) x 0..1
            dlo_importance = 4 / (1 + 3 * scaled_dlo_true)  # (B,3) x 4..1
            dlo_losses *= dlo_importance
        else:
            raise ValueError(f"Unknown dlo encoding: {self._dlo_encoding}")

        dlo_losses = tf.reduce_sum(dlo_losses, axis=-1)  # shape: (B,)
        dlo_losses = dlo_losses * mask
        dlo_losses = tf.reduce_sum(dlo_losses) / (tf.reduce_sum(mask) + 1e-8)

        return accept_losses + dlo_losses  # shape: scalar

    def get_config(self):
        config = super(ADLOLoss, self).get_config()
        config.update({
            "from_logits": self._from_logits,
            "dlo_encoding": self._dlo_encoding
        })
        return config


class AcceptAccuracy(tf.keras.metrics.Metric):
    """
    Metric function against the ADLO 'accept' output.
    By default, assumes output_logits=True in model.

    Computes the percentage of correct 'accept' booleans after scaling
    and discretizing.

    Assumes:
      y_true: (B,4), scaled
      y_pred: (B,4), logits or scaled
    Returns:
      metric scalar 0.0 .. 1.0
    """
    def __init__(self, name="accept_accuracy", from_logits=True, **kwargs):
        super(AcceptAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self._from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        accept_true = tf.cast(y_true[:, 0], tf.float32)
        if self._from_logits:
            accept_pred = tf.cast(tf.round(tf.nn.sigmoid(y_pred[:, 0])), tf.float32)
        else:
            accept_pred = tf.cast(tf.round(y_pred[:, 0]), tf.float32)
        matches = tf.equal(accept_true, accept_pred)
        matches = tf.cast(matches, self.dtype)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            matches *= sample_weight

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.shape(y_true)[0], self.dtype))

    def result(self):
        return self.correct / self.total

    def reset_states(self):
        self.total.assign(0.0)
        self.correct.assign(0.0)


class LocationError(tf.keras.metrics.Metric):
    """
    Metric against the ADLO 'delta location' output.
    Computes the RMS error on the 'delta location' coordinate.
    Masked to only include samples where accept_true=True.

    Assumes:
      y_true: (B,4), scaled
      y_pred: (B,4), with DL parts scaled (always)
    Returns:
      metric scalar 0.0 .. ~0.5
    """
    def __init__(self, name="loc_error", **kwargs):
        super(LocationError, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")  # sum over (B,)
        self.count = self.add_weight(name="count", initializer="zeros")  # sum over (B,)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        accept_true = y_true[:, 0]  # shape: (B,) x 0.0 or 1.0
        loc_true = y_true[:, 1:3]  # shape: (B,2) x -0.5 .. +0.5
        loc_pred = y_pred[:, 1:3]  # shape: (B,2) x -0.5 .. +0.5

        # compute RMSE per sample
        errors = tf.square(loc_true - loc_pred)  # shape: (B,2)
        errors = tf.math.sqrt(tf.reduce_mean(errors, axis=-1))  # shape: (B,)

        # DLO mask: include if accept_true, exclude otherwise
        mask = accept_true  # shape: (B,)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            errors *= sample_weight

        self.total_error.assign_add(tf.reduce_sum(errors * mask))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total_error / self.count

    def reset_states(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)


class OrientationError(tf.keras.metrics.Metric):
    """
    Metric against the ADLO 'delta orientation' output.

    Computes the abs mean error on the 'delta orientation' value.
    (This is the most comparable with how we're computing the LocationError).
    Masked to only include samples where accept_true=True.

    Assumes:
      y_true: (B,4), scaled
      y_pred: (B,4), with DO part scaled (always)
    Returns:
      metric scalar 0.0 .. ~1.0
    """
    def __init__(self, name="orientation_error", **kwargs):
        super(OrientationError, self).__init__(name=name, **kwargs)
        self.total_error = self.add_weight(name="total_error", initializer="zeros")  # sum over (B,)
        self.count = self.add_weight(name="count", initializer="zeros")  # sum over (B,)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        accept_true = y_true[:, 0]  # shape: (B,) x 0.0 or 1.0
        angle_true = y_true[:, 3]  # shape: (B,) x -1.0 .. +1.0
        angle_pred = y_pred[:, 3]  # shape: (B,) x -1.0 .. +1.0
        errors = tf.math.abs(angle_true - angle_pred)  # shape: (B,)

        # DLO mask - simply: include if accept_true, exclude otherwise
        mask = accept_true  # shape: (B,)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            errors *= sample_weight

        self.total_error.assign_add(tf.reduce_sum(errors * mask))
        self.count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total_error / self.count

    def reset_states(self):
        self.total_error.assign(0.0)
        self.count.assign(0.0)
