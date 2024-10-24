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
# - when outputting raw logits, the values need to have the following expression applied: tanh(o) * 0.5
#
# Output delta orientation:
# - The delta update to the current estimate of the agent orientation within the source map
# - delta angle as a multiple of pi, in range -1.0 .. +1.0
# - when outputting raw logits, the value needs to have the following expression applied: tanh(o)
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


def slam_model(map_shape, conv_filters, adlo_units, **kwargs):
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
        Whether to output logits, or softmax otherwise.

    Returns:
      model -- tf.keras.Model
    """

    merge_mode = kwargs.get('merge_mode', 'concat')
    output_logits = kwargs.get('output_logits', True)

    # Sanity check
    if np.size(map_shape) != 3:
      raise ValueError("Map shape must have 3 dims, found {np.size(map_shape)}")

    # Prepare map input
    # (pad so it's a multiple of our down/up-scaling blocks)
    map_input = Input(shape=map_shape, name='map_input')
    map_down, pad_w, pad_h = pad_block(map_input, map_shape)
    n_classes = map_shape[2]
    print(f"Map shape: {map_shape} + padding ({pad_h}, {pad_w}, 0)")

    # Prepare LDS input
    # (convert from (B,H,W) to (B,H,W,1) to make later work easier)
    # (pad so it's a multiple of our down/up-scaling blocks)
    lds_shape = (map_shape[0], map_shape[1], 1)
    lds_input = Input(shape=(map_shape[0], map_shape[1]), name='lds_input')  # raw input omits channels axis
    lds_down = tf.keras.layers.Reshape(target_shape=lds_shape)(lds_input)
    lds_down, _, _ = pad_block(lds_down, lds_shape)

    print(f"Skip-connection merge mode: {merge_mode}")
    print("Output: " + ("logits" if output_logits else "scaled"))

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
    map_out = Conv2D(conv_filters,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(up)
    map_out = Conv2D(filters=n_classes,
                     kernel_size=(1, 1),
                     padding='same',
                     activation=final_activation)(map_out)
    if pad_h > 0 or pad_w > 0:
        print(f"Added final cropping layer: w={pad_w}, h={pad_h}")
        map_out = Cropping2D(cropping=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(map_out)

    # Accept and Delta location/orientation output
    adlo_out = adlo_block(bottom, adlo_units, output_logits)

    model = tf.keras.Model(inputs=[map_input, lds_input], outputs=[map_out, adlo_out])
    return model


def pad_block(input, input_size):
    """
    Works out if padding is required and applies it if needed.
    :param input: Input image or map tensor (B,H,W,...)
    :param input_size: size of input (H,W,...)
    :return: padded input (B,H+pad_h,W+pad_w,...), total padded width, total padded height
    """
    pad_h = 16 - input_size[0] % 16
    pad_w = 16 - input_size[1] % 16
    if pad_h > 0 or pad_w > 0:
        padded_input = ZeroPadding2D(padding=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(input)
    else:
        padded_input = input
    return padded_input, pad_w, pad_h


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


def adlo_block(input, n_units, output_logits):
    """
    Accept and Delta Location/Orientation block.

    Generates an output tensor in the form of a vector (per batch row):
    - accept: logit/sigmoid - likelihood that the agent is present within the map, based on the LDS data
    - delta x: logit/(tanh/2) - percentage of window width to add to current estimated x-location
    - delta y: logit/(tanh/2) - percentage of window width to add to current estimated y-location
    - delta angle: logit/tanh - fraction of pi (+/-) to add to current estimated orientation

    :param input: Input tensor
    :param n_units: number of units in hidden layers
    :param output_logits: whether to output logits or scaled
    :return: adlo output (B,4)
    """

    # Apply some fully-connected layers
    adlo = Flatten()(input)
    adlo = Dense(units=n_units, activation='relu')(adlo)
    adlo = Dense(units=n_units, activation='relu')(adlo)

    # Squish down into our output shape
    output = Dense(units=4)(adlo)

    # (Optional) Apply activations
    if not output_logits:
        accept = tf.nn.sigmoid(output[:, 0])  # accept prob in range 0.0 .. 1.0
        delta_x = tf.nn.tanh(output[:, 2]) * 0.5  # delta x in range -0.5 .. +0.5 (fraction of window size)
        delta_y = tf.nn.tanh(output[:, 3]) * 0.5  # delta y in range -0.5 .. +0.5 (fraction of window size)
        delta_angle = tf.nn.tanh(output[:, 1])  # delta angle in range -1.0 .. 1.0 (multiples of pi)
        output = tf.stack([accept, delta_x, delta_y, delta_angle], axis=1)

    return output
