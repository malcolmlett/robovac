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
#
# Output delta location:
# - delta x,y as percentage of window size
# - probably prefer a linear activation function at the end, or perhaps tanh so that it can be
#   more accurate near the centre.
#
# Output delta orientation:
# - delta angle in some format yet to be decided.
# - I also haven't decided which order the delta location and orientation should be applied in.

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Add


def slam_model(input_size, n_filters, n_classes, **kwargs):
    """
    Constructs a basic UNet model.

    Has 4 2x2 downsampling blocks on both input arms, matched up with 4 2x2 upsampling blocks.
    Automatically adds padding layers to ensure that the input has dimensions
    that are multiples of 2**4 = 16.

    Args:
      input_size: tuple of int, (x, y, channels)
        Input shape of both input images
      n_filters: int
        Number of filters for the convolutional layers
      n_classes: int
        Number of output classes

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

    # Prepare inputs
    # (TODO probably need to construct full size tuple, as lds_input doesn't have channels)
    print(f"Input size: {input_size}")
    map_input = Input(input_size)
    map_input, pad_w, pad_h = pad_block(map_input)

    lds_input = Input(input_size)
    lds_input, _, _ = pad_block(input_size)

    print(f"Skip-connection merge mode: {merge_mode}")
    print("Output: " + ("logits" if output_logits else "softmax"))

    # Map downsampling input arm
    # (each block here returns two outputs (downsampled, convolved-only),
    #  the latter is used for skip-connections)
    map_down, map_skip1 = slam_down_block(map_input, n_filters)
    map_down, map_skip2 = slam_down_block(map_down, n_filters*2)
    map_down, map_skip3 = slam_down_block(map_down, n_filters*4)
    map_down, map_skip4 = slam_down_block(map_down, n_filters*8, dropout_prob=0.3)

    # LDS downsampling input arm
    lds_down, lds_skip1 = slam_down_block(lds_input, n_filters)
    lds_down, lds_skip2 = slam_down_block(lds_down, n_filters*2)
    lds_down, lds_skip3 = slam_down_block(lds_down, n_filters*4)
    lds_down, lds_skip4 = slam_down_block(lds_down, n_filters*8, dropout_prob=0.3)

    # Bottom layer (no scale changes)
    bottom, _ = slam_down_block(map_down, n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Upsampling output arm
    up = slam_up_block(bottom, map_skip4, lds_skip4, n_filters*8, **kwargs)
    up = slam_up_block(up, map_skip3, lds_skip3, n_filters*4, **kwargs)
    up = slam_up_block(up, map_skip2, lds_skip2, n_filters*2, **kwargs)
    up = slam_up_block(up, map_skip1, lds_skip1, n_filters, **kwargs)

    map_out = Conv2D(n_filters,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(up)

    # Collapse channels down to output desired classes - using logits so no activation function
    final_activation = None
    if not output_logits:
        final_activation = 'softmax'
    map_out = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same', activation=final_activation)(map_out)

    if pad_h > 0 or pad_w > 0:
        print(f"Added final cropping layer: w={pad_w}, h={pad_h}")
        map_out = Cropping2D(cropping=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(map_out)

    # Accept and Delta location/orientation output
    accept, delta = ...

    model = tf.keras.Model(inputs=map_input, outputs=[map_out, accept, delta])
    return model


def pad_block(input, input_size):
    pad_h = 16 - input_size[0] % 16
    pad_w = 16 - input_size[1] % 16
    if pad_h > 0 or pad_w > 0:
        print(f"Added padding layer: w={pad_w}, h={pad_h}")
        padded_input = ZeroPadding2D(padding=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(map_input)
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
