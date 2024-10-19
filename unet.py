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


def unet_model(input_size, n_filters, n_classes, **kwargs):
    """
    Constructs a basic UNet model.

    Has 4 2x2 downsampling blocks matched up with 4 2x2 upsampling blocks.
    Automatically adds padding layers to ensure that the input has dimensions
    that are multiples of 2**4 = 16.

    Args:
      input_size: tuple of int, (x, y, channels)
        Input shape
      n_filters: int
        Number of filters for the convolutional layers
      n_classes: int
        Number of output classes

    Keyword args:
      merge_mode -- str, default: 'concat'
        One of: 'concat' or 'add'.
        How skip connection should be combined with up-scaled input.

    Returns:
      model -- tf.keras.Model
    """

    merge_mode = kwargs.get('merge_mode', 'concat')

    # Prepare input
    # (Pad up to the nearest multiple of 16 if needed)
    inputs = Input(input_size)
    pad_h = 16 - input_size[0] % 16
    pad_w = 16 - input_size[1] % 16
    print(f"Input size: {input_size}")
    if pad_h > 0 or pad_w > 0:
        print(f"Added padding layer: w={pad_w}, h={pad_h}")
        padded_inputs = ZeroPadding2D(padding=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(inputs)
    else:
        padded_inputs = inputs

    print(f"Skip-connection merge mode: {merge_mode}")

    # Contracting Path (encoding)
    # (each block here returns two outputs (downsampled, convolved-only),
    #  the latter is used for skip-connections)
    cblock1 = unet_conv_block(padded_inputs, n_filters)
    cblock2 = unet_conv_block(cblock1[0], n_filters*2)
    cblock3 = unet_conv_block(cblock2[0], n_filters*4)
    cblock4 = unet_conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)

    # Bottom layer (no scale changes)
    cblock5 = unet_conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Expanding Path (decoding)
    # (feed in skip-connections
    ublock6 = unet_upsampling_block(cblock5[0], cblock4[1], n_filters*8, merge_mode=merge_mode)
    ublock7 = unet_upsampling_block(ublock6, cblock3[1], n_filters*4, merge_mode=merge_mode)
    ublock8 = unet_upsampling_block(ublock7, cblock2[1], n_filters*2, merge_mode=merge_mode)
    ublock9 = unet_upsampling_block(ublock8, cblock1[1], n_filters, merge_mode=merge_mode)

    conv9 = Conv2D(n_filters,
                   kernel_size=(3, 3),
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)

    # Collapse channels down to output desired classes - using logits so no activation function
    conv10 = Conv2D(filters=n_classes, kernel_size=(1, 1), padding='same')(conv9)

    outputs = conv10
    if pad_h > 0 or pad_w > 0:
        print(f"Added final cropping layer: w={pad_w}, h={pad_h}")
        outputs = Cropping2D(cropping=((pad_h//2, pad_h-pad_h//2), (pad_w//2, pad_w-pad_w//2)))(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def unet_conv_block(inputs, n_filters, dropout_prob=0.0, max_pooling=True):
    """
    Convolutional downsampling block.

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
      next_layer, skip_connection --  Next layer and skip connection outputs
    """

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

    # skip connection needs to be pre-downsampled as it'll be
    # used on the way up post-upscale
    skip_connection = conv

    # max_pooling for downsampling layers
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    return next_layer, skip_connection


def unet_upsampling_block(expansive_input, contractive_input, n_filters, **kwargs):
    """
    Convolutional upsampling block

    Args:
      expansive_input: Input tensor from previous layer
      contractive_input: Input tensor from previous skip layer
      n_filters: Number of filters for the convolutional layers

    Keyword args:
      merge_mode: str, default: 'concat'
        One of: 'concat' or 'add'.
        How skip connection should be combined with up-scaled input.

    Returns:
      conv - Tensor output
    """

    merge_mode = kwargs.get('merge_mode', 'concat')

    up = Conv2DTranspose(
                 filters=n_filters,
                 kernel_size=(3, 3),
                 strides=2,
                 padding='same')(expansive_input)

    # Merge the previous output and the contractive_input
    if merge_mode == 'concat':
        merge = Concatenate(axis=3)([up, contractive_input])
    elif merge_mode == 'add':
        merge = Add()([up, contractive_input])
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
