import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def weighted_peak_coordinates(pred, system='unit-scale'):
    """
    Computes the (x,y) coordinates of the peak in a heatmap, taken as the value-weighted
    sum of coordinates in a 3x3 grid surrounding the max value.

    Basically, reverses generate_heatmap_image().

    Args:
      pred - predicted heatmaps of shape (B, H, W, C)
      system - 'unit-scale' or 'pixels'.
        The models are dataset assume a 'unit-scale' where the coordinates are given
        as a fraction of the image width/height, relative to the image centre
        and each have the range -0.5 .. +0.5.

        The alternative is the original pixel coordinates, with floating-point precision.
    Returns:
      (B, C, 2) tensor of predicted subpixel (x, y) coordinates
    """
    B, H, W, C = tf.unstack(tf.shape(pred))

    # Prepare input
    # - map to (B, C, H, W) because we're iterating along B & C dimensions,
    #   while selecting into H & W dimensions
    # - pad +/- 1 along H and W dimensions to allow safe 3x3 extraction
    pred = tf.transpose(pred, [0, 3, 1, 2])
    pred = tf.pad(pred, [[0, 0], [0, 0], [1, 1], [1, 1]])

    # Find (x,y) argmax for each channel
    x_peaks = tf.reduce_max(pred, axis=2)                        # (B, W+2, C)
    x_peaks = tf.argmax(x_peaks, axis=-1, output_type=tf.int32)  # (B, C)
    y_peaks = tf.reduce_max(pred, axis=3)                        # (B, H+2, C)
    y_peaks = tf.argmax(y_peaks, axis=-1, output_type=tf.int32)  # (B, C)
    x_peaks = tf.reshape(x_peaks, [B, C, 1])                     # (B, C, 1)
    y_peaks = tf.reshape(y_peaks, [B, C, 1])                     # (B, C, 1)

    # Offsets for 3x3 patch
    dx = tf.constant([-1, 0, 1], tf.int32)
    dy = tf.constant([-1, 0, 1], tf.int32)
    dx, dy = tf.meshgrid(dx, dy)
    dx = tf.reshape(dx, [1, 1, -1])  # (1, 1, 9)
    dy = tf.reshape(dy, [1, 1, -1])  # (1, 1, 9)

    # Compute (x, y) indices for 3x3 patches centered at (x_peaks, y_peaks)
    grid_x_indices = x_peaks + dx  # broadcast to (B, C, 9)
    grid_y_indices = y_peaks + dy  # broadcast to (B, C, 9)
    grid_indices = tf.stack([grid_y_indices, grid_x_indices], axis=-1)  # (B, C, 9, 2)

    # Compute (x, y) coordinates for 3x3 patches centered at (x_peaks, y_peaks)
    # - remove padding
    value_type = pred.dtype
    grid_x_coords = tf.cast(grid_x_indices - 1, value_type)  # (B, C, 9)
    grid_y_coords = tf.cast(grid_y_indices - 1, value_type)  # (B, C, 9)

    # Collect indexed values and normalize as weights
    patch_vals = tf.gather_nd(pred, grid_indices, batch_dims=2)  # (B, C, 9)
    weights = patch_vals / tf.reduce_sum(patch_vals, axis=-1, keepdims=True)

    # Computed value-weighted sum of coordinates
    x_coords = tf.reduce_sum(weights * grid_x_coords, axis=-1)  # (B, C)
    y_coords = tf.reduce_sum(weights * grid_y_coords, axis=-1)  # (B, C)
    coords = tf.stack([x_coords, y_coords], axis=-1)            # (B, C, 2)

    if system == "unit-scale":
        size = tf.cast(tf.reshape(tf.stack([W, H]), [1, 1, 2]), coords.dtype)
        # must be exactly this in order to exactly reverse the 'unit-scale to pixel' conversions used elsewhere
        return (coords - (size // 2)) / size
    elif system == "pixels":
        return coords
    else:
        raise ValueError(f"Invalid coordinate system: {system}")


class HeatmapPeakCoord(layers.Layer):
    """
    Converts from (batch_size, height, width, channels) to (batch_size, channels, 2),
    giving the coordinates of the peak in each channel.
    The peak coordinates are weighted by the values in a 3x3 grid surrounding
    the single max value, for each channel.

    This layer just wraps the weighted_peak_coordinates() function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return weighted_peak_coordinates(inputs, system='unit-scale')


class CoordGrid2D(layers.Layer):
    """
    Returns a (batch_size, height, width, 2) tensor with a grid of (x,y) coordinates.
    Generally this will be subsequently concatenated with the input tensor.

    Based on R. Liu et al. (2018), "An Intriguing Failing of CNNs and the CoordConv Solution" (NeurIPS 2018).
    https://arxiv.org/abs/1807.03247
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # inputs.shape => [batch_size, height, width, channels]
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))

        # Generate a coordinate grid:
        # - x_coords: shape (width,) from -0.5 to 0.5
        # - y_coords: shape (height,) from -0.5 to 0.5
        x_coords = tf.linspace(-0.5, 0.5, width)
        y_coords = tf.linspace(-0.5, 0.5, height)

        # Use meshgrid to get 2D coordinate maps
        # - xx, yy shape => (height, width)
        xx, yy = tf.meshgrid(x_coords, y_coords)

        # Reshape for appending as channels
        xx = tf.expand_dims(xx, axis=-1)  # (height, width, 1)
        yy = tf.expand_dims(yy, axis=-1)  # (height, width, 1)

        # Tile across batch dimension
        xx_tiled = tf.tile(tf.expand_dims(xx, 0), [batch_size, 1, 1, 1])
        yy_tiled = tf.tile(tf.expand_dims(yy, 0), [batch_size, 1, 1, 1])

        # Output as grid with 2-channels
        # output.shape => [batch_size, height, width, 2]
        output = tf.concat([xx_tiled, yy_tiled], axis=-1)
        return output

