import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import LogLocator
import IPython.display as idisplay
import math
import tqdm
import cv2
import time
import re
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
import train_instrumentation as tinstr


def create_dataset(n=1000, width=149, height=149):
    """
    Creates full dataset as tuples of (input_image, coordinate_output, heatmap_output)
    """
    # generate random coordinates
    # - units: fraction of width/height, relative to centre, in range -0.5 .. +0.5
    # - avoid 10%-width edge -> so limit to range -0.4 .. +0.4
    # - discretize to align to pixels by converting to pixel coords and then back again
    sizes = tf.constant([[height, width]], dtype=tf.float32)
    corner_coords = tf.random.uniform(shape=(n, 2), minval=-0.4, maxval=+0.4)  # float in range -0.4 .. +0.4
    corner_coords = tf.cast(tf.round(corner_coords * sizes + sizes // 2), tf.int32)  # int in range 14 .. 134
    corner_coords = (tf.cast(corner_coords,
                             tf.float32) - sizes // 2) / sizes  # float in approx. range -0.4 .. +0.4, but discretized

    images = []
    heatmaps = []
    for corner_coord in tqdm.tqdm(corner_coords):
        abs_coord = np.array(corner_coord * 149 + 149 // 2, dtype=np.int32)
        image = generate_training_image(abs_coord[0], abs_coord[1])
        heatmap = generate_heatmap_image(abs_coord[0], abs_coord[1])
        images.append(tf.constant(image, tf.float32))
        heatmaps.append(tf.constant(heatmap, tf.float32))
    dataset = tf.data.Dataset.from_tensor_slices((images, corner_coords, heatmaps))
    return dataset


def to_input_dataset(image, coord, heatmap):
    """
    Maps a dataset row to a single input_image.
    Usage: dataset.map(to_coordinate_dataset)
    """
    return image


def to_coordinate_dataset(image, coord, heatmap):
    """
    Maps a dataset row to a tuple of (input_image, coordinate_output).
    Usage: dataset.map(to_coordinate_dataset)
    """
    return (image, coord)


def to_heatmap_dataset(image, coord, heatmap):
    """
    Maps a dataset row to a tuple of (input_image, heatmap_output).
    Usage: dataset.map(to_heatmap_dataset)
    """
    return (image, heatmap)


def _plot_dataset_entry(image, corner_coords):
    abs_coords = corner_coords * 149 + 149 // 2
    plt.imshow(image, cmap='gray')
    plt.scatter(abs_coords[0], abs_coords[1], s=100, edgecolors='b', facecolors='none', linewidth=2)
    plt.axis('off')


def _plotadd_pred_coord(pred_coords):
    edge_x = -0.5 if pred_coords[0] < -0.5 else +0.5 if pred_coords[0] > 0.5 else None
    edge_y = -0.5 if pred_coords[1] < -0.5 else +0.5 if pred_coords[1] > 0.5 else None

    if edge_x is not None or edge_y is not None:
        if edge_x is not None:
            edge_x = edge_x * 149 + 149 // 2
            plt.plot([edge_x, edge_x], [0, 148], color='r', linewidth=2)
        if edge_y is not None:
            edge_y = edge_y * 149 + 149 // 2
            plt.plot([0, 148], [edge_y, edge_y], color='r', linewidth=2)
    else:
        abs_coords = pred_coords * 149 + 149 // 2
        plt.scatter(abs_coords[0], abs_coords[1], s=80, edgecolors='r', facecolors='none', linewidth=2)
    plt.plot(0, 0, 149, 149, color='r', linewidth=2)


def plot_dataset(dataset, model=None, n=10, show_heatmap=False):
    """
    Plot dataset with/without model predictions, and with/without heatmaps.
    If model is omitted, then any shown heatmap and prediction marker (red) are indicative
    of the ground-truth heatmap. Otherwise they are derived from the model output.

    Blue circles show ground-truth coordinates.
    Red circles show predicted coordinates.
    Perfect alignment can be seen by the (slightly smaller) red circles sitting
    perfectly ringed by the blue circle.

    Args:
     - dataset: original full dataset
    """
    if show_heatmap:
        cols = min(6, n)
        rows = math.ceil((n * 2) / cols)
        plt.figure(figsize=(12, rows * 2), layout='constrained')
    else:
        cols = min(5, n)
        rows = math.ceil(n / cols)
        plt.figure(figsize=(10, rows * 2.5), layout='constrained')
    taken = dataset.take(n)

    if model is not None:
        predicted = model.predict(taken.map(to_input_dataset).batch(n))
        if predicted.shape[1:] != (2,):
            predicted_coords = weighted_peak_coordinates(predicted)
            predicted_coords = predicted_coords[:, 0, :]  # remove superfluous C dimension
        else:
            predicted_coords = predicted

        j = 0
        for i, (image, corner_coords, heatmap) in enumerate(taken):
            j += 1
            # if output.shape == (2,):
            #  corner_coords = output
            # else:
            #  corner_coords = weighted_peak_coordinates(tf.expand_dims(output, axis=0))
            #  corner_coords = corner_coords[0, 0, :]
            pred_coords = predicted_coords[i, ...]

            plt.subplot(rows, cols, j)
            _plot_dataset_entry(image, corner_coords)
            _plotadd_pred_coord(pred_coords)

            has_heatmap = (predicted[i, ...].shape != (2,))
            if has_heatmap and show_heatmap:
                j += 1
                plt.subplot(rows, cols, j)
                plt.imshow(predicted[i], cmap='gray')
                plt.axis('off')

    else:
        j = 0
        for i, (image, corner_coords, heatmap) in enumerate(taken):
            j += 1
            plt.subplot(rows, cols, j)
            plt.title(f"({corner_coords[0]:.3f},{corner_coords[1]:.3f})")
            _plot_dataset_entry(image, corner_coords)

            if show_heatmap:
                j += 1
                pred_coords = weighted_peak_coordinates(tf.expand_dims(heatmap, axis=0))
                pred_coords = pred_coords[0, 0, :]

                plt.subplot(rows, cols, j)
                plt.title("GT heatmap")
                _plot_dataset_entry(heatmap, corner_coords)
                _plotadd_pred_coord(pred_coords)
    plt.show()


def generate_training_image(x, y, width=149, height=149):
    """
    Generates the input image for training and validation.
    Note: only works for integer (x,y) because OpenCV.line() doesn't support sub-pixel resolution.
    """
    img = np.zeros((height, width), dtype=np.uint8)

    # Add speckled noise
    noise_density = 0.1  # Adjust for more/less speckles
    num_noise_pixels = int(noise_density * img.size)
    xs = np.random.randint(0, width, num_noise_pixels)
    ys = np.random.randint(0, height, num_noise_pixels)
    img[ys, xs] = 255

    # Draw vertical line (down from (x, y))
    cv2.line(img, (x, y), (x, height - 1), 255, thickness=3)

    # Draw horizontal line (right from (x, y))
    cv2.line(img, (x, y), (width - 1, y), 255, thickness=3)

    # Rescale value to 0.0 .. 1.0
    # Add channel dimension for grayscale
    return img[..., np.newaxis] / 255.0


def generate_soft_3x3(x_fp, y_fp, coordinate_radius=1.0):
    """
    Used to construct the ground-truth heatmap for a given coordinate.
    This function computes just the 3x3 grid surrounding the given coordinate,
    with values such that its weighted sum of coordinates resolves as the requested
    coordinate (with sub-pixel precision and accuracy).

    Generate a normalized 3x3 kernel centered around (x_fp, y_fp), relative to pixel centers.
    The output weights sum to 1.

    Computes weights as the percentage coverage by a partially enlarged square pixel placed
    exactly onto the floating-point-precision coordinate.
    Experiments found that this approach produces the most accurate results compared to some approaches.
    See https://github.com/malcolmlett/robovac/blob/main/experiments-slam/Experiment_ADLO_3a_SpatialAccuracy.ipynb for other experiments.

    Note: assumes that pixel integer coordinates are located at the pixel centres.
    This is consistent with OpenCV and matplotlib.

    Args:
      coordinate_radius - must be in range 0.0..1.0, where 0.5 = normal pixel size.
        Works best with 1.0.
    """
    def row_coverage(fp):
        row = np.array([-1, 0, +1])
        overlap_left = np.maximum(row - 0.5, fp - coordinate_radius)
        overlap_right = np.minimum(row + 0.5, fp + coordinate_radius)
        overlap = np.clip(overlap_right - overlap_left, 0.0, 1.0)
        return overlap

    # Convert (x_fp, y_fp) to grid coordinate system
    # - assumes grid is placed such that (x_fp,y_fp) is within bounds of central pixel (ie: in range -0.5..+0.5)
    x_fp = x_fp - round(x_fp)
    y_fp = y_fp - round(y_fp)

    # Compute weights
    # - doing x and y axis separately, and then combining into a grid
    dx = row_coverage(x_fp)
    dy = row_coverage(y_fp)
    weights = np.matmul(dy[:, np.newaxis], dx[np.newaxis, :])

    # Normalize weights and return
    weights /= np.sum(weights)
    return weights  # shape (3, 3)


def generate_heatmap_image(x, y, width=149, height=149):
    """
    Generates a heatmap that is mostly zeros, with a 3x3 spot at the target
    coordinate that identifies its exact location to sub-pixel resolution.
    The weighted sum of the coordinates of the non-zero positions,
    weighted by the heatmap magnitudes, will result in exactly the original coordinate.
    """
    img = np.zeros((height, width), dtype=np.float32)
    weights = generate_soft_3x3(x, y)

    xc = int(np.round(x))
    yc = int(np.round(y))

    for dy in range(-1, 2):
        for dx in range(-1, 2):
            xi = xc + dx
            yi = yc + dy
            if 0 <= xi < width and 0 <= yi < height:
                img[yi, xi] = weights[dy + 1, dx + 1]

    return img[..., np.newaxis]


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


def quick_show_preds(model, dataset, width=149, height=149):
    """
    Shows a summary of the model against a dataset.
    Args:
        - model - trained or untrained model
        - dataset - raw dataset (without batching)
        - width - used for conversion between units
        - height - used for conversion between units
    """
    # gather true and predicted coordinates in both unit-scale and pixels
    dataset_pred_coords = []
    dataset_true_coords_px = []
    dataset_pred_coords_px = []
    sizes = np.array([[height, width]])
    for (input_images, true_coords, _) in tqdm.tqdm(dataset.batch(32)):
        y_preds = model(input_images)
        if len(y_preds.shape) == 4:
            # (B, H, W, C)
            coord_preds = weighted_peak_coordinates(y_preds, system='unit-scale')  # (B, 1, 2)
            coord_preds_px = weighted_peak_coordinates(y_preds, system='pixels')   # (B, 1, 2)
            coord_preds = tf.reshape(coord_preds, (-1, 2))        # (B, 2)
            coord_preds_px = tf.reshape(coord_preds_px, (-1, 2))  # (B, 2)
        elif len(y_preds.shape) == 2:
            # (B, 2)
            coord_preds = y_preds
            coord_preds_px = y_preds * sizes + sizes // 2
        else:
            raise ValueError(f"Unrecognised y_pred shape: {y_preds.shape}")

        dataset_pred_coords.extend(coord_preds)
        dataset_pred_coords_px.extend(coord_preds_px)
        dataset_true_coords_px.extend(true_coords * sizes + sizes // 2)
    dataset_pred_coords = np.array(dataset_pred_coords)
    dataset_pred_coords_px = np.array(dataset_pred_coords_px)
    dataset_true_coords_px = np.array(dataset_true_coords_px)

    # compute error distances (in pixels)
    coord_errors_px = np.square(dataset_pred_coords_px - dataset_true_coords_px)  # (D,2)
    coord_errors_px = np.sqrt(tf.reduce_sum(coord_errors_px, axis=-1))  # (D,)

    print(f"pred.x: {np.min(dataset_pred_coords[:, 0])}..{np.max(dataset_pred_coords[:, 0])}")
    print(f"pred.y: {np.min(dataset_pred_coords[:, 1])}..{np.max(dataset_pred_coords[:, 1])}")

    plt.figure(figsize=(10, 2), layout='constrained')

    plt.subplot(1, 3, 1)
    plt.title("coord distance error")
    plt.hist(coord_errors_px, bins=20)
    plt.xlabel('pixels')
    plt.ylabel('histogram')
    plt.gca().tick_params(axis='y', which='both', length=0, labelleft=False)

    plt.subplot(1, 3, 2)
    plt.title("pred.x")
    plt.hist(dataset_pred_coords[:, 0], bins=20)
    plt.xlabel('raw')
    plt.ylabel('histogram')
    plt.gca().tick_params(axis='y', which='both', length=0, labelleft=False)

    plt.subplot(1, 3, 3)
    plt.title("pred.y")
    plt.hist(dataset_pred_coords[:, 1], bins=20)
    plt.xlabel('raw')
    plt.ylabel('histogram')
    plt.gca().tick_params(axis='y', which='both', length=0, labelleft=False)

    plt.show()


def get_layer_outputs(model, dataset, num=1, return_input=False):
    """
    Invokes the model once with a small set of data and obtain the raw outputs from each layer.
    Returns:
      model_output, [layer_outputs]
    """

    # prepare model
    monitoring_model = tf.keras.Model(
        inputs=tinstr._original_inputs(model),
        outputs=[model.outputs] + [layer.output for layer in model.layers])

    # do prediction
    dataset = dataset.take(num).batch(num)
    model_input = next(iter(dataset))[0]
    out = monitoring_model(model_input)
    model_output = out[0][0]
    layer_outputs = out[1:]  # includes last layer, which is usually equivalent to the model output

    if return_input:
        return model_input, model_output, layer_outputs
    else:
        return model_output, layer_outputs


def plot_layers(model, dataset, layer_filter, coord_layer_filter=None, coord_channels=None, num=1):
    """
    Plots images representing outputs from feature layers, and optionally from associated coordinate channels
    or layers.
    Args:
      coord_channels - number of channels in main selected layers that are treated as coordinates and displayed separately
      coord_layer_filter - separate filter for layers that hold coordinates
    """
    def index_layers_by_name_pattern(model, name_pattern):
        return [l_idx for l_idx, layer in enumerate(model.layers) if re.fullmatch(name_pattern, layer.name)]

    def coords_to_rgb(coord_grid):
        # assumes two channels for coordinates, but will produce results even if there's more
        coord_grid = coord_grid + 0.5
        blue = 1.0 - tf.sqrt(tf.reduce_sum(tf.square(coord_grid), axis=-1))
        blue = tf.clip_by_value(blue, 0.0, 1.0)  # gives a stronger distinction than RMS or simple mean
        return tf.stack([coord_grid[..., 0], coord_grid[..., 1], blue], axis=-1)

    def flatten_channels(tensor, reduction='pos-norm'):
        if reduction == 'norm':
            return tf.sqrt(tf.reduce_mean(tf.square(tensor), axis=-1))
        elif reduction == 'pos-norm':
            tensor = tf.clip_by_value(tensor, 0.0, float('inf'))
            return tf.sqrt(tf.reduce_mean(tf.square(tensor), axis=-1))
        else:
            raise ValueError(f"Invalid reduction: {reduction}")

    layer_indices = index_layers_by_name_pattern(model, layer_filter)
    coord_layer_indices = None if coord_layer_filter is None else index_layers_by_name_pattern(model, coord_layer_filter)
    has_coords = (coord_channels is not None and coord_channels > 0) or (coord_layer_indices is not None)

    # do prediction
    model_input, model_output, layer_outputs = get_layer_outputs(model, dataset, num=num, return_input=True)
    print(f"model input:     {model_input.shape}")
    print(f"model output:    {model_output.shape}")
    print(f"all layers:      {[t.shape for t in layer_outputs]}")
    print(f"filtered layers: {[layer_outputs[l_idx].shape for l_idx in layer_indices]}")
    if coord_layer_indices is not None:
        print(f"coord layers:    {[layer_outputs[l_idx].shape for l_idx in coord_layer_indices]}")

    if coord_layer_indices is not None:
        cols = 1 + max(len(layer_indices), len(coord_layer_indices))
    else:
        cols = 1 + len(layer_indices)
    rows = num * 2 if has_coords else num

    plt.figure(figsize=(15, 2 * rows), layout='constrained')

    for idx in range(num):
        row = idx * 2 if has_coords else idx
        col = 0
        plt.subplot(rows, cols, 1 + (row * cols) + col)
        plt.title(f"input (data #{idx})")
        plt.imshow(model_input[idx, :, :, 0])
        plt.axis('off')

        # feature layers
        for l_idx in layer_indices:
            layer = model.layers[l_idx]
            layer_data = layer_outputs[l_idx]
            if coord_channels is not None:
                # drop coord_channels from main feature layer
                layer_data = layer_data[..., :-coord_channels]
            col += 1
            plt.subplot(rows, cols, 1 + (row * cols) + col)
            plt.title(f"{layer.name}")
            plt.imshow(flatten_channels(layer_data[idx]))
            plt.axis('off')

        # coordinate channels or layers
        if has_coords:
            row = idx * 2 + 1
            col = 0

            # coordinate channel
            if coord_channels is not None and coord_channels > 0:
                for l_idx in layer_indices:
                    layer = model.layers[l_idx]
                    layer_data = layer_outputs[l_idx]
                    col += 1
                    plt.subplot(rows, cols, 1 + (row * cols) + col)
                    plt.imshow(coords_to_rgb(layer_data[idx, :, :, -coord_channels:]))
                    plt.axis('off')

            # coordinate layers
            else:
                for l_idx in coord_layer_indices:
                    layer = model.layers[l_idx]
                    layer_data = layer_outputs[l_idx]
                    col += 1
                    plt.subplot(rows, cols, 1 + (row * cols) + col)
                    plt.title(f"{layer.name}")
                    plt.imshow(coords_to_rgb(layer_data[idx]))
                    plt.axis('off')


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


class StrideGrid2D(layers.Layer):
    """
    Inspired by the CoordGrid2D, but designed to be more convenient for strided spatial pooling.
    The computed positions are relative to the stride window, rather than the global image.

    Returns a (batch_size, height, width, 2) tensor with a grid of (x,y) coordinates.
    Coordinates are aligned to the strides of an assumed subsequent pooling layer,
    and indicate the relative position of the cell in the pool, relative to the
    pool's centre.
    For example, with stride=2, each 2x2 sub-grid has coordinates:
       [[-0.5,-0.5], [+0.5,-0.5]],
        [-0.5,+0.5], [+0.5,+0.5]].

    Generally this will be subsequently concatenated with the input tensor
    before applying a position-wise pooling operation.
    """
    def __init__(self, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.strides = (strides, strides) if np.isscalar(strides) else strides

    def call(self, inputs):
        # inputs.shape => [batch_size, height, width, channels]
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))

        # Generate a coordinate grid patch:
        # - x_coords: shape (stride_width,) from -0.5 to 0.5
        # - y_coords: shape (stride_height,) from -0.5 to 0.5
        # - xx, yy shape => (stride_height, stride_width)
        # - concatted: (stride_height, stride_width, 2)
        stride_height, stride_width = self.strides
        x_coords = tf.linspace(-0.5, 0.5, stride_width)
        y_coords = tf.linspace(-0.5, 0.5, stride_height)
        xx, yy = tf.meshgrid(x_coords, y_coords)
        grid = tf.concat([tf.expand_dims(xx, axis=-1), tf.expand_dims(yy, axis=-1)], axis=-1)

        # Tile across all patches in image
        # - Tile, then pad if needed
        # - (width, height, 2)
        patch_count_x = width // stride_width
        patch_count_y = height // stride_height
        x_pad = tf.cast(width - patch_count_x * stride_width, tf.int32)
        y_pad = tf.cast(height - patch_count_y * stride_height, tf.int32)
        tiled = tf.tile(grid, [patch_count_x, patch_count_y, 1])
        paddings = tf.stack([
            tf.stack([0, y_pad]),
            tf.stack([0, x_pad]),
            tf.stack([0, 0])
        ])
        tiled = tf.pad(tiled, paddings, 'CONSTANT')

        # Tile across batch dimension
        output = tf.tile(tf.expand_dims(tiled, 0), [batch_size, 1, 1, 1])
        return output


class PositionwiseMaxPool2D(tf.keras.layers.Layer):
    """
    Inspired by max-pool, however, where max-pool treats each channel separately,
    this groups all features together for a given 2D position. This in theory allows for more
    direct transport of spatial information from a CoordGrid2D or StrideGrid2D input.

    Pooling operation that selects all channels from the same position as a single unit.
    Positions are chosen by arg-max over a reduction operation (sum-of-squares).

    Use channel_weights to exclude some channels from the reduction.
    For example, if the first 32 channels are from semantic features, while the last 2 channels
    come from a CoordGrid2D input, then given the first 32 channels a weight of 1.0 each, and the last 2 channels a weight of 0.0.
    This way only the semantic features are arg-maxed over. Otherwise the arg-max operation
    will be biased towards the position with a larger coordinate value.
    """

    def __init__(self, pool_size=(2, 2), strides=None, channel_weights=None, **kwargs):
        super().__init__(**kwargs)

        self.pool_size = (pool_size, pool_size) if np.isscalar(pool_size) else pool_size
        if strides is None:
          self.strides = self.pool_size
        else:
          self.strides = (strides, strides) if np.isscalar(strides) else strides
        self.channel_weights = channel_weights

    def call(self, inputs):
        batch_size, height, width, channels = tf.unstack(tf.shape(inputs))
        ksize = self.pool_size
        strides = self.strides

        # Extract 2x2 patches: shape (B, H//2, W//2, 2, 2, C)
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, strides[0], strides[1], 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_shape = tf.shape(patches)
        out_h, out_w = patch_shape[1], patch_shape[2]

        # Reshape to (B, out_h, out_w, 4, C)
        patches = tf.reshape(patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])

        # Compute reduction: (B, out_h, out_w, 4)
        # - using sum-of-squares, which is equivalent to a norm() once argmax is applied
        if self.channel_weights is None:
          norms = tf.reduce_sum(tf.square(patches), axis=-1)
        else:
          norms = tf.reduce_sum(tf.square(patches) * self.channel_weights, axis=-1)

        # Argmax to find winning positions: (B, out_h, out_w)
        indices = tf.argmax(norms, axis=-1, output_type=tf.int32)

        # Gather the full vectors corresponding to max-norm positions
        one_hot = tf.one_hot(indices, depth=ksize[0]*ksize[1])  # shape: (B, out_h, out_w, 4)
        one_hot = tf.expand_dims(one_hot, axis=-1)              # shape: (B, out_h, out_w, 4, 1)
        output = tf.reduce_sum(one_hot * patches, axis=-2)      # shape: (B, out_h, out_w, C)

        return output


class AttentionPool2D(tf.keras.layers.Layer):
    """
    An alternative to MaxPool2D that tries to improve on gradient flow
    by using a soft-argmax operation.

    2D downsampling layer that computes a soft-argmax from one input to attend to the second input.
    Typically used for softmax-weighted pooling of positional inputs, based
    on the strength of the feature vectors.

    Both inputs must have the same shape (B, H, W, C). Operates channel-wise.
    Channels from each input are paired up, in sequence.

    Output shape: (B, pooled_H, pooled_W, C).
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super().__init__(**kwargs)

        self.pool_size = (pool_size, pool_size) if np.isscalar(pool_size) else pool_size
        if strides is None:
          self.strides = self.pool_size
        else:
          self.strides = (strides, strides) if np.isscalar(strides) else strides
        self.padding = padding.upper()

    def call(self, keys, values):
        batch_size, height, width, channels = tf.unstack(tf.shape(keys))
        tf.assert_equal(tf.shape(keys), tf.shape(values), "Input tensors must have identical shape", summarize=4)
        ksize = self.pool_size

        # Extract pool patches, eg: (assuming 2x2) (B, H//2, W//2, 2, 2, C)
        key_patches = tf.image.extract_patches(
            images=keys,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        value_patches = tf.image.extract_patches(
            images=values,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        patch_shape = tf.shape(key_patches)
        out_h, out_w = patch_shape[1], patch_shape[2]

        # Reshape to (B, out_h, out_w, 4, C)
        key_patches   = tf.reshape(key_patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])
        value_patches = tf.reshape(value_patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])

        # Compute softmax-weighted mean of values, shape: (B, out_h, out_w, C)
        attention_scores = tf.nn.softmax(key_patches, axis=-2)
        output = tf.reduce_sum(attention_scores * value_patches, axis=-2)
        return output


class StridedSoftmax2D(tf.keras.layers.Layer):
    """
    Applies softmax to values within stride-patches.
    Treats channels independently.

    Used in conjunction with DotPool2D() to breaks out the steps of AttentionPool2D for easier interpretability.
    However, this broken-out form is limited in its flexibility. It requires that stride patches are exactly adjacent
    and non-overlapping.
    """

    def __init__(self, pool_size=(2, 2), strides=None, **kwargs):
        super().__init__(**kwargs)

        self.pool_size = (pool_size, pool_size) if np.isscalar(pool_size) else pool_size
        if strides is None:
          self.strides = self.pool_size
        else:
          self.strides = (strides, strides) if np.isscalar(strides) else strides
        self.padding = 'valid'.upper()  # can only do if using 'valid' padding

        if self.pool_size != self.strides:
            raise ValueError("pool_size and strides must be the same")

    def call(self, input):
        batch_size, height, width, channels = tf.unstack(tf.shape(input))
        ksize = self.pool_size

        # Extract pool patches, eg: (assuming 2x2) (B, H//2, W//2, 2, 2, C)
        # Then reshape to (B, out_h, out_w, 4, C)
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        patch_shape = tf.shape(patches)
        out_h, out_w = patch_shape[1], patch_shape[2]
        patches = tf.reshape(patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])

        # Compute softmax for each patch
        patches = tf.nn.softmax(patches, axis=-2)

        # Convert back to original shape
        # - only possible when strides=pool_size and padding=valid
        output = tf.reshape(patches, [batch_size, out_h, out_w, ksize[0], ksize[1], channels])
        output = tf.transpose(output, [0, 1, 3, 2, 4, 5])  # eg: (B, H//2, 2, W//2, 2, C)
        output = tf.reshape(output, [batch_size, height, width, channels])

        return output


class DotPool2D(tf.keras.layers.Layer):
    """
    2D downsampling layer that combines the values between two inputs via a flattened dot-product
    applied patch-wise and channel-wise. Worded differently, it applies a weighted sum,
    where the first input provides the weights and the second input provides the values.
    However, you could equally phrase them the other way around.

    Both inputs must have the same shape (B, H, W, C). Operates channel-wise.
    Channels from each input are paired up, in sequence.

    Output shape: (B, pooled_H, pooled_W, C).

    Used in conjunction with StridedSoftmax2D() to breaks out the steps of AttentionPool2D for easier interpretability.
    However, this broken-out form is limited in its flexibility. It requires that stride patches are exactly adjacent
    and non-overlapping.
    """

    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super().__init__(**kwargs)

        self.pool_size = (pool_size, pool_size) if np.isscalar(pool_size) else pool_size
        if strides is None:
          self.strides = self.pool_size
        else:
          self.strides = (strides, strides) if np.isscalar(strides) else strides
        self.padding = padding.upper()

    def call(self, keys, values):
        batch_size, height, width, channels = tf.unstack(tf.shape(keys))
        tf.assert_equal(tf.shape(keys), tf.shape(values), "Input tensors must have identical shape", summarize=4)
        ksize = self.pool_size

        # Extract pool patches, eg: (assuming 2x2) (B, H//2, W//2, 2, 2, C)
        key_patches = tf.image.extract_patches(
            images=keys,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        value_patches = tf.image.extract_patches(
            images=values,
            sizes=[1, ksize[0], ksize[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        patch_shape = tf.shape(key_patches)
        out_h, out_w = patch_shape[1], patch_shape[2]

        # Reshape to (B, out_h, out_w, 4, C)
        key_patches   = tf.reshape(key_patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])
        value_patches = tf.reshape(value_patches, [batch_size, out_h, out_w, ksize[0]*ksize[1], channels])

        # Compute dotted sum of values, shape: (B, out_h, out_w, C)
        output = tf.reduce_sum(key_patches * value_patches, axis=-2)
        return output


class MeanCoordError(tf.keras.metrics.Metric):
    """
    Computes the distance between true and predicted coordinates, in pixels, possibly after conversion
    from whatever output representation the model uses (eg: heatmap).
    """

    def __init__(self, encoding, name="mce", system='unit-scale', width=149, height=149, **kwargs):
        """
        Args:
          encoding = one of:
            "xy" - model outputs (B,2) of (x,y) coordinates in range +/- 0.5 as fraction of image width/height.
            "heatmap-peak" - model outputs (B,H,W,1) of heatmap, where 3x3 grid surrounding peak is used as weighted mean of pixel coordinates
          system - 'unit-scale' or 'pixels'.
            The models are dataset assume a 'unit-scale' where the coordinates are given
            as a fraction of the image width/height, relative to the image centre
            and each have the range -0.5 .. +0.5.

            The alternative is the original pixel coordinates, with floating-point precision.
          width - assumed image width when using 'xy' encoding and pixel system
          height - assumed image width when using 'xy' encoding and pixel system
        """
        super().__init__(name=name, **kwargs)
        self.encoding = encoding
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
        self.system = system
        self.sizes = tf.constant([[height, width]], dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # get true and pred coords in desired coordinate system
        if self.encoding == "xy":
            coord_true = y_true  # (B, 2)
            coord_pred = y_pred  # (B, 2)
            # convert to pixels if needed
            if self.system == 'pixels':
                coord_true = coord_true * self.sizes + self.sizes // 2  # (B, 2)
                coord_pred = coord_pred * self.sizes + self.sizes // 2  # (B, 2)
        elif self.encoding == "heatmap-peak":
            coord_true = weighted_peak_coordinates(y_true, system=self.system)  # (B, 2)
            coord_pred = weighted_peak_coordinates(y_pred, system=self.system)  # (B, 2)
        else:
            raise ValueError(f"Unrecognised output encoding: {self.encoding}")

        # calculate distances
        coord_errors = tf.square(coord_true - coord_pred)  # (B,2)
        coord_errors = tf.sqrt(tf.reduce_sum(coord_errors, axis=-1))  # (B,)

        self.total.assign_add(tf.reduce_sum(coord_errors))
        self.count.assign_add(tf.cast(tf.size(coord_errors), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


class _PeakWeightedLoss(tf.keras.losses.Loss):
    """
    Proto-loss class to be a part of a larger loss class via composition.

    Computes loss only near a peak (target or predicted). Applies a gaussian distributed weighting
    to all pixels, but tightly focused on the target peak.
    """

    def __init__(self, peak_source, sigma=2.0, disc_radius=None, debug=False):
        # Default disc radius found through trial and error to work for small and large sigmas.
        # With sigma=2.0, this pulls in just a little more than the 3x3 grid at flat weighting.
        super().__init__()
        if peak_source not in ['y_true', 'y_pred']:
            raise ValueError(f"Invalid peak_source: {peak_source}")
        self.peak_source = peak_source
        self.sigma = sigma
        self.disc_radius = tf.sqrt(sigma) * 1.5 if disc_radius is None else disc_radius
        self.debug = debug

    def call(self, y_true, y_pred):
        """
        Args:
          y_true/y_pred - (B, H, W, C)
        Returns:
          (B,) - loss for each sample in batch
        """
        B, H, W, C = tf.unstack(tf.shape(y_true))

        # identify peak coordinates from source
        peak_source = y_true if self.peak_source == "y_true" else y_pred
        x_peaks = tf.reduce_max(peak_source, axis=1)  # (B, W, C)
        x_peaks = tf.argmax(x_peaks, axis=1, output_type=tf.int32)  # (B, C)
        y_peaks = tf.reduce_max(peak_source, axis=2)  # (B, H, C)
        y_peaks = tf.argmax(y_peaks, axis=1, output_type=tf.int32)  # (B, C)
        x_peaks = tf.reshape(x_peaks, [B, 1, 1, C])  # (B, 1, 1, C)
        y_peaks = tf.reshape(y_peaks, [B, 1, 1, C])  # (B, 1, 1, C)

        # populate coordinate grid
        grid_x, grid_y = tf.meshgrid(tf.range(W), tf.range(H))  # (H, W)
        grid_x = tf.tile(tf.reshape(grid_x, [1, H, W, 1]), [B, 1, 1, C])  # (B, H, W, C)
        grid_y = tf.tile(tf.reshape(grid_y, [1, H, W, 1]), [B, 1, 1, C])  # (B, H, W, C)

        # compute square-distance grid
        dist_x = grid_x - x_peaks
        dist_y = grid_y - y_peaks
        dist2 = tf.cast(dist_x ** 2 + dist_y ** 2, tf.float32)  # (B, H, W, C)

        # compute weights
        #  - compute as gaussian of square-dist
        #  - then push gaussian up so that it's 1.0 @ disc_radius, and clipped in the middle to 1.0
        #  - total: max between them, then normalized as weights
        #  - this ensures that the existing gaussian shape of the 3x3 patch in y_true is given 100% equal weights across,
        #    but that there's also a little bubble around it pushing those values towards zero, while retaining
        #    smooth transitions.
        weights = tf.exp(-dist2 / (2 * self.sigma ** 2))  # (B, H, W, C) in range 0.0 .. 1.0
        edge = tf.exp(-self.disc_radius ** 2 / (2 * self.sigma ** 2))
        weights = tf.clip_by_value(weights / edge, 0.0, 1.0)
        scale = tf.reduce_sum(weights, axis=[1, 2], keepdims=True)  # (B, 1, 1, C)
        weights = weights / scale  # (B, H, W, C) normalised to sum to 1.0 for each (B, C)

        # DEBUG MODE - plot and log workings
        if self.debug:
            print(f"dist2: {dist2.shape} - {np.min(dist2)}..{np.max(dist2)}, weights: {weights.shape}")
            plt.figure(figsize=(10, 2), layout='constrained')
            plt.subplot(1, 4, 1)
            plt.imshow(tf.abs(dist_x[0]))
            plt.subplot(1, 4, 2)
            plt.imshow(tf.abs(dist_y[0]))
            plt.subplot(1, 4, 3)
            plt.imshow(dist2[0])
            plt.subplot(1, 4, 4)
            plt.imshow(weights[0])

        # calculate weighted MSE loss
        # - note: weights already normalised to computed weighted mean, so use sum to calculate mean
        error = tf.square(y_true - y_pred)  # (B, H, W, C)
        loss = tf.reduce_sum(error * weights, axis=[1, 2, 3])  # (B,)
        return loss


class HeatmapPeakCoordLoss(tf.keras.losses.Loss):
    """
    Loss calculated through three weighted components:
    - global MSE
    - MSE of area near true peak
    - MSE of area near predicted peak
    """

    def __init__(self, name="hpc_loss", component_weights=None, sigma=2.0, disc_radius=None):
        """
        Args:
          component_weights - 3-array of weights of the three loss components (will be normalized to sum to 1.0)
        """
        # Default disc radius found through trial and error to work for small and large sigmas.
        # With sigma=2.0, this pulls in just a little more than the 3x3 grid at flat weighting.
        super().__init__(name=name)
        if component_weights is not None and np.shape(component_weights) != (3,):
            raise ValueError(f"Wrong shape for component_weights: {component_weights}")
        if component_weights is None:
            component_weights = [1.0, 1.0, 1.0]
        component_weights = tf.constant(component_weights, dtype=tf.float32)
        self.component_weights = component_weights / tf.reduce_sum(component_weights)

        self.true_peak_losser = _PeakWeightedLoss("y_true", sigma=sigma, disc_radius=disc_radius)
        self.pred_peak_losser = _PeakWeightedLoss("y_pred", sigma=sigma, disc_radius=disc_radius)

    def call(self, y_true, y_pred):
        """
        Args:
          y_true/y_pred - (B, H, W, C)
        Returns:
          (B,) - loss for each sample in batch
        """
        image_wide_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3])  # (B,)
        true_peak_loss = self.true_peak_losser(y_true, y_pred)
        pred_peak_loss = self.pred_peak_losser(y_true, y_pred)

        return self.component_weights[0] * image_wide_loss + \
            self.component_weights[1] * true_peak_loss + \
            self.component_weights[2] * pred_peak_loss


def get_flops(model):
    """
    Uses TensorFlow profile to compute the FLOPS of a model.

    Internally this uses a TensorFlow v1 API and issues a warning.
    Ideally we'd use keras-flops, but there's an outstanding issue that prevents it
    from working: https://github.com/tokusumi/keras-flops/issues/17.
    Args:
        model - must have a defined input shape
    Returns:
        number of FLOPS
    """
    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
    graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
    flops = graph_info.total_float_ops
    return flops


class TrialHistory:
    """
    Properties:
        trial_keys - array
        metrics - dict with:
            mean_{metric-name}" - array of 'metric-name' results, at best epoch for each trial,
              averaged over all executions for each trial_key
            sd_{metric-name}" - array of std. dev. of 'metric-name' results, at best epoch for each trial,
              over all executions for each trial_key
        histories - array (by trial) of array (by execution) of History
        param_counts - array (by trial) of model parameter counts (trainable + non-trainable)
        flops - array (by trial) of model FLOPS (if computable)
    """
    def __init__(self, trial_keys, executions_per_trial):
        self.trial_keys = trial_keys
        self.executions_per_trial = executions_per_trial
        self.metrics = {}  # dict (by f"mean-{metric-key}" or f"sd-{metric-key}") of array (by trial, etc.)
        self.histories = []  # array (by trial) of array (by execution) of History
        self.param_counts = []
        self.flops = []

    @property
    def metrics_dataframe(self):
        return pd.DataFrame(self.metrics, index=pd.Index(self.trial_keys, name="trial_keys"))


def run_trials(trial_keys, trial_fn, objective='loss', scoring='last', executions_per_trial=1):
    """
    For each trial key, instantiates a model, runs a fixed number of epochs, and collects the best achieved metrics
    (minimum loss/error).

    Inspired by KerasTuner, but more appropriate for what we're trying to do.
    KerasTuner is focused on finding the single set of hyperparameter values that had the best results,
    whereas we want to get results across a grid of hyperparameter values.

    Args:
      trial_keys: list of values to pass to trial_fn, one call per value
      trial_fn: function that executes a fit with the given trial_keys and returns the result. The result
        may be formatted as one of:
            - History object
            - tuple(model, History)
      objective: metric to measure
      scoring: one of: 'last', 'min', 'max'. Determines how the "best" step is chosen.
      executions_per_trial: number of executions of trial_fn with same trial_key, to average over.

    Returns:
      TrialHistory instance
    """

    def get_best_iteration(history):
        if scoring == 'last':
            return len(history.history[objective]) - 1
        elif scoring == 'max':
            return np.argmax(history.history[objective])
        elif scoring == 'min':
            return np.argmin(history.history[objective])
        else:
            raise ValueError(f"Invalid scoring: '{scoring}'")

    def dict_array_append(dic, key, value):
        if key in dic:
            dic[key].append(value)
        else:
            dic[key] = [value]

    def standardise_trail_response(trial_fn_res):
        if isinstance(trial_fn_res, tuple):
            return trial_fn_res
        else:
            history = trial_fn_res
            return (None, history)

    # setup
    res = TrialHistory(trial_keys, executions_per_trial)
    start = time.perf_counter()

    # run a number of trials, one for each trial_key
    for trial_idx, trial_key in enumerate(trial_keys):
        idisplay.clear_output(wait=True)
        print(f"Trial {trial_idx + 1}/{len(trial_keys)}: trial_key={trial_key}")
        print(f"Time taken so far: {time.perf_counter() - start:.1f}s")
        print()

        # run a number of executions for each trial
        histories = []
        metrics_data = None
        for _ in range(executions_per_trial):
            model, history = standardise_trail_response(trial_fn(trial_key))
            best_it = get_best_iteration(history)
            histories.append(history)
            if metrics_data is None:
                metrics_data = {key: [] for key in history.history}
            for key in history.history:
                metrics_data[key].append(history.history[key][best_it])

        # get model stats
        if model is not None:
            param_count = model.count_params()  # trainable + non-trainable
            flops = get_flops(model)
            res.param_counts.append(param_count)
            res.flops.append(flops)

        # collect stats over trials
        res.histories.append(histories)
        for key in metrics_data:
            dict_array_append(res.metrics, f"mean_{key}", np.mean(metrics_data[key]))
            dict_array_append(res.metrics, f"sd_{key}", np.std(metrics_data[key]))
            dict_array_append(res.metrics, f"min_{key}", np.min(metrics_data[key]))
            dict_array_append(res.metrics, f"max_{key}", np.max(metrics_data[key]))

    # cleanup
    for key in res.metrics:
        res.metrics[key] = np.array(res.metrics[key])

    # summarise progress
    idisplay.clear_output(wait=True)
    print(f"Trial {len(trial_keys)}/{len(trial_keys)}: ✓ complete")
    print(f"Time taken: {time.perf_counter() - start:.1f}s")

    return res


def plot_trials(trials, trial_key_name="trial key", br_yscale='linear', br_ymax=None):
    """
    Plot result of run_trails()
    """
    def _plot_best_trials(metric='mean_loss', error_metric='sd_loss', xscale='log', yscale='linear', ymax=None):
        if error_metric is not None:
            plt.errorbar(trials.trial_keys, trials.metrics[metric], yerr=trials.metrics[error_metric])
        else:
            plt.plot(trials.trial_keys, trials.metrics[metric])
        plt.yscale(yscale)
        plt.xscale(xscale)
        plt.gca().set_xticks(trials.trial_keys)
        plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.gca().tick_params(axis='x', which='minor', length=0, labelbottom=False)
        # plt.minorticks_off()
        plt.ylim([0, ymax])

    def _plot_trial_histories(metric='loss', yscale='linear'):
        for trial_key, histories in zip(trials.trial_keys, trials.histories):
            losses = [history.history[metric] for history in histories]
            losses = np.mean(losses, axis=0)
            plt.plot(losses, label=trial_key)
        plt.legend()
        plt.yscale(yscale)
        plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))  # ensure integer tick marks
        if yscale == 'linear':
            plt.ylim([0, None])

    plt.figure(figsize=(15, 5), layout='constrained')

    # Best Results - Training MPE
    plt.subplot(2, 3, 1)
    plt.title("Best results (train set)")
    _plot_best_trials('mean_mpe', 'sd_mpe', yscale=br_yscale, ymax=br_ymax)
    plt.xlabel(trial_key_name)
    plt.ylabel("mean-pixel-error")

    # Best Results - Validation MPE
    plt.subplot(2, 3, 2)
    plt.title("Best results (val set)")
    _plot_best_trials('mean_val_mpe', 'sd_val_mpe', yscale=br_yscale, ymax=br_ymax)
    plt.xlabel(trial_key_name)
    plt.ylabel("mean-pixel-error")

    # Model - FLOPS
    if trials.param_counts or trials.flops:
        plt.subplot(2, 3, 3)
        plt.title("Model Size")
        if trials.param_counts:
            plt.plot(trials.trial_keys, trials.param_counts, label='param_counts')
        if trials.flops:
            plt.plot(trials.trial_keys, trials.flops, label='flops')
        plt.xlabel(trial_key_name)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.gca().set_xticks(trials.trial_keys)
        plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
        plt.gca().tick_params(axis='x', which='minor', length=0, labelbottom=False)

    # Histories - Training MPE
    plt.subplot(2, 3, 4)
    plt.title("Training curves (train set)")
    _plot_trial_histories(metric='mpe', yscale='log')
    plt.ylabel("mean-pixel-error")
    plt.xlabel("epoch")

    # Histories - Validation MPE
    plt.subplot(2, 3, 5)
    plt.title("Training curves (val set)")
    _plot_trial_histories(metric='val_mpe', yscale='log')
    plt.ylabel("mean-pixel-error")
    plt.xlabel("epoch")
