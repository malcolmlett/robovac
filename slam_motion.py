# Animates motion of the SLAM agent.

# TODO rename to "slam_operations" so I can use it for static application of the slam model etc.

import lds
import slam_data

import numpy as np
import cv2
import math
import tensorflow as tf


def predict_at_location(full_map, known_map, known_map_start, model, location, orientation, **kwargs):
    """

    Args:
        full_map: semantic map of entire floorplan
            Used for LDS map generation.
        known_map: known semantic map
            Used for input map generation.
        known_map_start: (x,y), units: physical
            Coordinate of top-left corner of known_map
        model: model for semantic map prediction.
            Or omit to exclude semantic_maps from the output.
        location: (x,y) float, units: physical
        orientation: float, unit: radians

    Keyword args:
        pixel_size: the usual
        max_distance: the usual

    Return:
        (map_prediction, accept, delta (x,y), delta orientation)
        - map_prediction: tensor (H,W,3)
        - accept: bool
        - delta (x,y): (2,) float numpy array, units: physical
        - delta orientation: float, unit: radians
    """
    # Config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)

    # Coords
    # - just like in generate_training_data_sample(), we align to map pixels
    #   when rendering
    window_size_px = np.ceil(max_distance / pixel_size).astype(int) * 2 + 1
    window_radius_px = (window_size_px - 1) // 2
    window_size_px = np.array([window_size_px, window_size_px])

    # Extract input map from known map
    # - align to known_map pixels, and fill the rest with 'unknown'
    unknown_value = np.zeros(slam_data.__CLASSES__, dtype=np.float32)
    unknown_value[slam_data.__UNKNOWN_IDX__] = 1
    map_window = tf.tile(unknown_value, multiples=(window_size_px, window_size_px, 1))

    centre_fpx = (location - known_map_start) / pixel_size
    start_fpx = centre_fpx - window_radius_px
    start_px = np.round(start_fpx).astype(int)
    loc_alignment_offset_fpx = start_fpx - start_px
    known_map_indices, map_window_indices = slam_data.get_intersect_ranges_tf(
        tf.shape(known_map), tf.shape(map_window), start_px)
    tf.tensor_scatter_nd_update(map_window, map_window_indices, tf.gather_nd(known_map, known_map_indices))

    # Generate LDS input from full map
    # - re-use input alignment
    occupancy_map = full_map[..., slam_data.__OBSTRUCTION_IDX__]
    ranges = lds.lds_sample(occupancy_map, location, orientation, **kwargs)
    lds_map = lds.lds_to_occupancy_map(
        ranges, angle=orientation, size_px=window_size_px, centre_px=loc_alignment_offset_fpx, pixel_size=pixel_size)

    # Do prediction
    # - turn into TF batch
    map_input = tf.expand_dims(map_window, axis=0)
    lds_input = tf.expand_dims(lds_map, axis=0)
    (map_output, adlo_output) = model.predict((map_input, lds_input))

    # Convert to result
    # TODO cast to float32
    map_pred = map_output[0]
    accept = adlo_output[0][0] >= 0.5
    delta_location = adlo_output[0][1:3] * window_size_px * pixel_size
    delta_orientation = adlo_output[0][3] * np.pi

    return map_pred, accept, delta_location, delta_orientation


def get_contour_pxcoords(file, **kwargs):
    """
    Extracts the raw trajectory contour from the image file.
    Args:
        file: filename
    Keyword args:
        colour = 3-list, tuple, or array, int in range 0..255
            default: [127, 127, 127]
    Returns:
        (N,2) x int - contour coordinates in pixels relative to the image
    """
    colour = np.array(kwargs.get('colour', [127, 127, 127]))

    # Load the image
    image = cv2.imread(file)

    # Define the color range for the path (replace with actual BGR values)
    lower_bound = colour-1  # lower BGR bound of the path color
    upper_bound = colour+1  # upper BGR bound of the path color

    # Create a mask to isolate the path based on the color range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours of the path
    # - use CHAIN_APPROX_NONE in order to get coords for every pixel
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # I think it's a tuple containing one entry for each counter
    # Then for each counter we have: (max_len, ?=1, (x,y))

    # return as just an array of (x, y) coordinates along the path
    return contours[0][:, 0, :]


def sample_trajectory(px_coords, **kwargs):
    """
    Generates a pseudo-realistic path of motion around the contour drawn on the floorplan.
    At each point of the trajectory, the agent is assumed to be at a particular physical (x,y) coordinate and facing
    in the direction of next motion.
    Args:
      px_coords: (N,2) x int - contour coordinates in pixels relative to the image
    Returns:
      (coords, orientations) - trajectory coordinates in physical units, and orientations in radians.
    """
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    step_size = kwargs.get('step_size', pixel_size * 10)

    # for convenience, we'll work in px units for most of this

    # sample index pairs: start + target target
    step_size_fpx = step_size / pixel_size  # floating-point accuracy
    indices1 = np.round(np.arange(0, len(px_coords), step_size_fpx)).astype(np.int32)
    indices2 = indices1 + math.floor(step_size_fpx)
    if indices2[-1] >= len(px_coords):
        indices2[-1] = len(px_coords) - 1

    # get px coords
    trajectory_px_coords = px_coords[indices1]
    target_px_coords = px_coords[indices2]

    # compute angles
    dy = target_px_coords[:, 1] - trajectory_px_coords[:, 1]
    dx = target_px_coords[:, 0] - trajectory_px_coords[:, 0]
    orientations = np.arctan2(dy, dx)

    # return in physical units
    trajectory_coords = trajectory_px_coords * pixel_size
    return trajectory_coords, orientations
