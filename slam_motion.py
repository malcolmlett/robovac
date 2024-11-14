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
    centre_fpx = (location - known_map_start) / pixel_size
    start_fpx = centre_fpx - window_radius_px
    start_px = np.round(start_fpx).astype(np.int32)
    loc_alignment_offset_fpx = start_fpx - start_px

    map_window = slam_data.unknown_map(window_size_px)
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
    map_pred = tf.math.softmax(map_output[0], axis=-1)
    accept = adlo_output[0][0] >= 0.5
    delta_location = (adlo_output[0][1:3] * window_size_px * pixel_size).astype(np.float32)
    delta_orientation = (adlo_output[0][3] * np.pi).astype(np.float32)

    return map_pred, accept, delta_location, delta_orientation


# The logic in here is very similar to that of slam_data.combine_semantic_maps(), particularly the maths,
# however there are different optimisations at work, so it's hard to combine them.
# Still best to keep them in sync though.
def update_map(semantic_map, semantic_map_start, update_map, update_map_centre, **kwargs):
    """
    Overlays a single "update window" map onto an existing map, potentially updating the extents
    of the existing map. Generally used for adding a predicted map window onto a larger known map.

    The combination is more than a naive average. It acknowledges that
    an "unknown" state from one semantic map simply means that it hasn't "observed"
    the state at a given position, while an "unknown" state for the same position
    across all semantic maps suggests that the position may indeed be "unobservable".
    Then applies a weighted average to resolve discrepancies between floor vs wall,
    with samples given weights according to how much they agree with the
    observable/unobservable outcome.

    Furthermore, it acknowledges that the prediction is accurate only within a certain radius.
    The rectangular or square predicted map is given a mask that divides its radius into 3 bands and has:
    * uniform 100% within its inner-most two bands,
    * 100..0% gradient within its outer band, and
    * 0% in the corners beyond its outer band.

    Args:
        semantic_map: (H,W,3)
            Semantic map to update.
        semantic_map_start: (2,) float, units: physical
            Coordinate of centre of top-left pixel on semantic_map.
        update_map: (H,W,3)
            New section of map to apply onto semantic_map.
            Usually a predicted semantic map for the observed window around the agent.
        update_map_centre: (2,) float, units: physical
            Coordinates of exact centre of update_map under the same absolute
            coordinate system as semantic_map_start.
            Usually the location of agent where it generated the predicted map from.
            Internally this is rounded in order to align update_map onto the pixels
            of semantic_map, without doing sub-pixel translations or anti-aliasing.

    Keyword args:
        pixel_size: usual meaning
        update_mode: one of 'merge', 'mask-merge', default: 'mask-merge'
            Not yet supported.

    Returns:
        (semantic_map, location_start) with semantic map containing the combined results,
        and the physical location of centre of top-left pixel of the newly updated map.
    """
    # TODO apply a centre-surround mask

    # note: internally this functions uses pixel coordinates throughout,
    # unless otherwise stated

    # config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    window_size_px = tf.gather(tf.shape(update_map), (0, 1))  # (H,W,3) -> [w,h]
    window_radius_px = window_size_px // 2

    # convert coordinates to pixels relative to input semantic map
    # - in physical units: update_map_offset
    #    = (update_centre - window_radius) - semantic_map_start
    #    = (update_centre - window_radius_px * PIXEL_SIZE) - semantic_map_start
    # - then divide everything by PIXEL_SIZE and re-arrange
    update_map_offset_px = np.round((update_map_centre - semantic_map_start) / pixel_size - window_radius_px)\
        .astype(np.int32)

    # calculate dimensions of new output map
    current_size_px = tf.gather(tf.shape(semantic_map), (1, 0))  # (2,) x int32 = (w,h)
    out_min_extent = tf.minimum([0, 0], update_map_offset_px)  # (2,) x int32 = (x,y)
    out_max_extent = tf.maximum(current_size_px, current_size_px + update_map_offset_px + window_size_px)
    out_shape = tf.gather(out_max_extent - out_min_extent, (1, 0))  # (h,w)
    location_start = semantic_map_start + tf.cast(out_min_extent, tf.float32) * pixel_size  # (2,) x float32

    # Initialize tensors for accumulation
    max_observed = tf.zeros(out_shape, dtype=tf.float32)
    sum_observed = tf.zeros(out_shape, dtype=tf.float32)
    out_sum = tf.zeros(tf.concat([out_shape, tf.constant([3], tf.int32)], axis=0), dtype=tf.float32)

    # compute max and sums across entire map
    # also compute sum over samples
    def _add_map(this_map, start_px, _out_sum, _sum_observed, _max_observed):
        observed = tf.reduce_sum(tf.gather(this_map, indices=(0, 1), axis=-1), axis=-1)  # (H,W) x 0..1 prob
        out_indices, map_indices = slam_data.get_intersect_ranges_tf(out_shape, tf.shape(this_map), start_px)

        # Accumulate values
        # sum{i} p(floor|Si)
        # sum{i} p(wall|Si)
        # out_sum[out_slices] = np.add(out_sum[out_slices], this_map[map_slices])
        _out_sum = tf.tensor_scatter_nd_add(_out_sum, out_indices, tf.gather_nd(this_map, map_indices))

        # sum p(observed|Sj) = normalizer
        # sum_observed[out_slices] = np.add(sum_observed[out_slices], observed[map_slices])
        _sum_observed = tf.tensor_scatter_nd_add(_sum_observed, out_indices, tf.gather_nd(observed, map_indices))

        # max p(observed|Sk) = p(observable)
        # max_observed[out_slices] = np.maximum(max_observed[out_slices], observed[map_slices])
        _max_observed = tf.tensor_scatter_nd_max(_max_observed, out_indices, tf.gather_nd(observed, map_indices))

        return _out_sum, _sum_observed, _max_observed

    # Add source and updated map onto output
    out_sum, sum_observed, max_observed = _add_map(
        semantic_map, -out_min_extent, out_sum, sum_observed, max_observed)
    out_sum, sum_observed, max_observed = _add_map(
        semantic_map, -out_min_extent, out_sum, sum_observed, max_observed)

    # Compute the final output map
    # - p(floor|observable) = sum{i} p(floor|Si) / sum p(observed|Sj) * max p(observed|Sk)
    # - p(wall |observable) = sum{i} p(wall |Si) / sum p(observed|Sj) * max p(observed|Sk)
    sum_observed = tf.where(sum_observed == 0, tf.constant(1e-8, dtype=tf.float32), sum_observed)  # avoid div-by-zero
    out = out_sum / sum_observed[..., tf.newaxis] * max_observed[..., tf.newaxis]

    # - p(unobservable) = 1 - max{i} p(observed|Si)
    out_unknown = tf.expand_dims(1 - max_observed, axis=-1)

    # combine
    out = tf.cast(tf.concat([out[..., 0:2], out_unknown], axis=-1), tf.float32)

    return out, location_start


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
