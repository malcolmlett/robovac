# Operation-time logic for SLAM.
# Includes functions for animating use of the SLAM agent.

import lds
import slam_data

import numpy as np
import tensorflow as tf
import cv2
import math
import os
import IPython.display as idisplay
import matplotlib.pyplot as plt
import imageio.v3 as iio
import tqdm
import time


def load_trajectory_pxcoords(path, **kwargs):
    """
    Loads the full resolution trajectory contour from the provided image file (usually a floorplan).
    For semantic map extraction, see the `slam_data` module.

    Args:
        path: image file path
    Keyword args:
        colour = 3-list, tuple, or array, int in range 0..255
            default: [127, 127, 127]
    Returns:
        (N,2) x int - contour coordinates in pixels relative to the image
    """
    colour = np.array(kwargs.get('colour', [128, 128, 128]))

    # Load the image
    image = cv2.imread(path)

    # Define the color range for the path (replace with actual BGR values)
    lower_bound = colour-1  # lower BGR bound of the path color
    upper_bound = colour+1  # upper BGR bound of the path color

    # Create a mask to isolate the path based on the color range
    mask = cv2.inRange(image, lower_bound, upper_bound)

    # Find contours of the path
    # - use CHAIN_APPROX_NONE in order to get coords for every pixel
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Returns a list of contours (usually just 1), and then for each contour we have: (n, 1, 2).
    # This seems to represent: (contour_len, ?=1, [x,y])

    # return as just an array of (x, y) coordinates along the path
    return contours[0][:, 0, :]


def sample_trajectory(px_coords, **kwargs):
    """
    Generates a pseudo-realistic path of motion around the contour drawn on the floorplan.
    At each point of the trajectory, the agent is assumed to be at a particular physical (x,y) coordinate and facing
    in the direction of next motion.

    Args:
        px_coords: (N,2) x int - contour coordinates in pixels relative to the image
    Keyword args:
        step_size: float, units: physical, default: equivalent of 10 pixels
            How far to move in each step.
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
        (map_prediction, accept, delta (x,y), delta orientation), (input_map, lds_input)
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
    map_window = tf.tensor_scatter_nd_update(map_window, map_window_indices, tf.gather_nd(known_map, known_map_indices))

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
    (map_output, adlo_output) = model.predict((map_input, lds_input), verbose=0)

    # Convert to result
    map_pred = tf.math.softmax(map_output[0], axis=-1)
    accept = adlo_output[0][0] >= 0.5
    delta_location = (adlo_output[0][1:3] * window_size_px * pixel_size).astype(np.float32)
    delta_orientation = (adlo_output[0][3] * np.pi).astype(np.float32)

    return (map_pred, accept, delta_location, delta_orientation), (map_window, lds_map)


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

    # note: internally this functions uses pixel coordinates throughout,
    # unless otherwise stated

    # config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    update_mode = kwargs.get('update_mode', 'mask-merge')
    window_size_px = tf.gather(tf.shape(update_map), (0, 1))  # (H,W,3) -> [w,h]
    window_radius_px = window_size_px // 2

    # convert coordinates to pixels relative to input semantic map
    # - in physical units: update_map_offset
    #    = (update_centre - window_radius) - semantic_map_start
    #    = (update_centre - window_radius_px * PIXEL_SIZE) - semantic_map_start
    # - then divide everything by PIXEL_SIZE and re-arrange
    update_map_offset_px = np.round((update_map_centre - semantic_map_start) / pixel_size).astype(np.int32)\
                           - window_radius_px

    # calculate dimensions of new output map
    current_size_px = tf.gather(tf.shape(semantic_map), (1, 0))  # (2,) x int32 = (w,h)
    out_min_extent = tf.minimum([0, 0], update_map_offset_px)  # (2,) x int32 = (x,y)
    out_max_extent = tf.maximum(current_size_px, update_map_offset_px + window_size_px)
    out_shape = tf.gather(out_max_extent - out_min_extent, (1, 0))  # (h,w)
    location_start = semantic_map_start + tf.cast(out_min_extent, tf.float32) * pixel_size  # (2,) x float32

    # Revise update_map location relative to output map
    update_map_offset_px -= out_min_extent

    # Initialize tensors for accumulation
    max_observed = tf.zeros(out_shape, dtype=tf.float32)
    sum_observed = tf.zeros(out_shape, dtype=tf.float32)
    out_sum = tf.zeros(tf.concat([out_shape, tf.constant([3], tf.int32)], axis=0), dtype=tf.float32)

    # Compute weight mask
    if update_mode == 'merge':
        mask = None
    elif update_mode == 'mask-merge':
        mask = create_map_update_weight_mask(window_size_px)
    else:
        raise ValueError(f"Unknown update_mode: {update_mode}")

    # compute max and sums across entire map
    # also compute sum over samples
    def _add_map(this_map, start_px, mask, _out_sum, _sum_observed, _max_observed):
        observed = tf.reduce_sum(tf.gather(this_map, indices=(0, 1), axis=-1), axis=-1)  # (H,W) x 0..1 prob
        out_indices, map_indices = slam_data.get_intersect_ranges_tf(out_shape, tf.shape(this_map), start_px)

        if mask is None:
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
        else:
            # Same as above but with the equivalent of the following being performed BEFORE the above
            # - for the update_map, the mask makes the outer edges all 'unknown' and graduates towards the inner
            #   disk being as certain as it originally was.
            # - for the semantic_map, the opposite is true, with its inner disk becoming fully 'unknown'
            # - the latter causes the existing _max_observed to be scaled down, and then we just do a normal
            #   max on the result, solving how to do a "weighted max".
            mask_3d = tf.expand_dims(mask, axis=-1)
            masked_this_map = this_map * mask_3d + slam_data.unknown_map(window_size_px) * (1-mask_3d)  # (H,W,3)
            masked_observed = observed * mask  # (H,W)

            masked_out_sum = tf.gather_nd(_out_sum, out_indices) * (1-mask_3d) +\
                                          slam_data.unknown_map(window_size_px) * mask_3d
            _out_sum = tf.tensor_scatter_nd_update(
                  _out_sum, out_indices,
                  masked_out_sum + masked_this_map)

            masked_sum_observed = tf.gather_nd(_sum_observed, out_indices) * (1-mask)
            _sum_observed = tf.tensor_scatter_nd_update(
                  _sum_observed, out_indices,
                  masked_sum_observed + masked_observed)

            # weighted-max: a weird concept
            # in the middle of the overlap between the gradient sections,
            # the masks make the highest max = 0.5, so divide by that to scale back
            masked_max_observed = tf.gather_nd(_max_observed, out_indices) * (1-mask)
            rescale = tf.maximum(mask, 1-mask)
            _max_observed = tf.tensor_scatter_nd_update(
                  _max_observed, out_indices,
                  tf.maximum(masked_max_observed, masked_observed) / rescale)
        return _out_sum, _sum_observed, _max_observed

    # Add source and updated map onto output
    out_sum, sum_observed, max_observed = _add_map(
        semantic_map, -out_min_extent, None, out_sum, sum_observed, max_observed)
    out_sum, sum_observed, max_observed = _add_map(
        update_map, update_map_offset_px, mask, out_sum, sum_observed, max_observed)

    # Compute the final output map
    # - p(floor|observable) = sum{i} p(floor|Si) / sum p(observed) * max p(observed)
    # - p(wall |observable) = sum{i} p(wall |Si) / sum p(observed) * max p(observed)
    sum_observed = tf.where(sum_observed == 0, tf.constant(1e-8, dtype=tf.float32), sum_observed)  # avoid div-by-zero
    out = out_sum / sum_observed[..., tf.newaxis] * max_observed[..., tf.newaxis]

    # - p(unobservable) = 1 - max{i} p(observed|Si)
    out_unknown = tf.expand_dims(1 - max_observed, axis=-1)

    # combine
    out = tf.cast(tf.concat([out[..., 0:2], out_unknown], axis=-1), tf.float32)

    # clip
    # - the above maths is a little unstable and sometimes produces values outside of range
    # - I'm just applying a simplistic approach of clipping, on the basis that it does the least damage
    out = tf.clip_by_value(out, 0.0, 1.0)

    return out, location_start


def create_map_update_weight_mask(window_size_px):
    """
    Computes a weight mask for a given window size for controlling how new map information
    is merged into an existing map. The central part of the new map is assumed to replace
    the existing map, while the outer edges should have no changes applied.

    The mask is generated such that the outer edges are all zero, and the circular disc
    inner part is divided into three bands as follows:
    - inner 2 bands: 100%
    - outer 1 band: linear gradient with 100% at inner edge, and 0% ot outer edge
    Args:
        window_size_px: (x,y), int
    Returns:
        weight tensor (H,W) x float in range 0.0 to 1.0
    """

    # Create a grid of (x, y) coordinates
    window_size_px = tf.cast(window_size_px, dtype=tf.float32)
    x = tf.range(window_size_px[0], dtype=tf.float32)
    y = tf.range(window_size_px[1], dtype=tf.float32)
    x_grid, y_grid = tf.meshgrid(x, y)

    # Calculate distances from the center of the mask
    max_dist_px = (tf.reduce_min(window_size_px)-1) / 2
    centre_px = (window_size_px-1) / 2
    distances = tf.sqrt((x_grid - centre_px[0])**2 + (y_grid - centre_px[1])**2)

    # Normalize distances by max_distance to get values in the range [0, 1]
    normalized_distances = distances / max_dist_px

    # Create the mask
    inner_band = tf.where(normalized_distances <= 2/3, 1.0, 0.0)
    outer_band = tf.where(
        (normalized_distances > 2/3) & (normalized_distances <= 1),
        1.0 - (normalized_distances - 2/3) * 3,
        0.0
    )

    # Combine the bands into a final mask
    mask = tf.maximum(inner_band, outer_band)
    return tf.cast(mask, tf.float32)


def show_trajectory(floorplan, coords, **kwargs):
    """
    Visualise a trajectory as a static plot overlaid onto the floorplan.

    Args:
        floorplan: semantic map to overlay trajectory
        coords: list or array, (N,2), units: physical
            x,y coordinates of agent at each step along trajectory
    Keyword args:
        pixel_size: usual meaning and default.
    """

    # config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)

    # convert to px units
    coords = coords / pixel_size

    plt.imshow(floorplan)
    plt.axis('off')
    plt.plot(coords[:, 0], coords[:, 1], c='k', linewidth=1)
    plt.show()


def animate_trajectory(floorplan, coords, angles, filename=None, **kwargs):
    """
    Visualise a trajectory as an animation of the agent moving around the floorplan.

    Args:
        floorplan: semantic map to overlay trajectory
        coords: list or array, (N,2), units: physical
            x,y coordinates of agent at each step along trajectory
        angles: list or array, (N,), units: radians
            orientation of agent at each step along trajectory
        filename: string, optional
            File to save animation to. gif format only.
            Otherwise attempts to animate directly on-screen.
    Keyword args:
        pixel_size: usual meaning and default.
        fps: default 2
            Frames per second.
    """

    # config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    fps = kwargs.get('fps', 2.0)
    os.makedirs("data", exist_ok=True)

    # convert to px units
    coords = coords / pixel_size

    if filename is not None:
        print(f"Generating animation and saving to: {filename}")

    frames = []
    for i in tqdm.tqdm(range(coords.shape[0])):
        angle = angles[i]
        loc1 = coords[i, :]
        loc2 = loc1 + np.array([np.cos(angle), np.sin(angle)]) * 5

        if filename is None:
            idisplay.clear_output(wait=True)
        plt.imshow(floorplan)
        plt.axis('off')

        # plot trajectory taken
        plt.plot(coords[0:(i + 1), 0], coords[0:(i + 1), 1], c='k', linewidth=1)

        # plot current location
        plt.scatter(loc1[0], loc1[1], c='m')
        plt.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 'm-')

        if filename is None:
            plt.show()
            time.sleep(1/fps)
        else:
            frame_filename = "data/frame.png"
            plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            frames.append(iio.imread(frame_filename))

    if filename is not None:
        iio.imwrite(filename, frames, fps=fps)
        print()
        print(f"Animation saved to: {filename}")

        # show animation
        idisplay.display(idisplay.Image(filename))


def animate_slam(floorplan, locations, orientations, model, filename=None, **kwargs):
    """
    Visualise the SLAM agent in action as it navigates around a floorplan and
    builds up a global map.

    The generated animation includes two versions of the map.
    The first is based on the ground-truth floorplan.
    The second is based on the constructed global map.
    The ground-truth trajectory and estimated trajectory are projected onto both.

    Args:
        floorplan: ground-truth semantic map
        locations: list or array, (N,2), units: physical
            x,y coordinates of agent at each step along trajectory
        orientations: list or array, (N,), units: radians
            orientation of agent at each step along trajectory
        model: the trained SLAM model to use
        filename: string, optional
            File to save animation to. gif format only.
            Otherwise attempts to animate directly on-screen.

    Keyword args:
        pixel_size: usual meaning and default.
        state_follows: one of 'true', 'pred', default: 'pred'
            When updating map, use the ground-truth ('true') location and orientation tracking,
            or the model-adjusted ('pred') location and orientation tracking.
        map_update_mode: 'update_mode' passed to update_map()
        fps: default 2
            Frames per second.
        clip: boolean, default: true
            If true, clips generated maps and images to the bounds of the ground-truth map,
            providing for a cleaner display and less movement of the maps. Any agent trajectories
            that extend beyond the bounds of the maps will be omitted from display.
            If false, the generated global map will be larger than the ground-truth map,
            and the full trajectories are always shown. The map images may have to "zoom out" in order to
            compensate.
            Forced true when saving to file.
    """

    # config
    state_follows = kwargs.get('state_follows', 'true')
    map_update_mode = kwargs.get('map_update_mode', 'mask-merge')
    fps = kwargs.get('fps', 2.0)
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    do_clip = kwargs.get('clip', True)
    if filename is not None:
        # force to use clipping because otherwise the frame size changes over time and can't save into a gif
        do_clip = True

    # init animation
    os.makedirs("data", exist_ok=True)
    frames = []
    if filename is not None:
        print(f"Generating animation and saving to: {filename}")

    # initial map is unknown but for the sake of the animation we'll use the same shape as floorplan
    # TODO look at whether it's possible to start with a small one
    global_map_shape = tf.gather(floorplan.shape, (1, 0))
    global_map = slam_data.unknown_map(global_map_shape)
    global_map_start = np.array([0, 0])

    # move through trajectory
    true_trajectory = []
    pred_trajectory = []
    prev_true_location = locations[0]
    prev_true_angle = orientations[0]
    pred_location = prev_true_location.copy()
    pred_angle = prev_true_angle.copy()
    for i in tqdm.tqdm(range(locations.shape[0])):
        # move agent
        true_location = locations[i]
        true_angle = orientations[i]
        true_location_movement = true_location - prev_true_location
        true_angle_movement = true_angle - prev_true_angle
        prev_true_location = true_location
        prev_true_angle = true_angle

        # update estimated state from movement
        pred_location += true_location_movement  # (note: does in-place update)
        pred_angle += true_angle_movement        # (note: does in-place update)

        # do prediction and map update
        est_location = pred_location if state_follows == "pred" else true_location
        est_angle = pred_angle if state_follows == "pred" else true_angle
        (map_pred, accept, delta_location, delta_orientation), (input_map, input_lds) = predict_at_location(
            floorplan, global_map, global_map_start, model, est_location, est_angle)

        # do map update
        global_map, global_map_start = update_map(
            global_map, global_map_start, map_pred, true_location, update_mode=map_update_mode)

        # revise estimated state from prediction
        pred_location += delta_location  # (note: does in-place update)
        pred_angle += delta_orientation  # (note: does in-place update)

        # track trajectories
        true_trajectory.append(true_location)
        pred_trajectory.append(pred_location.copy())  # clone needed due to in-place updates

        # add to animation
        if filename is None:
            idisplay.clear_output(wait=True)
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 5, (1, 2))
        _show_state_against_true_map(floorplan, true_location, true_angle, true_trajectory,
                                     accept, pred_location, pred_angle, pred_trajectory, pixel_size, do_clip)
        plt.subplot(1, 5, 3)
        plt.imshow(input_lds, cmap='gray')
        plt.axis('off')
        plt.subplot(1, 5, (4, 5))
        _show_state_against_predicted_map(global_map, global_map_start, floorplan.shape, pixel_size, do_clip)

        if filename is None:
            plt.show()
            # don't bother to sleep, as the model execution introduces enough delay
        else:
            frame_filename = "data/frame.png"
            plt.savefig(frame_filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            frames.append(iio.imread(frame_filename))

            # in case user stops generation early, save gif every 50 frames
            if i % 50 == 0:
                iio.imwrite(filename, frames, fps=fps)

    if filename is not None:
        iio.imwrite(filename, frames, fps=fps)
        print()
        print(f"Animation saved to: {filename}")

        # show animation
        idisplay.display(idisplay.Image(filename))


def _show_state_against_true_map(floorplan, true_location, true_angle, true_trajectory,
                                 pred_accept, pred_location, pred_orientation, pred_trajectory,
                                 pixel_size, do_clip):
    true_loc = true_location / pixel_size
    true_angle_loc = true_loc + np.array([np.cos(true_angle), np.sin(true_angle)]) * 10

    pred_loc = pred_location / pixel_size
    pred_angle_loc = pred_loc + np.array([np.cos(pred_orientation), np.sin(pred_orientation)]) * 10

    plt.imshow(floorplan)
    plt.axis('off')
    if do_clip:
        plt.xlim(0, floorplan.shape[1])
        plt.ylim(floorplan.shape[0], 0)  # Reverse y-axis for correct orientation when plotting onto an image

    def plot_trajectory(trajectory, c):
        trajectory = np.array(trajectory) / pixel_size
        for i in range(trajectory.shape[0] - 1):
            plt.plot(trajectory[i:i + 2, 0], trajectory[i:i + 2, 1], c=c, linewidth=1, clip_on=do_clip)

    plot_trajectory(true_trajectory, 'k')
    plot_trajectory(pred_trajectory, 'm')

    plt.scatter(true_loc[0], true_loc[1], c='k', s=50)
    plt.plot([true_loc[0], true_angle_loc[0]], [true_loc[1], true_angle_loc[1]], c='k')
    plt.scatter(pred_loc[0], pred_loc[1], c='m', s=50)
    plt.plot([pred_loc[0], pred_angle_loc[0]], [pred_loc[1], pred_angle_loc[1]], c='m')


def _show_state_against_predicted_map(global_map, global_map_start, clip_shape, pixel_size, do_clip):
    if do_clip:
        # clip map to same size and shape as floorplan
        start = tf.cast(tf.maximum(tf.round(-global_map_start / pixel_size), [0, 0]), tf.int32)
        size = tf.gather(clip_shape, (1, 0))
        end = start + size
        global_map = global_map[start[1]:end[1], start[0]:end[0]]

    plt.imshow(global_map)
    plt.axis('off')
