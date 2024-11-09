# Creates training data for the SLAM model.
#
# Datasets are represented as follows:
#    Dataset ((input_maps, lds_maps), (output_maps, adlos), metadatas)
# where:
#   input_maps    - input semantic map from current estimated position (B,H,W,C)
#                   - blank if current location unknown
#   lds_maps      - input occupancy map from LDS (B,H,W)
#   output_maps   - output semantic map, blank if don't care
#   adlos         - output ADLO
#   metadatas     - array, must be removed before training but is used elsewhere
#
# Semantic maps are encoded as follows:
#  - shape: (H,W,C)
#  - type: float, logits/softmax
#  - channels:
#     [0] = floor (white in RGB image encoding)
#     [1] = obstruction (black in RGB image encoding)
#     [2] = unknown (grey in RGB image encoding)
#  - filled with all values "unknown" to indicate "unknown" or "don't care" state.
#
# ADLO is encoded as follows:
#  - shape: (4,)
#  - type: float
#  - [0]: accept: logit/sigmoid - likelihood that the agent is present within the map, based on the LDS data
#  - [1]: delta x: -0.5 .. +0.5 - percentage of window width to add to current estimated x-location
#  - [2]: delta y: -0.5 .. +0.5 - percentage of window width to add to current estimated y-location
#  - [3]: delta angle: -1.0 .. +1.0 - fraction of pi (+/-) to add to current estimated orientation
#  - DLO values ignored when accept=false
#  - DLO values all zero when processing a blank input map and accepting the LDS map.
#
# The metadata array is encoded as follows:
#  - shape: (8,)
#  - type: float
#  - [0]: floorplan id - round to nearest int when using
#  - [1]: sample type - round to nearest int when using
#  - [2]: input map centre x, float, units: physical
#  - [3]: input map centre y, float, units: physical
#  - [4]: input map orientation, unit: radians
#  - [5]: agent x, float, units: physical
#  - [6]: agent y, float, units: physical
#  - [7]: agent orientation, unit: radians

import lds
import map_from_lds_train_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm
from typing import Any


# constants
__CLASSES__ = 3
__FLOOR_IDX__ = 0
__OBSTRUCTION_IDX__ = 1
__UNKNOWN_IDX__ = 2


# TODO rename to image_to_semantic_map
def one_hot_encode_floorplan(image):
    """
    Converts an RGB floorplan image into a semantic-map: a one-hot encoded tensor of the same form
    used as input and output maps in the SLAM model.

    Ordered channels are:
    - floor (white in the RGB image)
    - obstruction (black in the RGB image)
    - unknown (grey in the RGB image)

    Args:
      image: (H,W,3) RGB floorplan image

    Returns:
      (H,W,C) one-hot encoded tensor of the floorplan image
    """
    # sanity check
    if not np.array_equal(np.unique(image), np.array([0, 192, 255])):
        raise ValueError(f"Encountered unexpected values in image, expected [0, 192, 255], got: {np.unique(image)}")

    # get each channel
    floor_mask = tf.reduce_all(tf.equal(image, [255, 255, 255]), axis=-1)
    obstruction_mask = tf.reduce_all(tf.equal(image, [0, 0, 0]), axis=-1)
    unknown_mask = tf.reduce_all(tf.equal(image, [192, 192, 192]), axis=-1)

    # Stack the masks along the last dimension to create a one-hot encoded tensor
    one_hot_image = tf.stack([floor_mask, obstruction_mask, unknown_mask], axis=-1)

    return tf.cast(one_hot_image, tf.float32)


def generate_training_data(semantic_map, num_samples=5, **kwargs):
    """
    Generates a number of samples of training data.

    Args:
        semantic_map: (H,W,C)
        num_samples: int
    Keyword args:
        pixel_size: default __PIXEL_SIZE__
        max_distance: default __MAX_DISTANCE__
        sample_types: int, tuple, list or array
            Collection of sample types to generate from.
            Allowed values include: 0, 1, 2, 3.
        mask_output_map_by_sample_type: tuple, list or array
            Default: [False, False, True, False].
            Whether to blank out the ground-truth output map for any
            sample types. This draws a balance between the training data accurately
            reflecting how the model is intended to be used, and avoiding training
            the model against ambiguous requirements, versus providing more data for training.
            If masking out both sample types 3 and 4, map generation training speed is reduced.
            But sample type 3 in particular is very ambiguous with whether the model should
            predict a map aligned to the input map's orientation or to the LDS orientation.
        model: default None
            Model to use for generating input maps.
        predicted_samples: tuple (locations, semantic_maps).
            Pre-generated samples of semantic map predictions across the full map.
            Provide at-most one of this or 'model'.
        model_weight: float in range 0.0 to 1.0 (inclusive).
            Default: 1.0 if model is present, 0.0 otherwise.
            If provided, the generated input maps will be a weighted average
            between samples taken directly from the floorplan and maps
            predicted by the model.
    Returns:
         Dataset ((input_maps, lds_maps), (output_maps, adlos), metadatas)
    """

    tot_sample_types = 4
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)
    sample_types = np.array(kwargs.get('sample_types', range(tot_sample_types)))
    sample_types = np.ravel(np.array(sample_types))  # cleanup type variations
    mask_output_map_by_sample_type = np.array(kwargs.get('mask_output_map_by_sample_type', [False, False, True, False]))
    model = kwargs.get('model', None)
    model_weight = kwargs.get('model_weight', 1.0 if model is not None else 0.0)
    predicted_sample_locs, predicted_sample_maps = kwargs.get('predicted_samples', (None, None))
    print(f"Generating {num_samples} samples of training data")
    if model is not None or predicted_sample_maps is not None:
        print(f"Input maps: using predicted input maps with weight {model_weight}")
    else:
        print("Input maps: using floorplan samples only")
    print(f"Pixel size: {pixel_size}")
    print(f"Max distance: {max_distance}")
    print(f"Sample types: {sample_types}")
    if np.sum(mask_output_map_by_sample_type) == 0:
        print(f"Mask output for sample types: None")
    else:
        print(f"Mask output for sample types: {np.where(mask_output_map_by_sample_type)}")

    # identify ranges
    # - size of full map (W, H), physical units
    # - size of window (W, H), physical units
    map_range_low, map_range_high = get_location_range(semantic_map, **kwargs)
    window_size = (np.ceil(max_distance / pixel_size).astype(int) * 2 + 1) * pixel_size
    window_range = np.array([window_size, window_size])

    # generate sampling of predicted maps
    if model is not None:
        predicted_sample_locs, _, _, predicted_sample_maps = take_samples_covering_map(
            semantic_map, model, sampling_mode='random')
    kwargs['predicted_samples'] = (predicted_sample_locs, predicted_sample_maps)

    # do data generation
    print("Generating training data...")
    input_maps = []
    lds_maps = []
    output_maps = []
    adlos = []
    metadatas = []
    attempts = 0
    for _ in tqdm.tqdm(range(num_samples)):
        # keep trying until we generate one sample
        while True:
            attempts += 1
            sample_type = np.random.choice(sample_types)
            gen_output_map = not mask_output_map_by_sample_type[sample_type]
            accept = True
            loc_error = (0.0, 0.0)
            angle_error = 0.0

            # Agent true location: on floor
            # - facing any angle
            agent_location = random_floor_location(semantic_map, map_range_low, map_range_high)
            agent_angle = np.random.uniform(-np.pi, np.pi)

            if sample_type == 0:
                # New map, unknown location and orientation, loc/angle error disregarded
                # - blank input map, populated LDS map
                # - populated output map, A=accept, DLO=zeros

                # Map location = Estimated location: unknown
                map_location, map_angle = None, None

                input_map, lds_map, output_map, _ = generate_training_data_sample(
                    semantic_map, agent_location, agent_angle, False, gen_output_map, None, None, **kwargs)

            elif sample_type == 1:
                # Known map, known location/angle estimation with small normal error
                # - populated input map, populated LDS map
                # - populated output map, A=accept, DLO=accurate low non-zero

                # Map location = Estimated location: small normal error from agent location
                # - allowed to be an illegal position
                # - loc_error - normal about centre, std.dev = 10% of range
                # - angle_error - normal about zero, std.dev = 10% of 180 degrees
                loc_error = np.random.normal((0, 0), window_range / 2 * 0.1)
                angle_error = np.random.uniform(0, np.pi * 0.1)
                loc_error = np.clip(loc_error, -window_range / 2, +window_range / 2)
                angle_error = np.clip(angle_error, -np.pi, +np.pi)
                map_location = agent_location - loc_error
                map_angle = agent_angle - angle_error

                input_map, lds_map, output_map, _ = generate_training_data_sample(
                    semantic_map, map_location, map_angle, True, gen_output_map, loc_error, angle_error, **kwargs)

            elif sample_type == 2:
                # Known map, location unknown and searching, with LDS data partially on this map window.
                # - populated input map, populated LDS map
                # - populated or blank output map, A=accept, DLO=approximate high non-zero

                # Map location: independent uniform random location WITHIN window distance
                # - allowed to be in illegal position
                # - loc_error - uniform either side of zero, anywhere within window
                # - angle_error - uniform anywhere within 360-degree range
                loc_error = np.random.uniform(-window_range / 2, +window_range / 2)
                angle_error = np.random.uniform(-np.pi, np.pi)
                map_location = agent_location - loc_error
                map_angle = agent_angle - angle_error

                input_map, lds_map, output_map, _ = generate_training_data_sample(
                    semantic_map, map_location, map_angle, True, gen_output_map, loc_error, angle_error, **kwargs)

            elif sample_type == 3:
                # Known map, location unknown and searching, with LDS not on this map window.
                # LDS and map explicitly in different parts of the floor (max 1/4th overlap between LDS circles)
                # - populated input map, populated LDS map
                # - populated or blank output map, A=reject, DLO=zero, but don't care

                # Map location: independent uniform random location OUTSIDE window distance
                # - allowed to be in illegal position
                # - location - at most only 1/4th of diameter of LDS circles will overlap
                # - angle - uniform anywhere within 360-degree range
                min_distance = max_distance * 1.5
                map_location = None
                while map_location is None or np.linalg.norm(map_location - agent_location) < min_distance:
                    map_location = np.random.uniform(map_range_low, map_range_high)
                map_angle = np.random.uniform(-np.pi, np.pi)
                loc_error = agent_location - map_location
                angle_error = agent_angle - map_angle

                input_map, _, _, _ = generate_training_data_sample(
                    semantic_map, map_location, map_angle, True, False, loc_error, angle_error, **kwargs)
                _, lds_map, output_map, _ = generate_training_data_sample(
                    semantic_map, agent_location, agent_angle, False, gen_output_map, None, None, **kwargs)

                # for expected NN outputs
                # (error should be ignored by loss function, but if we do let it train on these values then we'd
                #  prefer the NN doesn't claim there's a loc error)
                accept = False
                loc_error = (0, 0)
                angle_error = 0

            else:
                raise ValueError(f"unrecognised sample_type: {sample_type}")
            if tot_sample_types < (3 + 1):
                raise ValueError("tot_sample_types needs revising: {tot_sample_types}")

            # emit result
            if lds_map is not None:
                adlo = np.array([
                    1.0 if accept else 0.0,
                    loc_error[0] / window_range[0],  # convert to range: -0.5 .. +0.5
                    loc_error[1] / window_range[1],  # convert to range: -0.5 .. +0.5
                    angle_error / np.pi  # convert to range: -1.0 .. +1.0
                ])
                metadata = np.array([
                    2.0,  # always static floorplan Id for now
                    sample_type,
                    map_location[0] if map_location is not None else np.nan,
                    map_location[1] if map_location is not None else np.nan,
                    0.0 if map_angle is not None else np.nan,  # map always at 0-degrees
                    agent_location[0],
                    agent_location[1],
                    agent_angle
                ])
                input_maps.append(input_map)
                lds_maps.append(lds_map)
                output_maps.append(output_map)
                adlos.append(adlo.astype(np.float32))
                metadatas.append(metadata.astype(np.float32))
                break

    print(f"Generated {len(input_maps)} samples after {attempts} attempts")
    return tf.data.Dataset.from_tensor_slices((
        (input_maps, lds_maps),
        (output_maps, adlos),
        metadatas
    ))


# TODO consider changing API so that it takes: agent true location, agent estimated location
# TODO consider splitting this out into two methods
def generate_training_data_sample(semantic_map, location, orientation, map_known, gen_output_map, location_error,
                                  orientation_error, **kwargs):
    """
    Simulates information that the agent might have and should infer given the agent's true location/orientation
    and the error in its estimate of that location/orientation.

    Can be used to simulate when the agent is moving around an existing known map, by setting map_known=True
    and supplying reasonable location and orientation errors.
    This causes the LDS data to be aligned to the map, depending only on the location and orientation error.
    In this situation, the main orientation parameter only selects where the LDS scan is started from,
    and introduces slight variations in the LDS data collected due to its 1-degree resolution.

    Can be used to simulate when searching the map for the agent's location, by setting orientation_error=np.inf.
    This causes the LDS data to be freely rotated relative to the map.

    Can be used to simulate when the agent has first started in a new area by setting map_known=False,
    in which case the input map is blank and the ground_truth map is rotated to the orientation of the agent.

    :param semantic_map: (H,W,C) = one-hot encoded floorplan
    :param location: (float, float) = (x,y) ground-truth location of agent
    :param orientation: (float) = radians ground-truth angle of orientation of agent
    :param map_known: bool, whether to simulate the agent knowing this section of map or having a blank input map
    :param gen_output_map: boolean, whether to generate the expected output map, or a blank one
    :param location_error: None or float tuple, (delta x, deltay).
       Simulates an error on the agent's estimated location. Causes the LDS map to be offset relative to the map,
       in the opposite direction of the error.
       None for no error.
       Generally this should be None if map_known=False.
    :param orientation_error: None or float, radians.
       Simulates an error on the agent's estimated orientation. Causes the LDS map to be rotated relative to the map,
       in the opposite direction of the error.
       None for no error.
       Inf or NaN for unknown orientation, causing the LDS map to be oriented with the true starting angle shifted to zero degrees.
       Generally this should be None if map_known=False.
    :param kwargs:
    :return:
      (input_map, lds_map, ground_truth_map, centre_offset)
      - input_map: (H,W,C) input window from known semantic map
      - lds_map: (H,W) input LDS as occupancy map
      - ground_truth_map: (H,W,C) expected output semantic map
      - centre_offset: (2,) = (x,y) of float in range (-1.0..1.0, -1.0..1.0) indicating where the exact location
        of the agent is relative to the map centre (pixel units, with sub-pixel resolution)
    """
    # config
    default_unknown_value = np.zeros(__CLASSES__, dtype=semantic_map.dtype)
    default_unknown_value[__UNKNOWN_IDX__] = 1

    max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    unknown_value = kwargs.get('unknown_value', default_unknown_value)
    predicted_sample_locs, predicted_sample_maps = kwargs.get('predicted_samples', (None, None))
    model_weight = kwargs.get('model_weight', 1.0 if predicted_sample_maps is not None else 0.0)

    location_error = np.array(location_error) if location_error is not None else np.array([0.0, 0.0])
    orientation_error = orientation_error if orientation_error is not None else 0.0
    window_size_px = np.ceil(max_distance / pixel_size).astype(int) * 2 + 1
    window_size_px = np.array([window_size_px, window_size_px])

    # take LDS sample
    lds_orientation = (orientation + orientation_error) if np.isfinite(orientation_error) else orientation
    ranges = lds.lds_sample(semantic_map[:, :, __OBSTRUCTION_IDX__], location + location_error,
                            lds_orientation, **kwargs)
    if (np.nanmax(ranges) < pixel_size) or ranges[~np.isnan(ranges)].size == 0:
        return None, None, None, None

    # generate input map
    # (aligned to map pixels and zero rotation)
    location_fpx = location / pixel_size  # sub-pixel resolution ("float pixels")
    location_px = np.round(location_fpx).astype(int)
    location_alignment_offset_fpx = location_fpx - location_px  # true centre relative to window centre
    if map_known:
        # crop from original floorplan
        input_map = map_from_lds_train_data.rotated_crop(
            semantic_map, location_px, 0.0, size=window_size_px, mask='none', pad_value=unknown_value)

        # combine with crop from predictions
        if predicted_sample_maps is not None and model_weight > 0.0:
            predicted_map = pre_sampled_crop(
                location, window_size_px, predicted_sample_locs, predicted_sample_maps, sampling_mode='centre-first',
                max_samples=5)
            input_map = input_map * (1.0 - model_weight) + predicted_map * model_weight
    else:
        # all unknown
        input_map = np.tile(unknown_value, (window_size_px[0], window_size_px[1], 1))

    # generate output map
    if not gen_output_map:
        # don't care: blank map
        output_map = np.tile(unknown_value, (window_size_px[0], window_size_px[1], 1))
    elif map_known:
        # map location known: align to map pixels and zero rotation
        output_map = map_from_lds_train_data.rotated_crop(
            semantic_map, location_px, 0.0, size=window_size_px, mask='none', pad_value=unknown_value)
    else:
        # map location unknown: align to exact centre of window and agent's ground truth orientation
        output_map = map_from_lds_train_data.rotated_crop(
            semantic_map, location_fpx, orientation, size=window_size_px, mask='inner-circle', pad_value=unknown_value)

    # generate LDS semantic map
    # (oriented according to agent's believed orientation, which omits the orientation_error,
    #  and is 0.0 if completely unknown)
    believed_orientation = orientation if map_known and np.isfinite(orientation_error) else 0.0
    lds_map = lds.lds_to_occupancy_map(
        ranges, angle=believed_orientation, size_px=window_size_px, centre_px=location_alignment_offset_fpx,
        pixel_size=pixel_size)

    return input_map, lds_map, output_map, location_alignment_offset_fpx


def random_floor_location(semantic_map, low=None, high=None, **kwargs):
    """
    Generates a random location within the floor region of the semantic map.
    Args:
        semantic_map: (H,W,C)
        low: list/tuple/array (x,y), inclusive, default: inferred from provided map
        high: list/tuple/array (x,y), exclusive, default: inferred from provided map
    Keyword args:
        pixel_size
    Returns:
        (x,y) float in physical units
    """
    if low is None and high is None:
        low, high = get_location_range(semantic_map, **kwargs)
    if low is None:
        low, _ = get_location_range(semantic_map, **kwargs)
    if high is None:
        _, high = get_location_range(semantic_map, **kwargs)

    # keep trying until we find a position that satisfies requirements
    cnt = 0
    while True:
        location = np.random.uniform(low, high)

        # np.random.uniform() should produce in range low <= x < high, but very occasionally
        # generates values == high, so eliminate those too
        if np.all(location >= low) and np.all(location < high) and \
                class_at_location(semantic_map, location) == __FLOOR_IDX__:
            return location

        cnt += 1
        if cnt > 1000:
            raise ValueError(f"Failed to generate random location after {cnt} attempts")


# TODO update so it can work against arrays/lists of locations
def class_at_location(semantic_map, location, **kwargs):
    """
    Identifies the class of the pixel at a given location in physical units.
    Args:
        semantic_map: (H,W,C)
        location: (x,y), in physical units
    Keyword args:
        pixel_size
    Returns:
        class, integer
    """
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    loc_px = np.round(location / pixel_size).astype(int)
    return np.argmax(semantic_map[loc_px[1], loc_px[0]])


# conversion from pixel to physical coordinates:
# - a pixel's physical coordinate is at its centre, so the physical coords have range +/- 0.5 of that location
# - a pixel-shape of (2,2) has px coords in range 0..1, if we extend to include the border then we have -0.5..+1.5
def get_location_range(semantic_map, exclude_border=False, **kwargs):
    """
    Centralises the maths to accurately define the range of physical locations across a semantic map.
    Optionally skips the half-pixel width around the edges.
    Both returned low-high have a small margin removed so that it doesn't matter whether subsequent processing treats
    them as inclusive or exclusive.
    Args:
      semantic_map: entire floorplan or selection
      exclude_border: whether to omit the half-pixel width on the edges.
        Can be used as a very simplistic heuristic to skip the outermost edge that probably has no useful points anyway.
    Returns:
      (low, high) - where low/high are both (x,y) coordinates
    """
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    h, w = np.array(semantic_map.shape[0:2])
    if exclude_border:
        low_px = np.array((0, 0))
        high_px = np.array((w - 1, h - 1))
    else:
        low_px = np.array((-0.4999, -0.4999))
        high_px = np.array((w - 0.5001, h - 0.5001))

    return low_px * pixel_size, high_px * pixel_size


class DatasetRevisor:
    """
    Supports curriculum learning where the model itself is used to regenerate new
    input maps within the training data. Intended to be used within a custom training
    loop that revises the dataset every few epochs.
    """

    # TODO update to a list of labelled floorplans
    def __init__(self, floorplan, model, **kwargs):
        """
        Args:
            floorplan: ground-truth semantic map of entire floor.
            model: model to use for prediction
        Keyword args:
          pixel_size: the usual
          max_distance: the usual
        """
        self.floorplan = floorplan
        self.model = model
        self.pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
        self.max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)

        self._sample_locations = None
        self._sample_maps = None

    def apply(self, dataset):
        """
        Applies the revision process against the supplied dataset, returning a new
        dataset with revised contents.
        Args:
            dataset: input dataset with entries of form:
                ((input_map, lds_map), (map_output, adlo_output), metadata).
        Returns:
            revised dataset with same structure as input dataset.
        """
        self.prepare()

        return dataset.map(map)

    def prepare(self):
        """
        Performs pre-computation steps.
        Takes a number of LDS samples around the entire floorplan and uses
        the model to convert to semantic map predictions. These are cached
        and used while revising the dataset.
        """
        sample_locations, _, _, sample_maps = take_samples_covering_map(
            self.floorplan, self.model,
            sampling_mode='random',
            pixel_size=self.pixel_size,
            max_distance=self.max_distance)
        self._sample_locations = sample_locations
        self._sample_maps = sample_maps

    def map(self, inputs, outputs, metadata):
        """
        Passed as a function to TF.Dataset.map()
        """
        (input_map, lds_map) = inputs
        (output_map, output_adlo) = outputs

        # skip if input_map is blanked out
        unknown_min = tf.reduce_min(input_map[..., __UNKNOWN_IDX__])
        if unknown_min == 1.0:
            return (input_map, lds_map), (output_map, output_adlo), metadata

        map_centre = metadata[2:4]  # x,y, units: physical
        map_orientation = metadata[4]
        size_px = input_map.shape[1:3]
        if map_orientation != 0:
            raise ValueError(f"Only supports zero-degrees oriented maps, found: {map_orientation}")

        new_input_map = pre_sampled_crop(
            map_centre, size_px, self._sample_locations, self._sample_maps,
            sampling_mode='centre-first', max_samples=5,
            pixel_size=self.pixel_size, max_distance=self.max_distance)

        return (new_input_map, lds_map), (output_map, output_adlo), metadata


def take_samples_covering_map(semantic_map, model=None, **kwargs):
    """
    Generates a bunch of sample points and LDS maps to cover the area of the given map.
    Then uses the model to predict the semantic maps, if a model is given.

    Args:
      semantic_map: entire floorplan or section
      model: model for semantic map prediction.
        Or omit to exclude semantic_maps from the output.

    Keyword args:
      pixel_size: the usual
      max_distance: the usual
      resolution: float, default: 0.25.
        Target resolution of sample points, as a fraction of the LDS max distance.
        For example, 0.25 aims to achieve sampling points on a grid with
        0.25*max_distance distance between them.
        In 'random' sampling mode, this is only approximate.
      sampling mode: 'random' or 'grid', default: 'random'.

    Return:
      (locations, orientations, lds_maps, semantic_maps), where:
        locations/orientation - physical locations and orientations of agent
        lds_maps - LDS maps taken by agent at those locations
        semantic_maps - predicted semantic maps by model from those LDS maps, omitted
          if no model given.
    """
    # Config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)
    resolution = kwargs.get('resolution', 0.25)
    sampling_mode = kwargs.get('sampling_mode', 'random')

    # Calculate number of samples to take
    # - if doing a uniform grid with distance d, and ignoring floor vs unknown,
    #   total sample points would be (w/d) * (h/d) = (w*h)/d**2
    # - so use that as the number of points to generate under random sampling mode
    w = semantic_map.shape[1] * pixel_size
    h = semantic_map.shape[0] * pixel_size
    dist = resolution * max_distance
    target_count = np.ceil(w * h / dist ** 2).astype(int)
    map_range_low, map_range_high = get_location_range(semantic_map, **kwargs)

    # Generate sample points
    if sampling_mode == 'random':
        locs = np.random.uniform(map_range_low, map_range_high, size=(target_count, 2)).astype(np.float32)
        orientations = np.random.uniform(-np.pi, np.pi, size=(target_count,)).astype(np.float32)
    elif sampling_mode == 'grid':
        x, y = np.meshgrid(np.arange(start=map_range_low[0], stop=map_range_high[0], step=dist),
                           np.arange(start=map_range_low[1], stop=map_range_high[1], step=dist))
        locs = np.stack((x.flatten(), y.flatten()), axis=1).astype(np.float32)
        orientations = np.zeros(shape=(locs.shape[0],), dtype=np.float32)
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    # Filter: only include points on the floor
    # - will produce slightly less sample points than target, but it's fine
    mask = np.full(locs.shape[0], True)
    for i in range(locs.shape[0]):
        loc = locs[i]
        mask[i] = class_at_location(semantic_map, loc, **kwargs) == __FLOOR_IDX__
    locs = locs[mask]
    orientations = orientations[mask]

    # Generate LDS maps
    # - just like in generate_training_data_sample(), we align to map pixels
    #   when rendering
    window_size_px = np.ceil(max_distance / pixel_size).astype(int) * 2 + 1
    window_size_px = np.array([window_size_px, window_size_px])
    locs_fpx = locs / pixel_size  # sub-pixel resolution ("float pixels")
    locs_px = np.round(locs_fpx).astype(int)
    locs_alignment_offset_fpx = locs_fpx - locs_px  # true centre relative to window centre

    # FIXME WORKAROUND
    # Somehow I'm getting 159x159 windows instead of 149x149 that I've trained the model on.
    # Need to figure out what's going on here.
    def clip_to_size(map):
        target_shape = np.array([149, 149])
        clip_start = (map.shape[0:2] - target_shape) // 2
        clip_end = clip_start + target_shape
        return map[clip_start[0]:clip_end[0], clip_start[1]:clip_end[1], ...]

    # Do LDS map generation
    # - generating the first one first so we can pre-allocate the result array
    print(f"Generating {locs.shape[0]} LDS maps...")
    occupancy_map = semantic_map[..., __OBSTRUCTION_IDX__]
    ranges = lds.lds_sample(occupancy_map, locs[0], orientations[0], **kwargs)
    first_map = lds.lds_to_occupancy_map(
        ranges, angle=orientations[0], size_px=window_size_px,
        centre_px=locs_alignment_offset_fpx[0], pixel_size=pixel_size)
    first_map = clip_to_size(first_map)
    lds_maps = np.zeros((locs.shape[0],) + first_map.shape)
    lds_maps[0] = first_map
    for i in range(1, locs.shape[0]):
        ranges = lds.lds_sample(occupancy_map, locs[i], orientations[i], **kwargs)
        lds_map = lds.lds_to_occupancy_map(
            ranges, angle=orientations[i], size_px=window_size_px,
            centre_px=locs_alignment_offset_fpx[i], pixel_size=pixel_size)
        lds_map = clip_to_size(lds_map)
        lds_maps[i] = lds_map

    # Do semantic map generation
    semantic_maps = None
    if model is not None:
        print("Generating semantic maps...")
        semantic_maps = _predict_maps(model, lds_maps)

    # return
    if semantic_maps is not None:
        return locs, orientations, lds_maps, semantic_maps
    else:
        return locs, orientations, lds_maps


def _predict_maps(model, lds_maps):
    """
    Uses the model to predict the semantic maps.
    Return:
      nd.array (N,W,H,C), predicted semantic maps
    """
    unknown_value = np.zeros(__CLASSES__, dtype=np.float32)
    unknown_value[__UNKNOWN_IDX__] = 1
    unknown_maps = np.tile(unknown_value, tuple(lds_maps.shape) + (1,))
    (semantic_maps, adlos) = model.predict((unknown_maps, lds_maps))
    semantic_maps = tf.math.softmax(semantic_maps, axis=-1)
    return tf.cast(semantic_maps, tf.float32)


def pre_sampled_crop(centre, size_px, sample_locations, sample_maps, **kwargs):
    """
    Combines a bunch of semantic maps to generate a single cropped input map,
    usually as part of training data generation.
    Args:
      centre: (x,y), unit: physical
      size_px: (w,h), unit: pixels.
      sample_locations: (N,2), unit: physical.
      sample_maps: (N,H,W,C)
        Semantic maps, usually from prediction.
    Keyword args:
      sampling_mode: default 'all'.
        One of: 'all', 'centre-first', 'uniform'.
        'all' - uses all available samples within the region of the crop.
        'centre-first' - picks the sample nearest to the centre plus
          a random number of additional samples, up to a total of
          max_samples. All samples are picked such that their
          centres are within the region of the crop.
        'uniform' - picks a random number of samples, up to a total of
          max_samples, choosing any sample that has any overlap with
          the region of the crop.
      max_samples: maximum number of samples to take, default: 5
        Ignored if sampling_mode is 'all'.
      pixel_size: default __PIXEL_SIZE__
      unknown_value: default __UNKNOWN_VALUE__
    Returns:
      semantic_map: (H,W,C)
    """
    # config
    centre = np.array(centre)
    size_px = np.array(size_px)
    max_distance = kwargs.get('max_distance', lds.__MAX_DISTANCE__)
    sampling_mode = kwargs.get('sampling_mode', 'all')
    max_samples = kwargs.get('max_samples', 5)

    # identify samples to use
    distances = np.linalg.norm(centre - sample_locations, axis=1)
    target_sample_count = np.random.randint(1, max_samples + 1)
    if sampling_mode == 'all':
        # pick all with any overlap at all
        sample_indices = np.arange(len(sample_locations))[distances <= max_distance * 2]
    elif sampling_mode == 'centre-first':
        centre_idx = np.argmin(distances)

        # choose from the set with significant overlap
        available_indices = np.arange(len(sample_locations))[distances <= max_distance]
        available_indices = available_indices[available_indices != centre_idx]
        sample_indices = np.random.choice(available_indices, size=min(target_sample_count - 1, len(available_indices)),
                                          replace=False)

        # final result is centre plus randomly chosen ones
        sample_indices = np.append(sample_indices, centre_idx)
    elif sampling_mode == 'uniform':
        # choose from the set with sufficient overlap
        available_indices = np.arange(len(sample_locations))[distances <= max_distance * 1.5]
        sample_indices = np.random.choice(available_indices, size=min(target_sample_count, len(available_indices)),
                                          replace=False)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")
    sample_indices = sample_indices.astype(int)

    # combine chosen samples
    centre_px = np.round(centre / lds.__PIXEL_SIZE__).astype(int)
    window_radius_px = (size_px - 1) // 2
    output_range_px = (centre_px[0] - window_radius_px[0], centre_px[1] - window_radius_px[1], size_px[0], size_px[1])
    combined_map, _ = combine_semantic_maps(
        sample_locations[sample_indices], tf.gather(sample_maps, sample_indices),
        output_range_px=output_range_px, **kwargs)
    return combined_map


# Combination is computed based on some probability logic.
# Let p(observed|Si)
#    = probability that particular position has been observed
#    = p(floor|Si) + p(wall|Si) = softmax estimate of floor + wall for a given sample
#    = semantic_maps[i][...,0] + semantic_maps[i][...,1]
# We first wish to compute p(observable)
#    = probability that a particular position is ever observable
# We assume that all samples provided represent the entire population of possible
# observations, and thus:
#    p(observable) = max{i} p(observed|Si)
# Next we wish to compute the probabilities for whether the position is a
# floor or wall, respectively, given that the position is observable.
# This requires rescaling from 3 possibilies to 2. Furthermore, to compute
# across the entire population of samples, it requires expanding out
# the bayes formula and summing over the samples - which results in a weighted
# average of the samples:
# So p(wall|observable)
#    = sum{i} p(wall|Si,observable) * p(Si,observable)
#    = sum{i} p(wall|Si,observable) * p(Si|observable) * p(observable)
# Now p(wall|Si,observable)
#    = probability that it's a wall for sample Si, and given that it's observable
#    = rescale from 3 to 2 possibilities
#          p(wall|Si)          p(wall|Si)
#    =  -------------- = ------------------------
#       p(observed|Si)   p(wall|Si) + p(floor|Si)
# And p(Si|observable)
#    = extent to which Si agrees with the p(observable) outcome,
#      divided by total number of agreeing Si's
#          p(observed|Si)
#    = ---------------------
#      sum{j} p(observed|Sj)
# So finally, p(wall|observable)
#    = sum{i} p(wall|Si,observable) * p(Si|observable) * p(observable)
#
#               p(wall|Si)       p(observed|Si)
#    = sum{i} -------------- * ------------------ * p(observable)
#             p(observed|Si)   sum p(observed|Sj)
#
#                   p(wall|Si)
#    = sum{i} --------------------- * max{k} p(observed|Sk)
#             sum{j} p(observed|Sj)
# Likewise for p(floor|observable).
# And lastly, we need to compute a new value for p(unobservable), which
# we can do without any of the other rigmarole because it's just the inverse
# of the probability that it's observable.
# Thus p(unobservable)
#    = 1 - p(observable)
#    = 1 - max{i} p(observed|Si)
@tf.function
def combine_semantic_maps(locs, semantic_maps, **kwargs):
    """
    Overlays and combines semantic maps spanning different positions.

    The combination is more than a naive average. It acknowledges that
    an "unknown" state from one semantic map simply means that it hasn't "observed"
    the state at a given position, while an "unknown" state for the same position
    across all semantic maps suggests that the position may indeed be "unobservable".
    Then applies a weighted average to resolve discrepancies between floor vs wall,
    with samples given weights according to how much they agree with the
    observable/unobservable outcome.

    Args:
        locs - float (N,2), units: physical
            Locations of agent when each semantic map was taken.
        semantic_maps - (N,H,W,C)
            Semantic maps, usually generated by model

    Keyword args:
        pixel_size: usual meaning
        output_range: float tensor, (x1, y1, w, h), unit: physical.
          Location span of output map, from start of centre of its top-left pixel.
          If not provided, output map is computed to span the union of the
          maps provided.
        output_range_px: float tensor, (x1, y1, w, h), unit: pixels.
          Same as output_range, but in pixel units relative to the original
          floorplan pixel coordinates. At most one should be provided.

    Returns:
        (combined_map, location_start) with semantic map containing the combined results,
        and the physical location of centre of top-left pixel
    """
    # note: internally this functions uses pixel coordinates throughout,
    # unless otherwise stated

    # config
    pixel_size = kwargs.get('pixel_size', lds.__PIXEL_SIZE__)
    output_range_apu = kwargs.get('output_range')  # absolute physical units
    output_range_apx = kwargs.get('output_range_px')  # absolute pixel units
    window_size_px = tf.gather(tf.shape(semantic_maps), (2,1))  # (N,H,W,3) -> [w,h]

    # compute coordinates of output map
    # - offset_px   - px coord of centre of top-left most window
    # - out_shape   - px height,width
    if output_range_apx is not None:
        output_range_apx = tf.cast(tf.round(output_range_apx), tf.int32)
    elif output_range_apu is not None:
        output_range_apx = tf.cast(tf.round(output_range_apu / pixel_size), tf.int32)

    half_window_px = (window_size_px - 1) // 2  # (2,) x int32
    if output_range_apx is not None:
        offset_px = tf.gather(output_range_apx, (0,1)) + half_window_px
        out_shape = tf.gather(output_range_apx, (2,3))  # (h,w)
    else:
        min_locs_px = tf.cast(tf.round(tf.reduce_min(locs, axis=0) / pixel_size), tf.int32) - half_window_px
        max_locs_px = tf.cast(tf.round(tf.reduce_max(locs, axis=0) / pixel_size), tf.int32) - half_window_px + window_size_px
        offset_px = min_locs_px + half_window_px  # (2,) x int32
        out_shape = tf.gather(max_locs_px - min_locs_px, (1,0))  # (h,w)
    location_start = tf.cast(offset_px - half_window_px, tf.float32) * pixel_size  # (2,) x float32

    # Initialize tensors for accumulation
    max_observed = tf.zeros(out_shape, dtype=tf.float32)
    sum_observed = tf.zeros(out_shape, dtype=tf.float32)
    out_sum = tf.zeros(tf.concat([out_shape, tf.constant([3], tf.int32)], axis=0), dtype=tf.float32)

    # compute max and sums across entire map
    # also compute sum over samples
    def process_map(i, out_sum, sum_observed, max_observed):
        this_map = tf.gather(semantic_maps, i, axis=0)  # (H,W,C)
        start_px = tf.cast(tf.round(locs[i] / pixel_size), tf.int32) - offset_px  # window-centre apu -> top-left px
        observed = tf.reduce_sum(tf.gather(this_map, indices=(0, 1), axis=-1), axis=-1)  # (H,W) x 0..1 prob
        out_indices, map_indices = get_intersect_ranges_tf(out_shape, tf.shape(this_map), start_px)

        # Accumulate values
        # sum{i} p(floor|Si)
        # sum{i} p(wall|Si)
        # out_sum[out_slices] = np.add(out_sum[out_slices], this_map[map_slices])
        out_sum = tf.tensor_scatter_nd_add(out_sum, out_indices, tf.gather_nd(this_map, map_indices))

        # sum p(observed|Sj) = normalizer
        # sum_observed[out_slices] = np.add(sum_observed[out_slices], observed[map_slices])
        sum_observed = tf.tensor_scatter_nd_add(sum_observed, out_indices, tf.gather_nd(observed, map_indices))

        # max p(observed|Sk) = p(observable)
        # max_observed[out_slices] = np.maximum(max_observed[out_slices], observed[map_slices])
        max_observed = tf.tensor_scatter_nd_max(max_observed, out_indices, tf.gather_nd(observed, map_indices))

        return i + 1, out_sum, sum_observed, max_observed

    # Loop over all semantic maps and combine
    i = tf.constant(0)
    cond = lambda i, out_sum, sum_observed, max_observed: tf.less(i, tf.shape(locs)[0])
    _, out_sum, sum_observed, max_observed = tf.while_loop(cond, process_map, [i, out_sum, sum_observed, max_observed])

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


def get_intersect_ranges(map1, map2, offset_px):
    """
    Makes life easier when doing computations with partially overlaying maps.
    Returns the slices in each where they overlap.
    Args:
        map1: tuple or map
            Occupancy map, semantic map, or shape thereof. (H,W) or (H,W,C)
        map2: tuple or map
            Occupancy map, semantic map, or shape thereof. (H,W) or (H,W,C)
        offset_px: int, (x,y), unit: pixels.
            Offset of map2 top-left relative to map1 top-left
    Returns:
        ((map1_row_slice, map1_col_slice), (map2_row_slice, map2_col_slice))
        containing python `slice` objects for indexing into
        the map arrays, or ((None, None), (None, None)) if there's no intersect
    """
    h1, w1 = _map_shape(map1)[0:2]
    h2, w2 = _map_shape(map2)[0:2]
    size1 = np.array([w1, h1])
    size2 = np.array([w2, h2])
    offset_px = np.array(offset_px)

    start1 = np.maximum(np.array([0, 0]), offset_px)
    end1 = np.minimum(size1, size2 + offset_px)
    start2 = np.maximum(np.array([0, 0]), -offset_px)
    end2 = np.minimum(size2, size1 - offset_px)

    if np.any(end1 - start1 <= 0):
        return (None, None), (None, None)
    else:
        # convert to row,col
        return (slice(start1[1], end1[1]), slice(start1[0], end1[0])), (slice(start2[1], end2[1]), slice(start2[0], end2[0]))


@tf.function
def get_intersect_ranges_tf(map_shape1, map_shape2, offset_px):
    """
    TF AutoGraph compatible equivalent of get_intersect_ranges().
    Makes life easier when doing computations with partially overlaying maps.
    Returns indices in each where they overlap.
    Args:
        map_shape1: int tensor
            Shape of first occupancy map or semantic map (H,W) or (H,W,C)
        map_shape2: int tensor
            Shape of second occupancy map or semantic map (H,W) or (H,W,C)
        offset_px: int tensor, (x,y), unit: pixels.
            Offset of map2 top-left relative to map1 top-left
    Returns:
        map1_indices, map2_indices.
        Both of shape (H',W',2), containing indices into the map arrays,
        suitable for passing into tf.gather_nd() and similar.
        The tensors of indices are both empty if there is no intersect.
    """
    size1 = tf.gather(map_shape1, (1, 0))
    size2 = tf.gather(map_shape2, (1, 0))

    # Compute the intersection ranges for map1 and map2
    start1 = tf.maximum([0, 0], offset_px)
    end1 = tf.minimum(size1, size2 + offset_px)
    start2 = tf.maximum([0, 0], -offset_px)
    end2 = tf.minimum(size2, size1 - offset_px)

    # Enforce limits
    start1 = tf.minimum(tf.maximum(start1, [0, 0]), size1)
    end1 = tf.minimum(tf.maximum(end1, [0, 0]), size1)
    start2 = tf.minimum(tf.maximum(start2, [0, 0]), size2)
    end2 = tf.minimum(tf.maximum(end2, [0, 0]), size2)

    # Generate per-axis ranges for each map
    # result #1: (H',) x int and (W',) x int
    # (empty if no intersection)
    map1_row_indices = tf.range(start1[1], end1[1])
    map1_col_indices = tf.range(start1[0], end1[0])
    map2_row_indices = tf.range(start2[1], end2[1])
    map2_col_indices = tf.range(start2[0], end2[0])

    # Compute (row,col) meshgrid
    # result #1: 2 x (H',W') x int
    map1_row_indices, map1_col_indices = tf.meshgrid(map1_row_indices, map1_col_indices, indexing='ij')
    map2_row_indices, map2_col_indices = tf.meshgrid(map2_row_indices, map2_col_indices, indexing='ij')

    # Stack
    # result #1: (H',W',2) x int
    map1_indices = tf.stack([map1_row_indices, map1_col_indices], axis=-1)
    map2_indices = tf.stack([map2_row_indices, map2_col_indices], axis=-1)

    return map1_indices, map2_indices


def _map_shape(map_or_shape: Any):
    """
    Args:
        map_or_shape: one of:
            - tuple, list, 1D array, TensorShape containing the shape
            - 2D+ numpy array or Tensor
    Returns:
        numpy array containing shape
    """
    if isinstance(map_or_shape, (tuple, list, tf.TensorShape)):
        return np.array(tuple(map_or_shape))
    elif isinstance(map_or_shape, np.ndarray) and map_or_shape.ndim == 1:
        return np.array(tuple(map_or_shape))
    else:
        return np.array(map_or_shape.shape)


def save_dataset(dataset, file):
    """
    Saves the dataset to the designated file, overwriting any existing file.

    Saves via Numpy format because I had problems with the TF Dataset's own saving format.
    1. It saves to a folder structure, which makes downloading it cumbersome.
    2. I had problems with it looking like it'd saved, but it actually produced an empty
       collection.

    :param dataset: A dataset returned by generate_training_data()
    :param file: file to save to
    """

    # Iterate through the dataset and collect the data
    input_maps = []
    lds_maps = []
    output_maps = []
    adlos = []
    metadatas = []
    for (input_map, lds_map), (output_map, adlo), metadata in dataset:
        input_maps.append(input_map.numpy())
        lds_maps.append(lds_map.numpy())
        output_maps.append(output_map.numpy())
        adlos.append(adlo.numpy())
        metadatas.append(metadata.numpy())

    print(f"Saving:")
    print(f"  input_maps:  {np.shape(input_maps)}")
    print(f"  lds_maps:    {np.shape(lds_maps)}")
    print(f"  output_maps: {np.shape(output_maps)}")
    print(f"  adlos:       {np.shape(adlos)}")
    print(f"  metadatas:   {np.shape(metadatas)}")

    np.savez_compressed(file,
                        input_maps=np.array(input_maps),
                        ld_maps=np.array(lds_maps),
                        output_maps=np.array(output_maps),
                        adlos=np.array(adlos),
                        metadatas=np.array(metadatas))
    print(f"Dataset saved to {file}")


def load_dataset(file):
    """
    Loads a dataset saved by save_dataset().

    Handles that some older datasets used 'ground_truth_maps' instead of 'output_maps'
    and omitted 'metadatas'
    :param file: The file to load from
    :return: tf.data.Dataset
    """
    with np.load(file, allow_pickle=False) as data:
        input_maps = data['input_maps']
        lds_maps = data['ld_maps']
        adlos = data['adlos']

        if 'output_maps' in data.keys():
            output_maps = data['output_maps']
        else:
            # older label
            output_maps = data['ground_truth_maps']

        if 'metadatas' in data.keys():
            metadatas = data['metadatas']
        else:
            # older format had no metadata
            metadatas = np.full((input_maps.shape[0], 8), np.nan, dtype=np.float32)
            print("Warning: no metadata available in this dataset. Using dummy values.")

    # older format: adlos were saved in float64
    # (convert, for consistency and so that validate_dataset() passes)
    if adlos.dtype != np.float32:
        print(f"Note: upgraded adlos datatype from {adlos.dtype}.")
        adlos = adlos.astype(np.float32)

    print(f"Loaded:")
    print(f"  input_maps:  {np.shape(input_maps)} x {input_maps.dtype}")
    print(f"  lds_maps:    {np.shape(lds_maps)} x {lds_maps.dtype}")
    print(f"  output_maps: {np.shape(output_maps)} x {output_maps.dtype}")
    print(f"  adlos:       {np.shape(adlos)} x {adlos.dtype}")
    print(f"  metadatas:   {np.shape(metadatas)} x {metadatas.dtype}")

    dataset = tf.data.Dataset.from_tensor_slices((
        (input_maps, lds_maps),
        (output_maps, adlos),
        metadatas
    ))
    print(f"Dataset loaded from {file}")
    return dataset


def validate_dataset(dataset):
    """
    Sanity checks generated data against some basic rules.
    :param dataset:
    """
    def assert_in_range(name, tensor, allowed_min, allowed_max):
        if np.min(tensor) < allowed_min or np.max(tensor) > allowed_max:
            raise ValueError(f"{name} has values outside of desired range, found: {np.min(tensor)}-{np.max(tensor)}")

    def assert_dtype(name, tensor, allowed_dtype):
        if tensor.dtype != allowed_dtype:
            raise ValueError(f"{name} has wrong datatype, found: {tensor.dtype}")

    count = 0
    for (input_map, lds_map), (output_map, adlo), metadata in dataset:
        assert_in_range("input_map", input_map, 0.0, 1.0)
        assert_in_range("lds_map", lds_map, 0.0, 1.0)
        assert_in_range("output_map", output_map, 0.0, 1.0)
        assert_in_range("accept", adlo[0], 0.0, 1.0)
        assert_in_range("loc-x", adlo[1], -0.5, 0.5)
        assert_in_range("loc-y", adlo[2], -0.5, 0.5)
        assert_in_range("orientation", adlo[3], -1.0, 1.0)

        assert_dtype("input_map", input_map, np.float32)
        assert_dtype("lds_map", lds_map, np.float32)
        assert_dtype("output_map", output_map, np.float32)
        assert_dtype("adlo", adlo, np.float32)
        assert_dtype("metadata", metadata, np.float32)

        count += 1
    print(f"Dataset tests passed ({count} entries verified)")


def show_dataset(dataset, num=5):
    for (input_map, lds_map), (output_map, adlo), metadata in dataset.take(num):
        show_data_sample(input_map, lds_map, output_map, adlo, metadata)


def show_data_sample(input_map, lds_map, output_map, adlo, metadata=None):
    print(f"input_map:  {input_map.shape}")
    print(f"lds_map:    {lds_map.shape}")
    print(f"output_map: {output_map.shape}")
    print(f"adlo:       {adlo}")
    print(f"metadata:   {metadata}")

    map_size = np.array([input_map.shape[1], input_map.shape[0]])

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 3, 1)
    plt.title('Input Map')
    plt.imshow(input_map)
    plt.axis('off')
    if not adlo[0]:
        # to be rejected so just add cross through map
        plt.plot([0, map_size[0] - 1], [0, map_size[1] - 1], c='y')
        plt.plot([0, map_size[0] - 1], [map_size[1] - 1, 0], c='y')
    else:
        # show adlo relative to centre dot
        centre = map_size / 2
        error_loc = centre + adlo[1:3] * map_size
        angle_loc = error_loc + np.array([np.cos(adlo[3] * np.pi), np.sin(adlo[3] * np.pi)]) * 50
        plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
        plt.scatter(centre[0], centre[1], c='k', s=50)
        plt.scatter(error_loc[0], error_loc[1], c='m', s=50)

    plt.subplot(1, 3, 2)
    plt.title('LDS')
    plt.imshow(lds_map, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Output Map')
    plt.imshow(output_map, cmap='gray')
    plt.axis('off')
    plt.show()


def show_predictions(model, dataset, num=5, **kwargs):
    """
    :param model: slam model
    :param dataset: slam dataset
    :param num: number of predictions to show
    Keyword args:
        from_logits: (bool) whether model outputs logits, or scaled values otherwise
        show_classes: (bool) whether to add extra columns for each individual class
        flexi: (bool) whether to enable support for more flexible data representations that might
               omit some of the inputs or outputs
    """
    flexi = kwargs.get('flexi', False)

    if flexi:
        flexi_show_predictions(model, dataset, num, **kwargs)
    else:
        inputs, outputs, metadatas = next(iter(dataset.batch(num)))
        preds = model.predict(inputs)
        for map_window, lds_map, ground_truth_map, adlo, metadata, map_pred, adlo_pred in zip(
                inputs[0], inputs[1], outputs[0], outputs[1], metadatas, preds[0], preds[1]):
            show_prediction(map_window, lds_map, ground_truth_map, adlo, metadata, map_pred, adlo_pred, **kwargs)


def flexi_show_predictions(model, dataset, num=1, **kwargs):
    """
    Extremely flexible form of 'show_predictions' that copes with different dataset representations
    Batch: tuple: (inputs, outputs)
    inputs: either a single input or a tuple of inputs
    outputs: either a single output or a tuple of outputs
    """
    batch = next(iter(dataset.batch(num)))
    preds = model.predict(batch[0])
    print(f"batch:    {type(batch)} x {len(batch)} x ({type(batch[0])},...)")
    for i in range(len(batch)):
        if isinstance(batch[i], tuple):
            print(f"batch[{i}]: {type(batch[i])} x {len(batch[i])}")
            print(f"batch[{i}][0]: {type(batch[i][0])}, shape {np.shape(batch[i][0])}")
            print(f"batch[{i}][1]: {type(batch[i][1])}, shape {np.shape(batch[i][1])}")
        else:
            print(f"batch[{i}]: {type(batch[i])} x {len(batch[i])}, shape {np.shape(batch[i])}")
    print(f"preds:    {type(preds)} x {len(preds)}, {np.shape(preds)}")

    if isinstance(batch[0], tuple):
        input_maps = batch[0][0]
        lds_maps = batch[0][1] if len(batch[0]) >= 2 else None
    else:
        input_maps = batch[0]
        lds_maps = [None] * len(input_maps)

    if isinstance(batch[1], tuple):
        output_map_trues = batch[1][0]
        adlo_trues = batch[1][1] if len(batch[1]) >= 2 else None
    else:
        output_map_trues = batch[1]
        adlo_trues = [None] * len(output_map_trues)

    if len(batch) >= 3:
        metadatas = batch[2]
    else:
        metadatas = [None] * len(output_map_trues)

    if isinstance(preds, tuple):
        output_map_preds = preds[0]
        adlo_preds = preds[1] if len(preds) >= 2 else None
    else:
        output_map_preds = preds
        adlo_preds = [None] * len(output_map_preds)

    for input_map, lds_map, output_map_true, adlo, metadata, map_pred, adlo_pred in zip(
            input_maps, lds_maps, output_map_trues, adlo_trues, metadatas, output_map_preds, adlo_preds):
        print(f"map_input: {np.shape(input_map)}")
        print(f"lds_input: {np.shape(lds_map)}")
        print(f"output_map_true: {np.shape(output_map_true)}")
        print(f"adlo_true: {np.shape(adlo)}")
        print(f"output_map_pred: {np.shape(map_pred)}")
        print(f"adlo_pred: {np.shape(adlo_pred)}")
        show_prediction(input_map, lds_map, output_map_true, adlo, metadata, map_pred, adlo_pred, **kwargs)


def show_prediction(input_map, lds_map, map_true, adlo, metadata, map_pred, adlo_pred, **kwargs):
    """
    :param input_map:
    :param lds_map:
    :param map_true:
    :param adlo:
    :param metadata:
    :param map_pred:
    :param adlo_pred:

    Keyword args:
      show_classes: one of 'none' (or False), 'all' (or True), 'pred'
    """
    from_logits = kwargs.get('from_logits', True)
    show_classes = kwargs.get('show_classes', 'none')
    map_size = np.array([input_map.shape[1], input_map.shape[0]])
    n_classes = input_map.shape[-1]

    if show_classes == True:
        show_classes = 'all'
    elif show_classes == False:
        show_classes = 'none'

    # apply scaling
    map_pred_scaled = tf.nn.softmax(map_pred, axis=-1) if from_logits else map_pred
    map_pred_categorical = tf.argmax(map_pred_scaled, axis=-1)
    if adlo_pred is not None:
        accept = tf.nn.sigmoid(adlo_pred[0]) if from_logits else adlo_pred[0]
        adlo_pred_scaled = tf.stack([accept, adlo_pred[1], adlo_pred[2], adlo_pred[3]], axis=0)
    else:
        adlo_pred_scaled = None

    # Log details that are not so great in visual form
    print(f"adlo:             {adlo}")
    print(f"adlo-pred raw:    {adlo_pred}")
    print(f"adlo-pred scaled: {adlo_pred_scaled}")
    print(f"metadata:         {metadata}")

    # Calculate total number of plots to display
    cols = 0
    cols = cols + (1 if input_map is not None else 0)
    cols = cols + (1 if lds_map is not None else 0)
    cols = cols + (1 if input_map is not None else 0)
    cols = cols + (1 if map_true is not None else 0)
    cols = cols + (n_classes if map_true is not None and show_classes else 0)
    cols = cols + (1 if map_pred_categorical is not None else 0)
    cols = cols + (n_classes if map_pred_scaled is not None and show_classes else 0)

    # Show plots
    plt.figure(figsize=(20, 2))  # limits by row height
    i = iter(range(1, cols+1))

    if input_map is not None:
        plt.subplot(1, cols, next(i))
        plt.title('Map')
        plt.imshow(input_map)
        plt.axis('off')
        if adlo is not None and not adlo[0]:
            # to be rejected so just add cross through map
            plt.plot([0, map_size[0] - 1], [0, map_size[1] - 1], c='y')
            plt.plot([0, map_size[0] - 1], [map_size[1] - 1, 0], c='y')
        elif adlo is not None:
            # show adlo relative to centre dot
            centre = map_size / 2
            error_loc = centre + adlo[1:3] * map_size
            angle_loc = error_loc + np.array([np.cos(adlo[3] * np.pi), np.sin(adlo[3] * np.pi)]) * 50
            plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
            plt.scatter(centre[0], centre[1], c='k', s=50)
            plt.scatter(error_loc[0], error_loc[1], c='m', s=50)

    if lds_map is not None:
        plt.subplot(1, cols, next(i))
        plt.title('LDS')
        plt.imshow(lds_map, cmap='gray')
        plt.axis('off')

    if map_true is not None:
        plt.subplot(1, cols, next(i))
        plt.title('Ground Truth')
        plt.imshow(map_true)
        plt.axis('off')

    if map_true is not None and show_classes == 'all':
        for channel in range(n_classes):
            plt.subplot(1, cols, next(i))
            plt.title(f"Truth:{channel}")
            plt.imshow(map_true[..., channel], cmap='gray')
            plt.axis('off')

    if map_pred_categorical is not None:
        plt.subplot(1, cols, next(i))
        plt.title('Predicted')
        plt.imshow(map_pred_categorical)
        plt.axis('off')
        if adlo_pred_scaled is not None and adlo_pred_scaled[0] < 0.5:
            plt.plot([0, map_size[0] - 1], [0, map_size[1] - 1], c='y')
            plt.plot([0, map_size[0] - 1], [map_size[1] - 1, 0], c='y')
        if adlo_pred_scaled is not None:
            centre = map_size / 2
            error_loc = centre + adlo_pred_scaled[1:3] * map_size
            angle = adlo_pred_scaled[3] * np.pi
            angle_loc = error_loc + np.array([np.cos(angle), np.sin(angle)]) * 50
            plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
            plt.scatter(centre[0], centre[1], c='k', s=50)
            plt.scatter(error_loc[0], error_loc[1], c='m', s=50)

    if map_pred_scaled is not None:
        plt.subplot(1, cols, next(i))
        plt.title('Pred Scaled')
        plt.imshow(map_pred_scaled)
        plt.axis('off')
        plt.plot([0, map_size[0]-1], [0, map_size[1]-1], c='y', alpha=1-adlo_pred_scaled[0].numpy())
        plt.plot([0, map_size[0]-1], [map_size[1]-1, 0], c='y', alpha=1-adlo_pred_scaled[0].numpy())

    if map_pred_scaled is not None and show_classes in ('all', 'pred'):
        for channel in range(n_classes):
            plt.subplot(1, cols, next(i))
            plt.title(f"Pred:{channel}")
            plt.imshow(map_pred_scaled[..., channel], cmap='gray')
            plt.axis('off')

    plt.show()


def show_map(semantic_map):
    """
    Displays a single semantic-map under the assumption that it's an entire floorplan.
    :param semantic_map:
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.title('Semantic Map')
    plt.imshow(semantic_map)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title('Floor')
    plt.imshow(semantic_map[:, :, 0], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title('Obstructions')
    plt.imshow(semantic_map[:, :, 1], cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title('Unknown')
    plt.imshow(semantic_map[:, :, 2], cmap='gray')
    # plt.axis('off')  # include axis because outside is white

    plt.show()
