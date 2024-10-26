# Creates training data for the SLAM model.

import lds
import map_from_lds_train_data
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

# using cm as the default unit for now
__PIXEL_SIZE__ = 4.471   # cm/px
__MAX_DISTANCE__ = 350   # 3.5m in cm


def generate_training_data(semantic_map, num_samples, **kwargs):
    tot_sample_types = 4
    pixel_size = kwargs.get('pixel_size', __PIXEL_SIZE__)
    max_distance = kwargs.get('max_distance', __MAX_DISTANCE__)
    sample_types = np.array(kwargs.get('sample_types', range(tot_sample_types)))
    print(f"Generating {num_samples} samples of training data")
    print(f"Pixel size: {pixel_size}")
    print(f"Max distance: {max_distance}")
    print(f"Sample types: {sample_types}")

    # identify ranges
    # - size of full map (W, H), physical units
    # - size of window (W, H), physical units
    full_map_loc_range = np.array([semantic_map.shape[1], semantic_map.shape[0]]) * pixel_size
    window_size = (np.ceil(max_distance / pixel_size).astype(int) * 2 + 1) * pixel_size
    window_range = np.array([window_size, window_size])

    input_maps = []
    lds_maps = []
    ground_truth_maps = []
    adlos = []
    attempts = 0
    for _ in tqdm.tqdm(range(num_samples)):
        # keep trying until we generate one sample
        while True:
            attempts += 1
            sample_type = np.random.choice(sample_types)
            location = np.random.uniform((0, 0), full_map_loc_range)
            angle = np.random.uniform(-np.pi, np.pi)
            accept = True
            loc_error = (0.0, 0.0)
            angle_error = 0.0

            if sample_type == 0:
                # New map, unknown location and orientation, loc/angle error disregarded
                map_window, lds_map, ground_truth_map, centre_offset = generate_training_data_sample(
                    semantic_map, location, angle, False, None, None, pixel_size=pixel_size, max_distance=max_distance)

            elif sample_type == 1:
                # Known map, known location/angle with some small uniform estimation error
                loc_error = np.random.normal((0, 0),
                                             window_range / 2 * 0.1)  # normal about centre, std.dev = 10% of range
                angle_error = np.random.uniform(0, np.pi * 0.1)  # normal about zero, std.dev = 10% of 180 degrees
                loc_error = np.clip(loc_error, -window_range / 2, +window_range / 2)
                angle_error = np.clip(angle_error, -np.pi, +np.pi)
                map_window, lds_map, ground_truth_map, centre_offset = generate_training_data_sample(
                    semantic_map, location, angle, True, loc_error, angle_error, pixel_size=pixel_size,
                    max_distance=max_distance)

            elif sample_type == 2:
                # Known map, location unknown and searching, with LDS data partially on this map window.
                # Location of map window independent of LDS data so simulate using uniform location and
                # orientation error within bounds of the window
                loc_error = np.random.uniform(-window_range / 2,
                                              +window_range / 2)  # uniform either side of zero, anywhere within window
                angle_error = np.random.uniform(-np.pi, np.pi)  # uniform anywhere within 360-degree range
                map_window, lds_map, ground_truth_map, centre_offset = generate_training_data_sample(
                    semantic_map, location, angle, True, loc_error, angle_error, pixel_size=pixel_size,
                    max_distance=max_distance)

            elif sample_type == 3:
                # Known map, location unknown and searching, with LDS not on this map window.
                # Generate LDS from completely independent random location/orientation within full map
                # TODO find a more efficient way to ensure that the ground-truth only includes what the LDS data
                #  can see.
                # TODO use some smarts to pick a location that is a full max_distance diameter away but still on the
                #  floorplan.
                lds_loc = np.random.uniform((0, 0), full_map_loc_range)  # uniform anywhere within full map
                lds_angle = np.random.uniform(-np.pi, np.pi)  # uniform anywhere within 360-degree range
                loc_error = lds_loc - location
                angle_error = lds_angle - angle
                map_window, _, _, centre_offset = generate_training_data_sample(
                    semantic_map, location, angle, True, loc_error, angle_error, pixel_size=pixel_size,
                    max_distance=max_distance)
                _, lds_map, ground_truth_map, _ = generate_training_data_sample(
                    semantic_map, lds_loc, lds_angle, False, None, None, pixel_size=pixel_size, max_distance=max_distance)

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

            # emit results
            if lds_map is not None:
                adlo = np.array([
                    1.0 if accept else 0.0,
                    loc_error[0] / window_range[0],  # convert to range: -0.5 .. +0.5
                    loc_error[1] / window_range[1],  # convert to range: -0.5 .. +0.5
                    angle_error / np.pi  # convert to range: -1.0 .. +1.0
                ])

                input_maps.append(map_window)
                lds_maps.append(lds_map)
                ground_truth_maps.append(ground_truth_map)
                adlos.append(adlo)
                break

    print(f"Generated {len(input_maps)} samples after {attempts} attempts")
    return tf.data.Dataset.from_tensor_slices((
        (input_maps, lds_maps),
        (ground_truth_maps, adlos)
    ))


def generate_training_data_sample(semantic_map, location, orientation, map_known, location_error, orientation_error, **kwargs):
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
    max_distance = kwargs.get('max_distance', 100)
    pixel_size = kwargs.get('pixel_size', 1.0)
    unknown_value = kwargs.get('unknown_value', np.array([0, 0, 1], dtype=semantic_map.dtype))
    location_error = np.array(location_error) if location_error is not None else np.array([0.0, 0.0])
    orientation_error = orientation_error if orientation_error is not None else 0.0
    window_size_px = np.ceil(max_distance / pixel_size).astype(int) * 2 + 1
    window_size_px = np.array([window_size_px, window_size_px])

    # take LDS sample
    lds_orientation = (orientation + orientation_error) if np.isfinite(orientation_error) else orientation
    ranges = lds.lds_sample(
        semantic_map[:, :, 1], location + location_error, lds_orientation, max_distance=max_distance,
        pixel_size=pixel_size)
    if (np.nanmax(ranges) < pixel_size) or ranges[~np.isnan(ranges)].size == 0:
        return None, None, None, None

    # generate input map
    # (aligned to map pixels, and zero rotation)
    location_fpx = location / pixel_size  # sub-pixel resolution ("float pixels")
    location_px = np.round(location_fpx).astype(int)
    location_alignment_offset_fpx = location_fpx - location_px  # true centre relative to window centre
    if map_known:
        map_window = map_from_lds_train_data.rotated_crop(
            semantic_map, location_px, 0.0, size=window_size_px, mask='none', pad_value=unknown_value)
    else:
        # all unknown
        map_window = np.tile(unknown_value, (window_size_px[0], window_size_px[1], 1))

    # generate ground-truth map
    # (map_known:  aligned to map pixels and zero rotation)
    # (!map_known: aligned to exact centre of window and agent's ground truth orientation)
    if map_known:
        ground_truth_map = map_from_lds_train_data.rotated_crop(
          semantic_map, location_px, 0.0, size=window_size_px, mask='none', pad_value=unknown_value)
    else:
        ground_truth_map = map_from_lds_train_data.rotated_crop(
          semantic_map, location_fpx, orientation, size=window_size_px, mask='inner-circle', pad_value=unknown_value)

    # generate LDS semantic map
    # (oriented according to agent's believed orientation, which omits the orientation_error,
    #  and is 0.0 if completely unknown)
    believed_orientation = orientation if map_known and np.isfinite(orientation_error) else 0.0
    lds_map = lds.lds_to_occupancy_map(
        ranges, start_angle=believed_orientation, size_px=window_size_px, centre_px=location_alignment_offset_fpx,
        pixel_size=pixel_size)

    return map_window, lds_map, ground_truth_map, location_alignment_offset_fpx

def save_dataset(dataset, file):
    # Iterate through the dataset and collect the data
    input_maps = []
    lds_maps = []
    ground_truth_maps = []
    adlos = []
    for (inputs, outputs) in dataset:
        input_maps.append(inputs[0].numpy())
        lds_maps.append(inputs[1].numpy())
        ground_truth_maps.append(outputs[0].numpy())
        adlos.append(outputs[1].numpy())

    print(f"Saving:")
    print(f"  input_maps:        {np.shape(input_maps)}")
    print(f"  lds_maps:          {np.shape(lds_maps)}")
    print(f"  ground_truth_maps: {np.shape(ground_truth_maps)}")
    print(f"  adlos:             {np.shape(adlos)}")

    np.savez_compressed(file,
                        input_maps=np.array(input_maps),
                        ld_maps=np.array(lds_maps),
                        ground_truth_maps=np.array(ground_truth_maps),
                        adlos=np.array(adlos))
    print(f"Dataset saved to {file}")


def load_dataset(file):
    data = np.load(file)
    input_maps = data['input_maps']
    lds_maps = data['ld_maps']
    ground_truth_maps = data['ground_truth_maps']
    adlos = data['adlos']

    print(f"Loaded:")
    print(f"  input_maps:        {np.shape(input_maps)}")
    print(f"  lds_maps:          {np.shape(lds_maps)}")
    print(f"  ground_truth_maps: {np.shape(ground_truth_maps)}")
    print(f"  adlos:             {np.shape(adlos)}")

    dataset = tf.data.Dataset.from_tensor_slices((
        (input_maps, lds_maps),
        (ground_truth_maps, adlos)
    ))
    print(f"Dataset loaded from {file}")
    return dataset


def validate_dataset(dataset):
    """
    Sanity checks generated data.
    :param dataset:
    """
    def assert_in_range(name, tensor, allowed_min, allowed_max):
        if np.min(tensor) < allowed_min or np.max(tensor) > allowed_max:
            raise ValueError(f"{name} has values outside of desired range, found: {np.min(tensor)}-{np.max(tensor)}")

    count = 0
    for (map_window, lds_map), (ground_truth_map, adlo) in dataset:
        assert_in_range("map_window", map_window, 0.0, 1.0)
        assert_in_range("lds_map", map_window, 0.0, 1.0)
        assert_in_range("ground_truth_map", map_window, 0.0, 1.0)
        assert_in_range("accept", adlo[0], 0.0, 1.0)
        assert_in_range("loc-x", adlo[1], -0.5, 0.5)
        assert_in_range("loc-y", adlo[2], -0.5, 0.5)
        assert_in_range("orientation", adlo[3], -1.0, 1.0)
        count += 1
    print(f"Dataset tests passed ({count} entries verified)")


def show_data_sample(map_window, lds_map, ground_truth_map, adlo):
    print(f"map_window:       {map_window.shape}")
    print(f"lds_map:          {lds_map.shape}")
    print(f"ground_truth_map: {ground_truth_map.shape}")
    print(f"adlo:             {adlo}")

    range = np.array([map_window.shape[1], map_window.shape[0]])
    centre = range / 2
    error_loc = centre + adlo[1:3] * range
    angle_loc = error_loc + np.array([np.cos(adlo[3] * np.pi), np.sin(adlo[3] * np.pi)]) * 50

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 3, 1)
    plt.title('Map')
    plt.imshow(map_window)
    plt.axis('off')
    plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
    plt.scatter(centre[0], centre[1], c='k', s=50)
    plt.scatter(error_loc[0], error_loc[1], c='m', s=50)

    plt.subplot(1, 3, 2)
    plt.title('LDS')
    plt.imshow(lds_map, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Ground Truth')
    plt.imshow(ground_truth_map, cmap='gray')
    plt.axis('off')
    plt.show()


def show_dataset(dataset, num=5):
    for (map_window, lds_map), (ground_truth_map, adlo) in dataset.take(num):
        show_data_sample(map_window, lds_map, ground_truth_map, adlo)


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
        inputs, outputs = next(iter(dataset.batch(num)))
        preds = model.predict(inputs)
        for map_window, lds_map, ground_truth_map, adlo, map_pred, adlo_pred in zip(
                inputs[0], inputs[1], outputs[0], outputs[1], preds[0], preds[1]):
            show_prediction(map_window, lds_map, ground_truth_map, adlo, map_pred, adlo_pred, **kwargs)


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
        map_inputs = batch[0][0]
        lds_inputs = batch[0][1] if len(batch[0]) >= 2 else None
    else:
        map_inputs = batch[0]
        lds_inputs = [None] * len(map_inputs)

    if isinstance(batch[1], tuple):
        ground_truth_maps = batch[1][0]
        adlos = batch[1][1] if len(batch[1]) >= 2 else None
    else:
        ground_truth_maps = batch[1]
        adlos = [None] * len(ground_truth_maps)

    if isinstance(preds, tuple):
        map_preds = preds[0]
        adlo_preds = preds[1] if len(preds) >= 2 else None
    else:
        map_preds = preds
        adlo_preds = [None] * len(map_preds)

    for map_input, lds_input, ground_truth_map, adlo, map_pred, adlo_pred in zip(
            map_inputs, lds_inputs, ground_truth_maps, adlos, map_preds,adlo_preds):
        # print(f"inputs:  {type(inputs)} x {len(inputs)}, {np.shape(inputs)}")
        # print(f"outputs: {type(outputs)} x {len(outputs)}, {np.shape(outputs)}")
        # print(f"preds:   {type(preds)} x {len(preds)}, {np.shape(preds)}")

        print(f"map_input: {np.shape(map_input)}")
        print(f"lds_input: {np.shape(lds_input)}")
        print(f"ground_truth_map: {np.shape(ground_truth_map)}")
        print(f"adlo: {np.shape(adlo)}")
        print(f"map_pred: {np.shape(map_pred)}")
        print(f"adlo_pred: {np.shape(adlo_pred)}")
        flexi_show_prediction(map_input, lds_input, ground_truth_map, adlo, map_pred, adlo_pred, **kwargs)


def show_prediction(map_window, lds_map, ground_truth_map, adlo, map_pred, adlo_pred, **kwargs):
    from_logits = kwargs.get('from_logits', True)
    show_classes = kwargs.get('show_classes', False)  # adds extra columns (TODO)
    map_size = np.array([map_window.shape[1], map_window.shape[0]])

    # apply scaling
    map_pred_scaled = tf.nn.softmax(map_pred, axis=-1) if from_logits else map_pred
    map_pred_categorical = tf.argmax(map_pred_scaled, axis=-1)

    if from_logits:
        accept = tf.nn.sigmoid(adlo_pred[0])
        delta_x = tf.nn.tanh(adlo_pred[1]) * 0.5
        delta_y = tf.nn.tanh(adlo_pred[2]) * 0.5
        delta_angle = tf.nn.tanh(adlo_pred[3])
        adlo_pred_scaled = tf.stack([accept, delta_x, delta_y, delta_angle], axis=0)
    else:
        adlo_pred_scaled = adlo_pred

    print(f"adlo:             {adlo}")
    print(f"adlo-predicted:   {adlo_pred_scaled}")

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 5, 1)
    plt.title('Map')
    plt.imshow(map_window)
    plt.axis('off')
    if adlo is not None:
        centre = map_size / 2
        error_loc = centre + adlo[1:3] * map_size
        angle_loc = error_loc + np.array([np.cos(adlo[3] * np.pi), np.sin(adlo[3] * np.pi)]) * 50
        plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
        plt.scatter(centre[0], centre[1], c='k', s=50)
        plt.scatter(error_loc[0], error_loc[1], c='m', s=50)
    if not adlo[0]:
        # if to be rejected, add cross through map
        plt.plot([0, map_size[0] - 1], [0, map_size[1] - 1], c='y')
        plt.plot([0, map_size[0] - 1], [map_size[1] - 1, 0], c='y')

    plt.subplot(1, 5, 2)
    plt.title('LDS')
    plt.imshow(lds_map, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.title('Ground Truth')
    plt.imshow(ground_truth_map)
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.title('Predicted')
    plt.imshow(map_pred_categorical)
    plt.axis('off')
    if adlo_pred_scaled[0] < 0.5:
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

    plt.subplot(1, 5, 5)
    plt.title('Pred Raw')
    plt.imshow(map_pred_scaled)
    plt.axis('off')
    plt.plot([0, map_size[0]-1], [0, map_size[1]-1], c='y', alpha=1-adlo_pred_scaled[0].numpy())
    plt.plot([0, map_size[0]-1], [map_size[1]-1, 0], c='y', alpha=1-adlo_pred_scaled[0].numpy())

    plt.show()


# TODO merge flexibilities into show_prediction
def flexi_show_prediction(map_window, lds_map, ground_truth_map, adlo, map_pred, adlo_pred):
    # apply scaling
    map_pred_scaled = tf.nn.softmax(map_pred, axis=-1)
    map_pred_categorical = tf.argmax(map_pred, axis=-1)

    if adlo_pred is not None:
        accept = tf.nn.sigmoid(adlo_pred[0])
        delta_x = tf.nn.tanh(adlo_pred[1]) * 0.5
        delta_y = tf.nn.tanh(adlo_pred[2]) * 0.5
        delta_angle = tf.nn.tanh(adlo_pred[3])
        adlo_pred_scaled = tf.stack([accept, delta_x, delta_y, delta_angle], axis=0)
    else:
        adlo_pred_scaled = None

    range = np.array([map_window.shape[1], map_window.shape[0]])
    centre = range / 2
    if adlo is not None:
        error_loc = centre + adlo[1:3] * range
        angle_loc = error_loc + np.array([np.cos(adlo[3] * np.pi), np.sin(adlo[3] * np.pi)]) * 50
    else:
        error_loc = None
        angle_loc = None

    print(f"adlo:             {adlo}")
    print(f"adlo-predicted:   {adlo_pred_scaled}")
    plt.figure(figsize=(15, 2))

    cols = 11
    if map_window is not None:
        plt.subplot(1, cols, 1)
        plt.title('Map')
        plt.imshow(map_window)
        plt.axis('off')
        plt.scatter(centre[0], centre[1], c='k', s=50)
        if error_loc is not None:
            plt.plot([error_loc[0], angle_loc[0]], [error_loc[1], angle_loc[1]], c='m')
            plt.scatter(error_loc[0], error_loc[1], c='m', s=50)

    if lds_map is not None:
        plt.subplot(1, cols, 2)
        plt.title('LDS')
        plt.imshow(lds_map, cmap='gray')
        plt.axis('off')

    if ground_truth_map is not None:
        plt.subplot(1, cols, 3)
        plt.title('Ground Truth')
        plt.imshow(ground_truth_map)
        plt.axis('off')

        plt.subplot(1, cols, 4)
        plt.title('0')
        plt.imshow(ground_truth_map[..., 0], cmap='gray')
        plt.axis('off')

        plt.subplot(1, cols, 5)
        plt.title('0')
        plt.imshow(ground_truth_map[..., 1], cmap='gray')
        plt.axis('off')

        plt.subplot(1, cols, 6)
        plt.title('0')
        plt.imshow(ground_truth_map[..., 2], cmap='gray')
        plt.axis('off')

    if map_pred_categorical is not None:
        plt.subplot(1, cols, 7)
        plt.title('Predicted')
        plt.imshow(map_pred_categorical)
        plt.axis('off')

    if map_pred_scaled is not None:
        plt.subplot(1, cols, 8)
        plt.title('Pred Raw')
        plt.imshow(map_pred_scaled)
        plt.axis('off')

        plt.subplot(1, cols, 9)
        plt.title('0')
        plt.imshow(map_pred_scaled[..., 0], cmap='gray')
        plt.axis('off')

        plt.subplot(1, cols, 10)
        plt.title('1')
        plt.imshow(map_pred_scaled[..., 1], cmap='gray')
        plt.axis('off')

        plt.subplot(1, cols, 11)
        plt.title('2')
        plt.imshow(map_pred_scaled[..., 2], cmap='gray')
        plt.axis('off')

    plt.show()


def one_hot_encode_floorplan(image):
    """
    Converts an RGB floorplan image into a one-hot encoded tensor of the same form
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