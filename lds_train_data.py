# Creates training data for LDS floorplan to ground-truth floorplan prediction.

import numpy as np
import cv2
import lds


def generate_training_data(semantic_map, num_samples, **kwargs):
    """
    Generates training data sampled from the semantic map.

    Keyword args:
    - max_distance: float, default: 100
        Maximum distance that an LDS can observe (unit: physical units)
    - pixel_size: float, default: 1.0.
        The size of the pixel in the desired physical unit.
        Defaults to 1.0, meaning that we output in pixel units.
    - nothing_value: bool or float, default: min value from semantic_map
        The data value that indicates nothing is present at the pixel.
        All other values are treated as pixels.
    - occupied_value: bool or float, default: max value from semantic_map
        The data value used by LDS hits.

    Returns:
    Each training data X item contains:
    - 2d-array (r,c) raw semantic_map from LDS data alone
    Each training data Y item contains:
    - 2d-array (r,c) expected semantic_map
    """
    pixel_size = kwargs.get('pixel_size', 1.0)

    train_x = []
    train_y = []
    while len(train_x) < num_samples:
        loc = np.random.uniform((0,0), semantic_map.shape) * pixel_size
        angle = np.random.uniform(-np.pi, np.pi)

        print(f"x,y = {loc}, angle={angle}")
        (lds_map, truth_map) = generate_training_data_sample(semantic_map, loc, angle, **kwargs)
        if lds_map is not None:
            train_x.append(lds_map)
            train_y.append(truth_map)

    return train_x, train_y


# For each attempted sample:
# 1. pick a random location and orientation on the map
# 2. take a sample
# 3. discard if the ranges all end up within the radius of a single pixel (ie: our random chosen location was
#    inside a wall)
# 4. generate a zeroed-out max_distance x max_distance semantic map, centered on the chosen location
# 5. populate a True/1 into the nearest-neighbour pixel corresponding to each LDS hit.
# 6. that becomes the training X data
# 7. separately, take a max_distance x max_distance square snapshot of the floorplan, centred on the chosen location.
# 8. apply a mask that zeros-out the corners - anything beyond the max_distance radius from the centre
# 9. that becomes the training Y labels
def generate_training_data_sample(semantic_map, centre, angle, **kwargs):
    """
    Generates a single training sample from the semantic map.
    Returns (None,None) if the sample is meaningless, which can occur when
    the specified center is within the bounds of the pixels (all LDS ranges have value ~0.0),
    or if there are no LDS hits (all LDS ranges are NaN).

    Parameters:
    - semantic_map: array (r,c) of bool or float
        Encoded as a 2D array of values.
    - centre: [x,y] of float
        Point from which LDS sample is taken (unit: physical units)
    - angle: float
        Starting angle of LDS sample (unit: radians)

    Keyword args:
    - max_distance: float, default: 100
        Maximum distance that an LDS can observe (unit: physical units)
    - pixel_size: float, default: 1.0.
        The size of the pixel in the desired physical unit.
        Defaults to 1.0, meaning that we output in pixel units.
    - nothing_value: bool or float, default: min value from semantic_map
        The data value that indicates nothing is present at the pixel.
        All other values are treated as pixels.
    - occupied_value: bool or float, default: max value from semantic_map
        The data value used by LDS hits.

    Returns:
    (lds_map, ground_truth_map), where:
      - lds_map: array (r,c) of bool or flat
          Square max_distance x max_distance semantic map populated from LDS data alone.
      - ground_truth_map: array (r,c) of bool or flat
          Square max_distance x max_distance masked snapshot from ground truth source semantic map.
    """
    # config
    max_distance = kwargs.get('max_distance', 100)
    pixel_size = kwargs.get('pixel_size', 1.0)
    nothing_value = kwargs.get('nothing_value', np.min(semantic_map))
    occupied_value = kwargs.get('occupied_value', np.max(semantic_map))

    # take sample
    ranges = lds.lds_sample(semantic_map, centre, angle, max_distance=max_distance, pixel_size=pixel_size,
                            nothing_value=nothing_value)
    if (np.nanmax(ranges) < pixel_size) or ranges[~np.isnan(ranges)].size == 0:
        return None, None

    # generate LDS semantic map
    # (note: from LDS data alone we don't know absolute centre or orientation, so we work relative to origin here)
    # (note: size_px is odd numbered, with equivalent of max_distance out from centre pixel)
    lds_points = lds.lds_to_2d(ranges, (0,0), 0.0)
    size_px = np.ceil(max_distance / pixel_size).astype(int) * 2 + 1
    lds_points_px = np.round(lds_points / pixel_size + (size_px-1)/2).astype(int)
    lds_map = np.full((size_px, size_px), nothing_value, dtype=semantic_map.dtype)
    lds_map[lds_points_px[:,1], lds_points_px[:,0]] = occupied_value

    # generate ground truth map
    # (here we know the centre and orientation, but we have to translate it to the same location and orientation
    #  as used by the LDS map)
    centre_px = centre / pixel_size  # sub-pixel resolution
    ground_truth_map = rotated_crop(semantic_map, centre_px, angle, size=(size_px, size_px), mask='inner-circle')

    return lds_map, ground_truth_map


# Considerations:
# - Usually operating with a large crop that spans much of the original image.
# Algorithm:
# - Rotates the full image in the opposite direction,
#   with some extra padding so we don't loose anything
# - Then takes a simple rectangular crop
# - But does have to deal with clipping at the edges
def rotated_crop(image, centre, angle, size, **kwargs):
    """
    Takes a crop of a particular size and angle from any image-like source.
    Fills in any unknown areas with zeros.

    Arguments:
    - image: array (r,c) of bool, uint8, float32, etc.
        The image to take a crop from.
    - centre: [x,y] of float
        Centre of the crop, with sub-pixel precision.
        May be outside the image bounds.
    - angle: float
        The angle of the crop in radians, counterclockwise.
    - size: [w,h] of int
        The size of the crop.

    Keyword Args:
    - mask: str (optional), default: 'none
        The mask to apply after cropping, zeroing out anything
        that isn't accepted by the mask.
        Mask is one of:
          - 'none' - do mask
          - 'inner-circle' - retains only the inner circle
    """
    # config
    mask = kwargs.get('mask', 'none')

    # handle boolean image types
    target_type = image.dtype
    if target_type == 'bool':
        image = image.astype(np.uint8)

    # pad original image so we don't lose things when we rotate it
    image_radius = np.linalg.norm(image.shape[0:2]) / 2
    pad_size_x = int(np.ceil(image_radius - image.shape[1] / 2))
    pad_size_y = int(np.ceil(image_radius - image.shape[0] / 2))
    image = cv2.copyMakeBorder(image, pad_size_y, pad_size_y, pad_size_x, pad_size_y, cv2.BORDER_CONSTANT, value=0)

    h, w = image.shape[0:2]
    centre = (centre[0] + pad_size_x, centre[1] + pad_size_y)

    # rotate whole image about the centre point
    rotation_matrix = cv2.getRotationMatrix2D(centre, -np.rad2deg(angle), scale=1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    # take crop around the centre point
    # (here we have to convert from sub-pixel resolution to pixels,
    #  we take the mathematical floor on the left/top edge of the crop.
    #  It's important to note that this places our target centre slightly biased
    #  towards the bottom/right edge of the crop)
    crop_shape = np.array(image.shape)
    crop_shape[0] = size[1]
    crop_shape[1] = size[0]
    cropped = np.full(crop_shape, 0, dtype=image.dtype)

    # the following has been verified to produce accurate results on small odd and even sizes
    x1 = max(0, int(centre[0] - (size[0] - 1) / 2))
    x2 = min(w, int(centre[0] - (size[0] - 1) / 2) + size[0])
    y1 = max(0, int(centre[1] - (size[1] - 1) / 2))
    y2 = min(h, int(centre[1] - (size[1] - 1) / 2) + size[1])

    start_x = max(0, -int(centre[0] - (size[0] - 1) / 2))
    start_y = max(0, -int(centre[1] - (size[1] - 1) / 2))
    cropped[start_y:start_y + (y2 - y1), start_x:start_x + (x2 - x1), ...] = rotated[y1:y2, x1:x2, ...]

    # apply mask
    if mask == 'none':
        pass
    elif mask == 'inner-circle':
        # the following has been verified to produce accurate results on small odd and even sizes
        crop_radius = (np.min(size) - 1) / 2
        y, x = np.ogrid[:size[1], :size[0]]
        mask = ((x - int(size[0] / 2)) ** 2 + (y - int(size[1] / 2)) ** 2) <= crop_radius ** 2
        # broadcasting doesn't cope for some reason so manually expand mask if needed
        if cropped.ndim == 3:
            mask = mask[:, :, np.newaxis]
        cropped = cropped * mask
    else:
        raise ValueError(f"Unknown mask type: {mask}")

    # convert back to target type
    if target_type == 'bool':
        cropped = cropped.astype(bool)

    return cropped