# Animates motion of the SLAM agent.

import numpy as np
import cv2
import math
import lds


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
