# LDS (Laser Distance Sensor) emulation

# Assumptions in approach:
# - Laser traces are have infinitesimal width
# - All decimal-valued coordinates identify infinitesimal width positions somewhere within the width of the pixel
#   that it overlays. With #.0 at the near-origin side of the pixel, and #.9999 at the opposite side of the same pixel.
# - All integer-valued coordinates identify a pixel-width square spanning the range (#.0,#.0) to (#.9999,#.9999) and
#   with its centre at (#.5,#.5).
#
# For collision with individual pixels, the approach is:
# - Treat each pixel as a circle, having a centre in the middle of the pixel and a diameter equal to the diagonal of
#   the pixel (ie:  2–√ ).
# - Additionally, each pixel has a holographic line that always passes through the centre and has a normal in the
#   direction of the trace. The line extends out from the centre in both directions with radius equal to the radius
#   of the pixel.
# - A collision in determined by doing the single linear algebra dot-product trick to identify the intersection point
#   between the trace and the pixel's line, and then to check whether the intersection is within the line radius.
# - When selecting between potential collisions, the min distance to intersection point is taken.
#
# Setup:
# - step_size = position increment used by traces, in their direction
# - grid_size = distance between centres of nearest-neighbours cache grid
# - step_size = grid_size
# - Nearest-neighbours bubble size: radius = grid_size
# This ensures that the bubbles overlap considerably on horizontal/vertical and diagonal axes.
# It's possible that it'd work with 0.5sqrt(2)step-size - so that the bubbles meet exactly on the diagonal and
# overlap a little on the horizontal/vertical. However there's potentially an edge case with traces jumping
# semi-diagonally over a bubble and missing potential collisions. The larger bubble size resolves this, I believe.

import math
import numpy as np


def lds_to_2d(ranges, centre, start_angle):
    """
    Converts LDS range data to Euclidean coordinates.
    :param ranges: array (n,) - range values may contain nans, which are ignored
    :param centre: array (2,) = [x,y] - centre point for ranges
    :param start_angle: angle of first range (radians)
    :return: array (n,2) of [x,y] coords
    """
    angles = np.linspace(0, np.pi * 2, num=ranges.shape[0], endpoint=False) + start_angle
    steps = np.column_stack((np.cos(angles), np.sin(angles)))
    points = steps * ranges.reshape(-1, 1) + centre
    return points


# Algorithm:
# 1. Initialise a grid of cached nearest-neighbour "bubbles", having a circular shape and containing a count + list of
#    all pixels within its area.
# 2. Initialise 360 traces, with:
#    - initial position at centre
#    - step (x,y) calculated according to their angle and the step size
#    - a value indicating how much of the trace has been consumed (in range 0.0 to 1.0)
#    - a value indicating its current range (initially NaN)
# 3. Proceed with all traces simultaneously:
#    1. Take one step
#    2. Identify the single cache bubble having the nearest centre by doing a round(x,y)/grid_size operation against
#       the trace's position to identify the array lookups into the cache for each trace
# 4. For each trace that hasn't already been consumed and for which the cache has pixels:
#    1. Take all the pixels in the cache bubble and find the collision, if any, with the min distance
#    2. For all traces that have had collisions, mark them as consumed.
def lds_sample(data, centre, angle=0.0, **kwargs):
    """
    Generates LDS data sampled from a given centre position within an architectural image.
    LDS data represents a sampling across a 360 degree clockwise spread, starting on the requested angle.

    Parameters:
    - data: array (r,c) of bool or float
        Architectural image
    - centre: [x,y] of float
        Point from which LDS sample is taken
    - angle: radians
        Starting angle of LDS sample

    Keyword args:
    - resolution: float, default: 1 degree (360 traces)
        Angle between each trace (radians)
    - step_size: float, default: 5
        Position increment used by traces, in their direction
    - max_distance: float, default: 100

    Returns:
    - ranges: array (n,) of float
        LDS data for each sample angle, or NaN for no hit
    """

    # load options
    resolution = kwargs.get('resolution', np.deg2rad(1.0))
    step_size = kwargs.get('step_size', 5)
    max_distance = kwargs.get('max_distance', 100)

    # initialise cache
    grid_size = step_size
    grid = construct_nn_grid(data, grid_size, grid_radius=step_size)
    grid_counts = np.array([[grid[r, c]['count'] for c in range(grid.shape[1])] for r in range(grid.shape[0])])

    # initialise traces
    num_traces = int(np.round(2 * np.pi / resolution))
    angles = np.linspace(0, np.pi * 2, num=num_traces, endpoint=False) + angle
    ranges = np.full((num_traces,), np.nan)
    steps = np.column_stack((np.cos(angles), np.sin(angles))) * step_size
    max_x = data.shape[1] - 1
    max_y = data.shape[0] - 1

    for step_i in range(math.ceil(max_distance / step_size)):
        # move all trace points
        points = steps * step_i + centre

        # identify applicable grid blocks and which to traces to execute against
        #  - filter: only process for traces that haven't already been consumed
        #  - filter: don't go outside bounds of data
        #  - filter: ignore grid blocks with no pixels
        grid_xs = np.round(points[:, 0] / grid_size).astype(int)
        grid_ys = np.round(points[:, 1] / grid_size).astype(int)
        filter = np.isnan(ranges)
        filter &= (points[:, 0] >= 0) & (points[:, 0] <= max_x)
        filter &= (points[:, 1] >= 0) & (points[:, 1] <= max_y)
        if np.sum(filter) == 0:
            continue  # skip if there's nothing to do
        has_counts = grid_counts[(grid_ys[filter], grid_xs[filter])] > 0
        filter[filter] = has_counts
        if np.sum(filter) == 0:
            continue  # skip if there's nothing to do

        # process each accepted block
        for idx in np.where(filter)[0]:
            intersection, distance, pixel_coord, pixel_value = find_collision(
                centre, steps[idx], grid[grid_ys[idx], grid_xs[idx]])
            if not np.isnan(distance):
                ranges[idx] = distance

    # truncate ranges to max distance
    ranges[ranges > max_distance] = np.nan

    return ranges


def construct_nn_grid(data, grid_size, grid_radius=None, **kwargs):
    """
    Constructs a lookup array populated with lists of nearest-neighbour pixels
    located within circular regions around a grid of centres.

    Parameters:
    - data: array(r,c) of bool or float
        An image that represents a 2D world of pixel-sized objects having a single floating value each.
    - grid_size: float
        The spacing between each centre.
    - grid_radius: float, optional (default: same as grid_size)
        The radius of each circular region.

    Keyword args:
    - nothing_value: float, optional (default: 0.0).
        The data value that indicates nothing is present at the pixel.
        All other values are treated as pixels.

    Returns:
    - 2-array of dicts {count: int, pixel_coords: array(N,2), pixel_values: array(N,)}
        Constructed grid of nearest-neighbour results.
        Each array position lists the number of non-empty pixels, their coordinates
        as [[x,y]], and their values.
    """

    # setup
    grid_radius = grid_radius or grid_size
    nothing_value = kwargs.get('nothing_value', 0.0)
    max_x = data.shape[1] - 1
    max_y = data.shape[0] - 1
    rows = math.ceil(data.shape[0] / grid_radius) + 1  # so that ceil(len/radius) is last index
    cols = math.ceil(data.shape[1] / grid_radius) + 1  # so that ceil(len/radius) is last index
    grid = np.empty((rows, cols), dtype=object)

    for yi in range(rows):
        for xi in range(cols):
            # identify grid location
            centre_x = xi * grid_size
            centre_y = yi * grid_size
            left = max(0, centre_x - grid_radius)  # inclusive
            right = min(max_x, centre_x + grid_radius)  # inclusive
            top = max(0, centre_y - grid_radius)  # inclusive
            bottom = min(max_y, centre_y + grid_radius)  # inclusive

            # fetch square-shaped block
            # (pixel coordinates are computed at their centres)
            pixel_values = data[top:bottom + 1, left:right + 1].ravel()
            xs, ys = np.meshgrid(np.arange(left + 0.5, right + 1.5), np.arange(top + 0.5, bottom + 1.5), indexing='xy')
            pixel_coords = np.column_stack((xs.ravel(), ys.ravel()))

            # filter block: remove all empty pixels
            filter = pixel_values != nothing_value
            pixel_values = pixel_values[filter]
            pixel_coords = pixel_coords[filter]

            # filter block: remove everything outside of grid_radius
            filter = np.sum((pixel_coords - [centre_x, centre_y]) ** 2, axis=1) <= grid_radius ** 2
            pixel_values = pixel_values[filter]
            pixel_coords = pixel_coords[filter]

            grid[yi, xi] = {
                'count': pixel_values.shape[0],
                'pixel_coords': pixel_coords,
                'pixel_values': pixel_values
            }
    return grid


# Algorithm:
# Compute for all pixels within the cache bubble simultaneously
# Compute the intersection points with each pixel's holographic line.
# Filter to remove all intersections that are > pixel radius from the pixel centre.
# Compute the distances to each remaining intersection point.
# Take the min distance, if any remain.
def find_collision(start, direction, pixels):
    """
    Finds the closest collision, if any, between a trace and a collection of pixels.

    Arguments:
    - start: array(1,2) = [x,y]
        Starting point of trace
    - direction: array(1,2) = (dx,dy)
        Direction of trace (any scale)
    - pixels: dict{pixel_coords, pixel_values} an entry as produced by construct_nn_grid()

    Returns tuple containing:
    - intersection: array(1,2)=[x,y]
        Coord of intersection, or nan otherwise
    - distance: float
        Trace distance from start to intersection point, or nan otherwise
    - pixel_coord: array(1,2)=[x,y]
    - pixel_value: bool or float
    """
    intersection = np.nan
    distance = np.nan
    pixel_coord = np.nan
    pixel_value = np.nan

    # compute intersection points with all pixels
    # - pixel_direction: a line through pixel centre, orthogonal to the trace line,
    #                    scaled to the length of a pixel_radius so we can easily use it to determine length
    pixel_coords = pixels['pixel_coords']
    pixel_values = pixels['pixel_values']
    pixel_radius = math.sqrt(0.5)  # diagonal distance from centre of pixel to corner
    pixel_direction = [direction[1], -direction[0]]
    pixel_direction = pixel_direction / np.linalg.norm(
        pixel_direction) * pixel_radius  # rescale to be multiplies of pixel_radius
    pixel_t = np.dot(start - pixel_coords, pixel_direction) / np.dot(pixel_direction, pixel_direction)

    # filter: intersection point must be within pixel_radius from pixel_centre (|t| <= 1.0)
    filter = abs(pixel_t) <= 1.0
    if np.sum(filter) > 0:
        # compute intersections
        intersections = pixel_coords[filter] + pixel_t[filter].reshape(-1, 1) * pixel_direction

        # filter: intersection must be in positive side of trace direction
        trace_t = np.dot(pixel_coords[filter] - start, direction) / np.dot(direction, direction)
        intersections = intersections[trace_t >= 0.0]

        # select intersection with min distance
        if intersections.size > 0:
            distances = np.linalg.norm(intersections - start, axis=1)
            idx = np.argmin(distances)
            intersection = intersections[idx]
            distance = distances[idx]
            pixel_coord = pixel_coords[filter][idx]
            pixel_value = pixel_values[filter][idx]

    return intersection, distance, pixel_coord, pixel_value