# LDS (Laser Distance Sensor) emulation

# Assumptions in approach:
# - Laser traces are have infinitesimal width
# - All integer-valued coordinates identify a pixel-width square centered on its coordinate (#.0,#.0) and
#   spanning the range >= (-#.5,-#.5) to < (+#.5,+#.5).
# - All decimal-valued coordinates are taken to have infinitesimal width.
# - Note that the above aligns with many plotting libraries, including matplotlib.imshow(), so that you can overlay
#   decimal-valued coordinates directly over the image without conversions.
#
# For collision with individual pixels, the approach is:
# - Treat each pixel as a circle, having a centre at its coordinate and a diameter equal to the diagonal of
#   the pixel (ie: `sqrt(2)`).
# - Additionally, each pixel has a holographic line that always passes through the centre and has a normal in the
#   direction of the trace. The line extends out from the centre in both directions with radius equal to the radius
#   of the pixel.
# - A collision in determined by taking a dot-product to identify the intersection point
#   between the trace and the pixel's line, and then to check whether the intersection is within the line radius.
# - When selecting between potential collisions, the min distance to intersection point is taken.
# - We use metres as the unit for physical space coordinates.
#
# Common parameters:
# - step_size = position increment used by traces, in their direction
# - grid_size = distance between centres of nearest-neighbours cache grid
# - pixel_size = size of pixels in meters
# Typically step_size = grid_size, and nearest-neighbours bubble radius = grid_size.
# This ensures that the bubbles overlap considerably on horizontal/vertical and diagonal axes.
# It's possible that it'd work with 0.5sqrt(2)step-size - so that the bubbles meet exactly on the diagonal and
# overlap a little on the horizontal/vertical. However, there's potentially an edge case with traces jumping
# semi-diagonally over a bubble and missing potential collisions. The larger bubble size resolves this, I believe.

import math
import numpy as np


def lds_to_2d(ranges, centre, start_angle):
    """
    Converts LDS range data to Euclidean coordinates.
    Applies a unit-less conversion, retaining the same unit in the 2D coords as used by the range values.
    :param ranges: array (n,) - range values may contain nans, which are dropped
    :param centre: array (2,) = [x,y] - centre point for ranges
    :param start_angle: angle of first range (radians)
    :return: array (n,2) of [x,y] coords (without nans)
    """
    angles = np.linspace(0, np.pi * 2, num=ranges.shape[0], endpoint=False) + start_angle
    steps = np.column_stack((np.cos(angles), np.sin(angles)))
    points = steps * ranges.reshape(-1, 1) + centre
    return points[~np.isnan(ranges)]


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
def lds_sample(semantic_map, centre, angle=0.0, **kwargs):
    """
    Generates LDS data sampled from a floor plan from a given centre position and reference angle.
    LDS data represents a sampling across a 360 degree clockwise spread, starting on the requested angle.

    Coordinates are specified in an "output unit", with a conversion from pixels to the output
    unit defined by `pixel_size` (default: 1.0).

    Parameters:
    - semantic_map: array (r,c) of bool or float
        Encoded as a 2D array of values.
    - centre: [x,y] of float
        Point from which LDS sample is taken (unit: output units)
    - angle: float
        Starting angle of LDS sample (unit: radians)

    Keyword args:
    - resolution: float, default: 1 degree (360 traces)
        Angle between each trace (radians)
    - step_size: float, default: 5
        Position increment used by traces, in their direction (unit: output units)
    - max_distance: float, default: 100
        Maximum distance that an LDS can observe (unit: output units)
    - nothing_value: float, default: 0.0.
        The data value that indicates nothing is present at the pixel.
        All other values are treated as pixels.
    - pixel_size: float, default: 1.0.
        The size of the pixel in the desired output unit.
        Defaults to 1.0, meaning that we output in pixel units.
        Alternatively, for example, provide the width of a pixel in meters
        in order to work with meter coordinates.

    Returns:
    - ranges: array (n,) of float
        LDS data for each sample angle, or NaN for no hit
        (unit: output units).
    """

    # config
    resolution = kwargs.get('resolution', np.deg2rad(1.0))
    step_size = kwargs.get('step_size', 5)
    max_distance = kwargs.get('max_distance', 100)
    nothing_value = kwargs.get('nothing_value', 0.0)
    pixel_size = kwargs.get('pixel_size', 1.0)

    # initialise lookup cache
    grid_size_px = math.floor(step_size / pixel_size)  # conservatively prefer regions slightly closer together
    grid_radius_px = math.ceil(step_size / pixel_size)  # conservatively prefer regions slightly larger
    grid_size = grid_size_px * pixel_size  # in output unit, used for lookups later on
    grid = construct_nn_grid(semantic_map, grid_size_px, grid_radius=grid_radius_px, nothing_value=nothing_value,
                             pixel_size=pixel_size)
    grid_counts = np.array([[grid[r, c]['count'] for c in range(grid.shape[1])] for r in range(grid.shape[0])])

    # initialise traces
    num_traces = int(np.round(2 * np.pi / resolution))
    angles = np.linspace(0, np.pi * 2, num=num_traces, endpoint=False) + angle
    ranges = np.full((num_traces,), np.nan)
    steps = np.column_stack((np.cos(angles), np.sin(angles))) * step_size

    for step_i in range(math.ceil(max_distance / step_size)):
        # move all trace points
        points = steps * step_i + centre  # in output_units

        # identify applicable grid blocks and which traces to execute against
        #  - filter: only process for traces that haven't already been consumed
        #  - filter: don't go outside bounds of data
        #  - filter: ignore grid blocks with no pixels
        grid_cols = np.round(points[:, 0] / grid_size).astype(int)
        grid_rows = np.round(points[:, 1] / grid_size).astype(int)
        mask = np.isnan(ranges)
        mask &= (grid_cols >= 0) & (grid_cols < grid.shape[1])
        mask &= (grid_rows >= 0) & (grid_rows < grid.shape[0])
        if np.sum(mask) == 0:
            continue  # skip if there's nothing to do
        has_counts = grid_counts[(grid_rows[mask], grid_cols[mask])] > 0
        mask[mask] = has_counts
        if np.sum(mask) == 0:
            continue  # skip if there's nothing to do

        # process each accepted block
        for idx in np.where(mask)[0]:
            intersection, distance, pixel_coord, pixel_value = find_collision(
                centre, steps[idx], grid[grid_rows[idx], grid_cols[idx]], pixel_size=pixel_size)
            if not np.isnan(distance):
                ranges[idx] = distance

    # truncate ranges to max distance
    ranges[ranges > max_distance] = np.nan

    return ranges


def construct_nn_grid(semantic_map, grid_size, grid_radius=None, **kwargs):
    """
    Constructs a lookup array populated with lists of nearest-neighbour pixels
    located within circular regions around a grid of centres.

    By default, returns coordinates in pixel units.
    Supply a pixel_size parameter to convert to any other unit.

    Parameters:
    - semantic_map: array(r,c) of bool or float
        An image that represents a 2D world of pixel-sized objects having a single value each to classify different
        kinds of objects.
    - grid_size: float
        The spacing between each centre (unit: pixels)
    - grid_radius: float, optional (default: same as grid_size)
        The radius of each circular region (unit: pixels)

    Keyword args:
    - nothing_value: float, default: 0.0.
        The data value that indicates nothing is present at the pixel.
        All other values are treated as pixels.
    - pixel_size: float, default: 1.0.
        The size of the pixel in the desired output unit.
        Defaults to 1.0 meaning that we output in pixel units.
        Alternatively, for example, provide the width of a pixel in meters
        in order to work with meter coordinates subsequently.

    Returns:
    - 2-array(r,c) of dicts {count: int, pixel_coords: array(N,2), pixel_values: array(N,)}
        Constructed grid of nearest-neighbour results.
        Each array position lists the number of non-empty pixels, their coordinates
        as [[x,y]] (in pixel_size units), and their values.
    """

    # config
    grid_radius = grid_radius or grid_size
    nothing_value = kwargs.get('nothing_value', 0.0)
    pixel_size = kwargs.get('pixel_size', 1.0)

    # setup
    max_x = semantic_map.shape[1] - 1
    max_y = semantic_map.shape[0] - 1
    rows = math.ceil(semantic_map.shape[0] / grid_size) + 1  # so that ceil(len/size) is last index
    cols = math.ceil(semantic_map.shape[1] / grid_size) + 1  # so that ceil(len/size) is last index
    grid = np.empty((rows, cols), dtype=object)

    for yi in range(rows):
        for xi in range(cols):
            # identify grid location
            centre_x = xi * grid_size
            centre_y = yi * grid_size
            left = max(0, centre_x - grid_radius)        # inclusive
            right = min(max_x, centre_x + grid_radius)   # inclusive
            top = max(0, centre_y - grid_radius)         # inclusive
            bottom = min(max_y, centre_y + grid_radius)  # inclusive

            # fetch square-shaped block
            # (pixel coordinates represent their centres)
            xs, ys = np.meshgrid(np.arange(left, right+1), np.arange(top, bottom+1), indexing='xy')
            pixel_coords = np.column_stack((xs.ravel(), ys.ravel()))
            pixel_values = semantic_map[top:bottom + 1, left:right + 1].ravel()

            # filter block: remove all empty pixels
            mask = pixel_values != nothing_value
            pixel_coords = pixel_coords[mask]
            pixel_values = pixel_values[mask]

            # filter block: remove everything outside of grid_radius
            mask = np.sum((pixel_coords - [centre_x, centre_y]) ** 2, axis=1) <= grid_radius ** 2
            pixel_coords = pixel_coords[mask]
            pixel_values = pixel_values[mask]

            grid[yi, xi] = {
                'count': pixel_values.shape[0],
                'pixel_coords': pixel_coords * pixel_size,
                'pixel_values': pixel_values
            }
    return grid


# Algorithm:
# Compute for all pixels within the cache bubble simultaneously
# Compute the intersection points with each pixel's holographic line.
# Filter to remove all intersections that are > pixel radius from the pixel centre.
# Compute the distances to each remaining intersection point.
# Take the min distance, if any remain.
def find_collision(start, direction, pixels, **kwargs):
    """
    Finds the closest collision, if any, between a trace and a collection of pixels.
    Applies unit-less computations, retaining the same unit in output as used by the inputs,
    which must all use the same units.

    Arguments:
    - start: array(1,2) = [x,y]
        Starting point of trace (unit: output units)
    - direction: array(1,2) = (dx,dy)
        Direction of trace (any scale)
    - pixels: dict{pixel_coords, pixel_values}
        An entry as produced by construct_nn_grid() (unit: output units)

    Keyword args:
    - pixel_size: float, default: 1.0.
        The size of the pixel in the desired output unit.
        Defaults to 1.0 meaning that we output in pixel units.
        Alternatively, for example, provide the width of a pixel in meters
        in order to work with meter coordinates subsequently.

    Returns tuple containing:
    - intersection: array(1,2)=[x,y]
        Coord of intersection, or nan otherwise (unit: output units)
    - distance: float
        Trace distance from start to intersection point, or nan otherwise (unit: output units)
    - pixel_coord: array(1,2)=[x,y]
        (unit: output units)
    - pixel_value: bool or float
    """
    # config
    pixel_size = kwargs.get('pixel_size', 1.0)

    # setup
    intersection = np.nan
    distance = np.nan
    pixel_coord = np.nan
    pixel_value = np.nan

    # compute intersection points with all pixels
    # - pixel_direction: a line through pixel centre, orthogonal to the trace line,
    #                    scaled to the length of a pixel_radius so we can easily use it to determine length
    pixel_coords = pixels['pixel_coords']
    pixel_values = pixels['pixel_values']
    pixel_radius = math.sqrt(0.5) * pixel_size  # diagonal distance from centre of pixel to corner
    pixel_direction = [direction[1], -direction[0]]
    pixel_direction = pixel_direction / np.linalg.norm(
        pixel_direction) * pixel_radius  # rescale to be multiplies of pixel_radius
    pixel_t = np.dot(start - pixel_coords, pixel_direction) / np.dot(pixel_direction, pixel_direction)

    # filter: intersection point must be within pixel_radius from pixel_centre (|t| <= 1.0)
    mask = abs(pixel_t) <= 1.0
    if np.sum(mask) > 0:
        # compute intersections
        intersections = pixel_coords[mask] + pixel_t[mask].reshape(-1, 1) * pixel_direction

        # filter: intersection must be in positive side of trace direction
        trace_t = np.dot(pixel_coords[mask] - start, direction) / np.dot(direction, direction)
        intersections = intersections[trace_t >= 0.0]

        # select intersection with min distance
        if intersections.size > 0:
            distances = np.linalg.norm(intersections - start, axis=1)
            idx = np.argmin(distances)
            intersection = intersections[idx]
            distance = distances[idx]
            pixel_coord = pixel_coords[mask][idx]
            pixel_value = pixel_values[mask][idx]

    return intersection, distance, pixel_coord, pixel_value
