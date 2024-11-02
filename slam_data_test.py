import numpy as np
from slam_data import *


def run_test_suite():
    _map_shape_test()
    get_intersect_ranges_test()


def get_intersect_ranges_test():
    assert get_intersect_ranges((139, 294), (139, 294), (0, 0)) ==\
           ((slice(0, 139), slice(0, 294)), (slice(0, 139), slice(0, 294)))
    assert get_intersect_ranges((139, 294), (139, 294), (10, 0)) ==\
           ((slice(0, 139), slice(10, 294)), (slice(0, 139), slice(0, 284)))
    assert get_intersect_ranges((139, 294), (139, 294), (0, 10)) ==\
           ((slice(10, 139), slice(0, 294)), (slice(0, 129), slice(0, 294)))
    assert get_intersect_ranges((139, 294), (139, 294), (10, 10)) ==\
           ((slice(10, 139), slice(10, 294)), (slice(0, 129), slice(0, 284)))
    assert get_intersect_ranges((139, 294), (139, 294), (-10, 0)) ==\
           ((slice(0, 139), slice(0, 284)), (slice(0, 139), slice(10, 294)))
    assert get_intersect_ranges((139, 294), (139, 294), (0, -10)) ==\
           ((slice(0, 129), slice(0, 294)), (slice(10, 139), slice(0, 294)))
    assert get_intersect_ranges((139, 294), (139, 294), (-10, -10)) ==\
           ((slice(0, 129), slice(0, 284)), (slice(10, 139), slice(10, 294)))
    assert get_intersect_ranges((139, 294), (100, 200), (0, 0)) ==\
           ((slice(0, 100), slice(0, 200)), (slice(0, 100), slice(0, 200)))
    assert get_intersect_ranges((100, 200), (139, 294), (0, 0)) ==\
           ((slice(0, 100), slice(0, 200)), (slice(0, 100), slice(0, 200)))
    assert get_intersect_ranges((50, 50), (60, 60), (45, 45)) ==\
           ((slice(45, 50), slice(45, 50)), (slice(0, 5), slice(0, 5)))

    assert get_intersect_ranges((50, 50), (60, 60), (50, 50)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (294, 0)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (-294, 0)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (0, 139)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (0, -139)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (294, -139)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (-294, 139)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (300, 300)) == ((None, None), (None, None))
    assert get_intersect_ranges((139, 294), (139, 294), (-300, -300)) == ((None, None), (None, None))


def _map_shape_test():
    assert np.array_equal(_map_shape((1, 2, 3)), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape([1, 2, 3]), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.array([1, 2, 3])), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.zeros((1, 2, 3))), np.array([1, 2, 3]))
