# note: python import * excludes private functions, so have to add just those explicitly
import numpy as np
from slam_data import *
from slam_data import _map_shape


def run_test_suite():
    _map_shape_test()
    get_intersect_ranges_test()
    get_intersect_ranges_tf_test()


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


def get_intersect_ranges_tf_test():
    def get_intersect_ranges_safe(map_shape1, map_shape2, offset_px):
        return get_intersect_ranges_tf(
            tf.convert_to_tensor(map_shape1),
            tf.convert_to_tensor(map_shape2),
            tf.convert_to_tensor(offset_px))

    def assert_equal(actual, expected):
        (actual1, actual2), (actual3, actual4) = actual
        (expected1, expected2), (expected3, expected4) = expected
        tf.debugging.assert_equal(actual1, expected1)
        tf.debugging.assert_equal(actual2, expected2)
        tf.debugging.assert_equal(actual3, expected3)
        tf.debugging.assert_equal(actual4, expected4)

    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 0)),
                 ((tf.range(0, 139), tf.range(0, 294)), (tf.range(0, 139), tf.range(0, 294))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (10, 0)),
                 ((tf.range(0, 139), tf.range(10, 294)), (tf.range(0, 139), tf.range(0, 284))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 10)),
                 ((tf.range(10, 139), tf.range(0, 294)), (tf.range(0, 129), tf.range(0, 294))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (10, 10)),
                 ((tf.range(10, 139), tf.range(10, 294)), (tf.range(0, 129), tf.range(0, 284))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-10, 0)),
                 ((tf.range(0, 139), tf.range(0, 284)), (tf.range(0, 139), tf.range(10, 294))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, -10)),
                 ((tf.range(0, 129), tf.range(0, 294)), (tf.range(10, 139), tf.range(0, 294))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-10, -10)),
                 ((tf.range(0, 129), tf.range(0, 284)), (tf.range(10, 139), tf.range(10, 294))))
    assert_equal(get_intersect_ranges_safe((139, 294), (100, 200), (0, 0)),
                 ((tf.range(0, 100), tf.range(0, 200)), (tf.range(0, 100), tf.range(0, 200))))
    assert_equal(get_intersect_ranges_safe((100, 200), (139, 294), (0, 0)),
                 ((tf.range(0, 100), tf.range(0, 200)), (tf.range(0, 100), tf.range(0, 200))))
    assert_equal(get_intersect_ranges_safe((50, 50), (60, 60), (45, 45)),
                 ((tf.range(45, 50), tf.range(45, 50)), (tf.range(0, 5), tf.range(0, 5))))

    assert_equal(get_intersect_ranges_safe((50, 50), (60, 60), (50, 50)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (294, 0)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-294, 0)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 139)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, -139)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (294, -139)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-294, 139)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (300, 300)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-300, -300)),
                 ((tf.range(0, 0), tf.range(0, 0)), (tf.range(0, 0), tf.range(0, 0))))


def _map_shape_test():
    assert np.array_equal(_map_shape((1, 2, 3)), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape([1, 2, 3]), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.array([1, 2, 3])), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.zeros((1, 2, 3))), np.array([1, 2, 3]))
