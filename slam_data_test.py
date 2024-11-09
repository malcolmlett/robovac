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

    def range_indices(row_start, row_end, col_start, col_end):
        rows, cols = tf.meshgrid(tf.range(row_start, row_end), tf.range(col_start, col_end), indexing='ij')
        return tf.stack([rows, cols], axis=-1)

    def assert_equal(actual, expected):
        actual1, actual2 = actual
        expected1, expected2 = expected
        actual1 = dim_squeeze(actual1)
        actual2 = dim_squeeze(actual2)
        expected1 = dim_squeeze(expected1)
        expected2 = dim_squeeze(expected2)
        tf.debugging.assert_equal(actual1.shape, expected1.shape)
        tf.debugging.assert_equal(actual2.shape, expected2.shape)
        tf.debugging.assert_equal(actual1, expected1)
        tf.debugging.assert_equal(actual2, expected2)

    # deal with the fact that the ranges calculation happily returns
    # empty tensors with shapes like [139, 0, 2]
    def dim_squeeze(tensor):
        return tf.zeros([0, 0, 0], dtype=tensor.dtype) if tf.size(tensor) == 0 else tensor

    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 00)),
                 (range_indices(0, 139, 0, 294), range_indices(0, 139, 0, 294)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (10, 0)),
                 (range_indices(0, 139, 10, 294), range_indices(0, 139, 0, 284)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 10)),
                 (range_indices(10, 139, 0, 294), range_indices(0, 129, 0, 294)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (10, 10)),
                 (range_indices(10, 139, 10, 294), range_indices(0, 129, 0, 284)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-10, 0)),
                 (range_indices(0, 139, 0, 284), range_indices(0, 139, 10, 294)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, -10)),
                 (range_indices(0, 129, 0, 294), range_indices(10, 139, 0, 294)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-10, -10)),
                 (range_indices(0, 129, 0, 284), range_indices(10, 139, 10, 294)))
    assert_equal(get_intersect_ranges_safe((139, 294), (100, 200), (0, 0)),
                 (range_indices(0, 100, 0, 200), range_indices(0, 100, 0, 200)))
    assert_equal(get_intersect_ranges_safe((100, 200), (139, 294), (0, 0)),
                 (range_indices(0, 100, 0, 200), range_indices(0, 100, 0, 200)))
    assert_equal(get_intersect_ranges_safe((50, 50), (60, 60), (45, 45)),
                 (range_indices(45, 50, 45, 50), range_indices(0, 5, 0, 5)))

    assert_equal(get_intersect_ranges_safe((50, 50), (60, 60), (50, 50)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (294, 0)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-294, 0)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, 139)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (0, -139)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (294, -139)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-294, 139)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (300, 300)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))
    assert_equal(get_intersect_ranges_safe((139, 294), (139, 294), (-300, -300)),
                 (range_indices(0, 0, 0, 0), range_indices(0, 0, 0, 0)))


def _map_shape_test():
    assert np.array_equal(_map_shape((1, 2, 3)), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape([1, 2, 3]), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.array([1, 2, 3])), np.array([1, 2, 3]))
    assert np.array_equal(_map_shape(np.zeros((1, 2, 3))), np.array([1, 2, 3]))
