# note: python import * excludes private functions, so have to add just those explicitly
import numpy as np
from slam_data import *
from slam_data import _map_shape


def run_test_suite():
    _map_shape_test()
    get_intersect_ranges_test()
    get_intersect_ranges_tf_test()
    compute_model_revisement_weight_test()
    rotated_crop_test()
    print("All slam_data tests passed.")


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


def compute_model_revisement_weight_test():
    assert DatasetRevisor.compute_model_revisement_weight(0, 100) == 0.0
    assert DatasetRevisor.compute_model_revisement_weight(99, 100) == 1.0
    assert DatasetRevisor.compute_model_revisement_weight(100, 100.0) == 1.0
    assert DatasetRevisor.compute_model_revisement_weight(200, 100.0) == 1.0


def rotated_crop_test():
    def test_image_at_90(angle):
        g = np.array([
            [0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0.],
            [1., 1., 1., 1., 1., 1., 1.],
            [0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0.],
        ])
        b = np.array([
            [1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1.],
        ])
        # add non-symmetry
        if angle == 0:
            g[3, 6] = 0
        elif angle == 90:
            g[0, 3] = 0
        elif angle == 180:
            g[3, 0] = 0
        elif angle == -90:
            g[6, 3] = 0
        else:
            raise ValueError(angle)
        r = np.ones_like(g) - g - b
        return np.dstack((r, g, b))

    def test_image_at_45(angle):
        g = np.array([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0.],
            [0., 0., 1., 0., 1., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ])
        b = np.array([
            [1., 1., 0., 0., 0., 1., 1.],
            [1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0., 1., 1.],
        ])
        # add non-symmetry
        if angle == 45:
            g[1, 5] = 0
        elif angle == 135:
            g[1, 1] = 0
        elif angle == -135:
            g[5, 1] = 0
        elif angle == -45:
            g[5, 5] = 0
        else:
            raise ValueError(angle)
        r = np.ones_like(g) - g - b
        return np.dstack((r, g, b))

    def expected_subpixel_result():
        # result after rotating by 45degrees about centre (3.4, 3.0)
        g = np.array([
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 1., 1., 0.],
            [0., 0., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 1., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
        ])
        b = np.array([
            [1., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0., 1., 1.],
        ])
        r = np.ones_like(g) - g - b
        return np.dstack((r, g, b))

    def inner_circle_mask(inp):
        mask = np.array([
            [1., 1., 1., 0., 1., 1., 1.],
            [1., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 1.],
            [1., 0., 0., 0., 0., 0., 1.],
            [1., 1., 1., 0., 1., 1., 1.],
        ])
        out = inp.copy()
        out[mask == 1.0, 0] = 0.
        out[mask == 1.0, 1] = 0.
        out[mask == 1.0, 2] = 1.
        return out

    def shift_image(inp, shift):
        dx, dy = shift
        start_src_x = max(0, 0 - dx)
        end_src_x = min(inp.shape[1], -dx + inp.shape[1])
        start_src_y = max(0, 0 - dy)
        end_src_y = min(inp.shape[0], -dy + inp.shape[0])

        start_tgt_x = start_src_x + dx
        end_tgt_x = end_src_x + dx
        start_tgt_y = start_src_y + dy
        end_tgt_y = end_src_y + dy

        rg = np.zeros(shape=inp.shape[0:2])
        b = np.ones(shape=inp.shape[0:2])
        out = np.dstack((rg, rg, b))
        out[start_tgt_y:end_tgt_y, start_tgt_x:end_tgt_x] = inp[start_src_y:end_src_y, start_src_x:end_src_x]
        return out

    def extend_image(inp, pad_value):
        """ Inserts a column at the front and a row at the bottom """
        # insert column at front
        inp = np.insert(inp, 0, pad_value, axis=1)
        # insert row at bottom
        inp = np.insert(inp, inp.shape[0], pad_value, axis=0)
        return inp

    # 90-degree rotations without mask, pixel-aligned
    pad_value = np.array([0., 0., 1.])
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(0))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(90), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(-90))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(180), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(180))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-90), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(90))

    # 45-degree rotations without mask, pixel-aligned
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(45), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_45(-45))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(135), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_45(-135))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-135), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_45(135))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-45), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_45(45))

    # 90-degree rotations with mask, pixel-aligned
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(0), size=(7, 7), mask='inner-circle',pad_value=pad_value),
        inner_circle_mask(test_image_at_90(0)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(90), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_90(-90)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(180), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_90(180)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-90), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_90(90)))

    # 45-degree rotations with mask, pixel-aligned
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(45), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_45(-45)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(135), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_45(-135)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-135), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_45(135)))
    assert np.array_equal(
        rotated_crop(img, (3, 3), np.deg2rad(-45), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_45(45)))

    # various crops from a larger image, pixel-aligned
    img = extend_image(test_image_at_90(0), pad_value)
    assert np.array_equal(
        rotated_crop(img, (4, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(0))
    assert np.array_equal(
        rotated_crop(img, (4, 3), np.deg2rad(90), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_90(-90))
    assert np.array_equal(
        rotated_crop(img, (4, 3), np.deg2rad(45), size=(7, 7), mask='none', pad_value=pad_value),
        test_image_at_45(-45))
    assert np.array_equal(
        rotated_crop(img, (4, 3), np.deg2rad(-45), size=(7, 7), mask='inner-circle', pad_value=pad_value),
        inner_circle_mask(test_image_at_45(45)))

    # subpixel support - simple sub-pixel translations
    # (note: with interpolation on you get even more interesting results, but it's harder to define the
    #  expected values so I'm not bothering)
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (2.4, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (+1, 0)))
    assert np.array_equal(
        rotated_crop(img, (2.6, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, 0)))
    assert np.array_equal(
        rotated_crop(img, (3.4, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, 0)))
    assert np.array_equal(
        rotated_crop(img, (3.6, 3), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (-1, 0)))
    assert np.array_equal(
        rotated_crop(img, (3, 2.4), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, +1)))
    assert np.array_equal(
        rotated_crop(img, (3, 2.6), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, 0)))
    assert np.array_equal(
        rotated_crop(img, (3, 3.4), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, 0)))
    assert np.array_equal(
        rotated_crop(img, (3, 3.6), np.deg2rad(0), size=(7, 7), mask='none', pad_value=pad_value),
        shift_image(img, (0, -1)))

    # subpixel support - sub-pixel translation with rotation
    img = test_image_at_90(0)
    assert np.array_equal(
        rotated_crop(img, (3.4, 3.0), np.deg2rad(45), size=(7, 7), mask='none', pad_value=pad_value),
        expected_subpixel_result())



