import unittest
import tensorflow as tf
from tensorflow.keras import layers
import blog_code as blog


def run_test_suite():
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    unittest.TextTestRunner(verbosity=2).run(suite)


class CustomLayers(unittest.TestCase):
    def test_HeatmapPeakCoord(self):
        # test using symbolic tensors
        input = tf.keras.Input(shape=(149, 149, 3))
        out = blog.HeatmapPeakCoord()(input)
        print(f"symbolic output: {type(out)}, shape: {out.shape}")
        self.assertEqual(out.shape, (None, 3, 2), f"Got output shape: {out.shape}")

        # test using real tensors
        input = tf.stack([blog.generate_heatmap_image(10, 10), blog.generate_heatmap_image(10.51, 13.2),
                          blog.generate_heatmap_image(29.74, 17.432)], axis=0)
        assert input.shape == (3, 149, 149, 1)
        out = blog.HeatmapPeakCoord()(input)
        print(f"actual output: {type(out)}, shape: {out.shape}")
        self.assertEqual(out.shape, (3, 1, 2), f"Got output shape: {out.shape}")
