import unittest
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import blog_code as blog


def run_test_suite():
    suite = unittest.defaultTestLoader.loadTestsFromName(__name__)
    unittest.TextTestRunner(verbosity=2).run(suite)


class DataGeneration(unittest.TestCase):
    def test_heatmap_coordinates_roundtrip(self):
        # batch size = 3, channels = 1 -> pixels
        images = tf.stack([
            blog.generate_heatmap_image(10, 10),
            blog.generate_heatmap_image(10.51, 13.2),
            blog.generate_heatmap_image(29.74, 17.432)], axis=0)
        coords = blog.weighted_peak_coordinates(images, system='pixels')
        self.assertEqual(images.shape, (3, 149, 149, 1))
        self.assertEqual(coords.shape, (3, 1, 2))
        self.assertTrue(np.allclose(coords, [[[10.0, 10.0]], [[10.51, 13.2]], [[29.74, 17.432]]]))

        # batch size = 1, channels = 3 -> pixels
        images = tf.stack([
            blog.generate_heatmap_image(10, 10),
            blog.generate_heatmap_image(10.51, 13.2),
            blog.generate_heatmap_image(29.74, 17.432)], axis=-1)
        images = tf.reshape(images, [1, 149, 149, -1])
        coords = blog.weighted_peak_coordinates(images, system='pixels')
        self.assertEqual(images.shape, (1, 149, 149, 3))
        self.assertEqual(coords.shape, (1, 3, 2))
        self.assertTrue(np.allclose(coords, [[[10.0, 10.0], [10.51, 13.2], [29.74, 17.432]]]))

        # batch size = 3, channels = 1 -> unit-scale
        images = tf.stack([
            blog.generate_heatmap_image(10, 10),
            blog.generate_heatmap_image(10.51, 13.2),
            blog.generate_heatmap_image(29.74, 17.432)], axis=0)
        coords = blog.weighted_peak_coordinates(images, system='unit-scale')
        expected = np.array([[[10.0, 10.0]], [[10.51, 13.2]], [[29.74, 17.432]]])
        expected = (expected - 149//2) / 149
        self.assertEqual(images.shape, (3, 149, 149, 1))
        self.assertEqual(coords.shape, (3, 1, 2))
        self.assertTrue(np.allclose(coords, expected))


class CustomLayers(unittest.TestCase):
    def test_HeatmapPeakCoord(self):
        # test using symbolic tensors
        input = tf.keras.Input(shape=(149, 149, 3))
        out = blog.HeatmapPeakCoord()(input)
        self.assertEqual(out.shape, (None, 3, 2), f"Got output shape: {out.shape}")

        # test using real tensors
        input = tf.stack([blog.generate_heatmap_image(10, 10), blog.generate_heatmap_image(10.51, 13.2),
                          blog.generate_heatmap_image(29.74, 17.432)], axis=0)
        assert input.shape == (3, 149, 149, 1)
        out = blog.HeatmapPeakCoord()(input)
        self.assertEqual(out.shape, (3, 1, 2), f"Got output shape: {out.shape}")

    def test_CoordGrid(self):
        input = tf.zeros(shape=(32, 16, 16, 3))
        out = blog.CoordGrid2D()(input)
        self.assertEqual(out.shape, (32, 16, 16, 2), f"Got output shape: {out.shape}")
        self.assertTrue(np.allclose(out[0, 0, :, 0], np.linspace(-0.5, 0.5, 16)))
        self.assertTrue(np.allclose(out[0, :, 0, 1], np.linspace(-0.5, 0.5, 16)))

    def test_PositionwiseMaxPool2D(self):
        # Combining coord grid with channel_mask
        input = tf.constant([[
            [[0.5, 0.5, 0.3], [0.1, 0.2, 0.3], [0.5, 0.3, 0.5], [0.7, 0.8, 0.9]],
            [[0.4, 0.3, 0.5], [0.3, 0.3, 0.3], [0.4, 0.3, 0.5], [0.3, 0.3, 0.3]],
            [[0.1, 0.2, 0.3], [0.0, 0.0, 0.3], [0.3, 0.5, 0.5], [0.6, 0.6, 0.7]],
            [[0.2, 0.3, 0.2], [0.0, 0.0, 0.3], [0.4, 0.3, 0.5], [0.5, 0.6, 0.5]]
        ]])
        expected1 = tf.constant([[
            [[0.5, 0.5, 0.3, -0.5, -0.5], [0.7, 0.8, 0.9, +0.5, -0.5]],
            [[0.2, 0.3, 0.2, -0.5, +0.5], [0.6, 0.6, 0.7, +0.5, 1 / 6.]]
        ]])
        coords = blog.CoordGrid2D()(input)
        x = layers.Concatenate()([input, coords])
        x = blog.PositionwiseMaxPool2D(channel_weights=[1, 1, 1, 0, 0])(x)
        self.assertTrue(np.allclose(x.numpy(), expected1.numpy()))

        # Combining stride grid with channel_mask
        expected2 = tf.constant([[
            [[0.4, 0.3, 0.5, -0.5, +0.5], [0.7, 0.8, 0.9, +0.5, -0.5]],
            [[0.1, 0.2, 0.3, -0.5, -0.5], [0.6, 0.6, 0.7, +0.5, -0.5]],
        ]])
        coords = blog.StrideGrid2D()(input)
        x = layers.Concatenate()([input, coords])
        x = blog.PositionwiseMaxPool2D(channel_weights=[0, 0, 1, 0, 0])(x)
        self.assertTrue(np.allclose(x.numpy(), expected2.numpy()))

    def test_AttentionPool2D(self):
        # test using symbolic tensors
        input = tf.keras.Input(shape=(32, 32, 3))
        features = blog.StrideGrid2D()(input)
        coords = blog.CoordGrid2D()(input)
        out = blog.AttentionPool2D()(features, coords)
        self.assertEqual(out.shape, (None, 16, 16, 2), f"Got output shape: {out.shape}")


class CustomMetricsAndLosses(unittest.TestCase):
    def test_MeanCoordError(self):
      # check works without errors with symbolic tensors
      # - can't pass tf.keras.Input() to metric, so validate via dummy model instead
      inp = tf.keras.Input(shape=(149,149))
      out = layers.Reshape((149, 149, 1))(inp)
      model = tf.keras.Model(inp, out)
      model.compile(optimizer="adam", loss="mse", metrics=[blog.MeanCoordError("heatmap-peak")])
      x = tf.random.normal((32,149,149))
      y = tf.random.normal((32,149,149,1))
      model.train_on_batch(x, y)  # Internally calls metric.update_state with symbolic tensors

      # check with actual values
      y_true = tf.stack([blog.generate_heatmap_image(50,50)], axis=0)
      y_pred = tf.stack([blog.generate_heatmap_image(110,10)], axis=0)
      metric = blog.MeanCoordError(encoding='heatmap-peak', system='pixels')
      metric.update_state(y_true, y_pred)
      expected = np.sqrt((50-110)**2 + (50-10)**2)
      self.assertAlmostEqual(metric.result().numpy(), expected, delta=0.01)

      y_true = (np.array([50.,50.]) - 149//2) / 149
      y_pred = (np.array([110.,10.]) - 149//2) / 149
      metric = blog.MeanCoordError('xy', system='pixels')
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(metric.result().numpy(), expected, delta=0.01)

      y_true = (np.array([50.,50.]) - 149//2) / 149
      y_pred = (np.array([110.,10.]) - 149//2) / 149
      metric = blog.MeanCoordError('xy', system='unit-scale')
      metric.update_state(y_true, y_pred)
      self.assertAlmostEqual(metric.result().numpy(), expected/149.0, delta=0.01)
