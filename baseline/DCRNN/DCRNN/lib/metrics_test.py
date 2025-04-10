import unittest
import sys
import os

# 动态添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from lib import metrics


class MyTestCase(unittest.TestCase):
    def test_masked_mape_np(self):
        preds = np.array([
            [1, 2, 2],
            [3, 4, 5],
        ], dtype=np.float32)
        labels = np.array([
            [1, 2, 2],
            [3, 4, 4]
        ], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 24.0, mape, delta=1e-5)

    def test_masked_mape_np2(self):
        preds = np.array([
            [1, 2, 2],
            [3, 4, 5],
        ], dtype=np.float32)
        labels = np.array([
            [1, 2, 2],
            [3, 4, 4]
        ], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels, null_val=4)
        self.assertEqual(0., mape)

    def test_masked_mape_np_all_zero(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels, null_val=0)
        self.assertEqual(0., mape)

    def test_masked_mape_np_all_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels)
        self.assertEqual(0., mape)

    def test_masked_mape_np_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [np.nan, np.nan],
            [np.nan, 3]
        ], dtype=np.float32)
        mape = metrics.masked_mape_np(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 3., mape, delta=1e-5)

    def test_masked_mae_np_vanilla(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 4],
            [3, 4]
        ], dtype=np.float32)
        mae = metrics.masked_mae_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0.5, mae, delta=1e-5)

    def test_masked_mae_np_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, np.nan],
            [3, 4]
        ], dtype=np.float32)
        mae = metrics.masked_mae_np(preds=preds, labels=labels)
        self.assertAlmostEqual(0., mae, delta=1e-5)

    def test_masked_rmse_np_vanilla(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 4],
            [3, 4]
        ], dtype=np.float32)
        rmse = metrics.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(1., rmse, delta=1e-5)

    def test_masked_rmse_np_nan(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, np.nan],
            [3, 4]
        ], dtype=np.float32)
        rmse = metrics.masked_rmse_np(preds=preds, labels=labels)
        self.assertAlmostEqual(0., rmse, delta=1e-5)

    def test_masked_rmse_np_all_zero(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32)
        rmse = metrics.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., rmse, delta=1e-5)

    def test_masked_rmse_np_missing(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 0],
            [3, 4]
        ], dtype=np.float32)
        rmse = metrics.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., rmse, delta=1e-5)

    def test_masked_rmse_np2(self):
        preds = np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32)
        labels = np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32)
        rmse = metrics.masked_rmse_np(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(np.sqrt(1 / 3.), rmse, delta=1e-5)


class TFRMSETestCase(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.sess.__enter__()

    def tearDown(self):
        self.sess.__exit__(None, None, None)

    def test_masked_mse_tf_null(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        mse = metrics.masked_mse_tf(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(1 / 3.0, self.sess.run(mse), delta=1e-5)

    def test_masked_mse_tf_vanilla(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        mse = metrics.masked_mse_tf(preds=preds, labels=labels)
        self.assertAlmostEqual(1.25, self.sess.run(mse), delta=1e-5)

    def test_masked_mse_tf_all_zero(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.float32))
        mse = metrics.masked_mse_tf(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., self.sess.run(mse), delta=1e-5)

    def test_masked_mse_tf_nan(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 2],
            [3, np.nan]
        ], dtype=np.float32))
        mse = metrics.masked_mse_tf(preds=preds, labels=labels)
        self.assertAlmostEqual(0., self.sess.run(mse), delta=1e-5)

    def test_masked_mse_tf_all_nan(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [np.nan, np.nan],
            [np.nan, np.nan]
        ], dtype=np.float32))
        mse = metrics.masked_mse_tf(preds=preds, labels=labels, null_val=0)
        self.assertAlmostEqual(0., self.sess.run(mse), delta=1e-5)

    def test_masked_mae_tf_vanilla(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        mae = metrics.masked_mae_tf(preds=preds, labels=labels)
        self.assertAlmostEqual(0.75, self.sess.run(mae), delta=1e-5)

    def test_masked_rmse_tf_vanilla(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 0],
            [3, 3]
        ], dtype=np.float32))
        rmse = metrics.masked_rmse_tf(preds=preds, labels=labels)
        self.assertAlmostEqual(np.sqrt(1.25), self.sess.run(rmse), delta=1e-5)

    def test_masked_mape_tf_vanilla(self):
        preds = tf.constant(np.array([
            [1, 2],
            [3, 4],
        ], dtype=np.float32))
        labels = tf.constant(np.array([
            [1, 2],
            [3, 3]
        ], dtype=np.float32))
        mape = metrics.masked_mape_tf(preds=preds, labels=labels)
        self.assertAlmostEqual(1 / 12.0, self.sess.run(mape), delta=1e-5)


if __name__ == '__main__':
    unittest.main()