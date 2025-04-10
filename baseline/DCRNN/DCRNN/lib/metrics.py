import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def masked_mse_tf(preds, labels, null_val=np.nan):
    """
    Mean Squared Error with masking.
    :param preds: TensorFlow tensor of predictions.
    :param labels: TensorFlow tensor of ground truth labels.
    :param null_val: Value to mask (e.g., NaN or 0).
    :return: Masked MSE.
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.square(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_mae_tf(preds, labels, null_val=np.nan):
    """
    Mean Absolute Error with masking.
    :param preds: TensorFlow tensor of predictions.
    :param labels: TensorFlow tensor of ground truth labels.
    :param null_val: Value to mask (e.g., NaN or 0).
    :return: Masked MAE.
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    loss = tf.abs(tf.subtract(preds, labels))
    loss = loss * mask
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    return tf.reduce_mean(loss)


def masked_rmse_tf(preds, labels, null_val=np.nan):
    """
    Root Mean Squared Error with masking.
    :param preds: TensorFlow tensor of predictions.
    :param labels: TensorFlow tensor of ground truth labels.
    :param null_val: Value to mask (e.g., NaN or 0).
    :return: Masked RMSE.
    """
    return tf.sqrt(masked_mse_tf(preds=preds, labels=labels, null_val=null_val))


def masked_mape_tf(preds, labels, null_val=np.nan):
    """
    Mean Absolute Percentage Error with masking.
    :param preds: TensorFlow tensor of predictions.
    :param labels: TensorFlow tensor of ground truth labels.
    :param null_val: Value to mask (e.g., NaN or 0).
    :return: Masked MAPE.
    """
    if np.isnan(null_val):
        mask = ~tf.is_nan(labels)
    else:
        mask = tf.not_equal(labels, null_val)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.where(tf.is_nan(mask), tf.zeros_like(mask), mask)
    # 避免除以 0
    safe_labels = tf.where(tf.equal(labels, 0), tf.ones_like(labels) * 1e-10, labels)
    mape = tf.abs(tf.divide(tf.subtract(preds, labels), safe_labels))
    mape = mape * mask
    mape = tf.where(tf.is_nan(mape), tf.zeros_like(mape), mape)
    return tf.reduce_mean(mape)


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


# Builds loss function with inverse transform.
def masked_mae_loss(scaler, null_val):
    def loss(preds, labels):
        # 反标准化 preds 和 labels
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        # 使用 tf.debugging.assert_all_finite 检查 NaN 和 Inf
        preds = tf.debugging.assert_all_finite(preds, "Preds contain NaN or Inf after inverse transform!")
        labels = tf.debugging.assert_all_finite(labels, "Labels contain NaN or Inf after inverse transform!")
        # 计算反标准化后的 MAE
        return masked_mae_tf(preds=preds, labels=labels, null_val=null_val)
    return loss


def masked_rmse_loss(scaler, null_val):
    def loss(preds, labels):
        # 反标准化 preds 和 labels
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        # 使用 tf.debugging.assert_all_finite 检查 NaN 和 Inf
        preds = tf.debugging.assert_all_finite(preds, "Preds contain NaN or Inf after inverse transform!")
        labels = tf.debugging.assert_all_finite(labels, "Labels contain NaN or Inf after inverse transform!")
        return masked_rmse_tf(preds=preds, labels=labels, null_val=null_val)
    return loss


def masked_mape_loss(scaler, null_val):
    def loss(preds, labels):
        # 反标准化 preds 和 labels
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        # 使用 tf.debugging.assert_all_finite 检查 NaN 和 Inf
        preds = tf.debugging.assert_all_finite(preds, "Preds contain NaN or Inf after inverse transform!")
        labels = tf.debugging.assert_all_finite(labels, "Labels contain NaN or Inf after inverse transform!")
        return masked_mape_tf(preds=preds, labels=labels, null_val=null_val)
    return loss


def calculate_metrics(df_pred, df_test, null_val):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred: Predicted values (numpy array or DataFrame).
    :param df_test: Ground truth values (numpy array or DataFrame).
    :param null_val: Value to mask (e.g., NaN or 0).
    :return: MAE, MAPE, RMSE.
    """
    # 将 DataFrame 转换为 numpy 数组
    preds = df_pred.to_numpy() if hasattr(df_pred, 'to_numpy') else df_pred
    labels = df_test.to_numpy() if hasattr(df_test, 'to_numpy') else df_test
    mape = masked_mape_np(preds=preds, labels=labels, null_val=null_val)
    mae = masked_mae_np(preds=preds, labels=labels, null_val=null_val)
    rmse = masked_rmse_np(preds=preds, labels=labels, null_val=null_val)
    return mae, mape, rmse