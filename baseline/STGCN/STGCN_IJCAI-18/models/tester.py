# @Time     : Jan. 10, 2019 17:52
# @Author   : Veritas YIN
# @FileName : tester.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion
# @Modified : Mar. 31, 2025 by Grok 3 (xAI)

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
import logging

logger = logging.getLogger('stgcn')

def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            pred = pred[:, -1, :, :]  # Shape: (batch_size, n_route, 1)
            batch_size, n_route, _ = pred.shape
            pred_expanded = np.zeros((batch_size, n_route, test_seq.shape[-1]))  # Shape: (batch_size, n_route, C_0)
            pred_expanded[:, :, 0] = pred[:, :, 0]  # Only update the flow channel (index 0)
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred_expanded
            step_list.append(pred)
        pred_list.append(step_list)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]

def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, 0:1], y_val, x_stats)  # Only evaluate the flow channel

    chks = evl_val < min_va_val
    if np.any(chks):
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, 0:1], y_pred, x_stats)  # Only evaluate the flow channel
        min_val = evl_pred
    return min_va_val, min_val

def model_test(inputs, batch_size, n_his, n_pred, inf_mode, load_path='./output/models/'):
    start_time = time.time()
    checkpoint_state = tf.train.get_checkpoint_state(load_path)
    if checkpoint_state is None:
        raise FileNotFoundError(f"No checkpoint found in {load_path}. Please ensure the model has been saved during training.")
    model_path = checkpoint_state.model_checkpoint_path

    test_graph = tf.Graph()
    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        logger.info(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        if inf_mode == 'sep':
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            step_idx = tmp_idx = np.arange(n_pred)  # Predict all steps
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, 0:1], y_test, x_stats)  # Only evaluate the flow channel

        # Ensure evl is a 2D array with shape (n_pred, 3)
        assert len(evl.shape) == 2 and evl.shape[1] == 3, f"Expected evl to be a 2D array with shape (n_pred, 3), got {evl.shape}"

        # Log results for each horizon
        for i, ix in enumerate(tmp_idx):
            te = evl[i]  # te should be [MAPE, MAE, RMSE]
            log_msg = f'Horizon {ix + 1:02d}, MAE: {te[1]:.2f}, MAPE: {te[0]*100:.2f}%, RMSE: {te[2]:.2f}'
            logger.info(log_msg)

        # Average over all horizons
        avg_mape = np.mean(evl[:, 0])
        avg_mae = np.mean(evl[:, 1])
        avg_rmse = np.mean(evl[:, 2])
        log_msg = f'Average over {n_pred} horizons, MAE: {avg_mae:.2f}, MAPE: {avg_mape*100:.2f}%, RMSE: {avg_rmse:.2f}'
        logger.info(log_msg)

        log_msg = f'Model Test Time {time.time() - start_time:.3f}s'
        logger.info(log_msg)
    logger.info('Testing model finished!')