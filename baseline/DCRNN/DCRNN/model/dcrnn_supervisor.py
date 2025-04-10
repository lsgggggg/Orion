from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
import yaml

from lib import utils, metrics
from lib.AMSGrad import AMSGrad
from lib.metrics import masked_mae_loss, masked_rmse_loss, masked_mape_loss

from model.dcrnn_model import DCRNNModel


class DCRNNSupervisor(object):
    """
    Do experiments using Graph Random Walk RNN model.
    """

    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._writer = tf.summary.FileWriter(self._log_dir)
        self._logger.info(kwargs)

        # Data preparation
        self._data = utils.load_dataset(**self._data_kwargs)
        for k, v in self._data.items():
            if hasattr(v, 'shape'):
                self._logger.info((k, v.shape))

        # 调试：打印数据的均值和范围
        self._logger.info("x_train mean: %s, min: %s, max: %s", np.mean(self._data['x_train']), np.min(self._data['x_train']), np.max(self._data['x_train']))
        self._logger.info("y_train mean: %s, min: %s, max: %s", np.mean(self._data['y_train']), np.min(self._data['y_train']), np.max(self._data['y_train']))
        self._logger.info("x_val mean: %s, min: %s, max: %s", np.mean(self._data['x_val']), np.min(self._data['x_val']), np.max(self._data['x_val']))
        self._logger.info("y_val mean: %s, min: %s, max: %s", np.mean(self._data['y_val']), np.min(self._data['y_val']), np.max(self._data['y_val']))

        # Build models.
        scaler = self._data['scaler']
        with tf.name_scope('Train'):
            with tf.variable_scope('DCRNN', reuse=False):
                self._train_model = DCRNNModel(is_training=True, scaler=scaler,
                                               batch_size=self._data_kwargs['batch_size'],
                                               adj_mx=adj_mx, **self._model_kwargs)

        with tf.name_scope('Test'):
            with tf.variable_scope('DCRNN', reuse=True):
                self._test_model = DCRNNModel(is_training=False, scaler=scaler,
                                              batch_size=self._data_kwargs['test_batch_size'],
                                              adj_mx=adj_mx, **self._model_kwargs)

        # Learning rate.
        self._lr = tf.get_variable('learning_rate', shape=(), initializer=tf.constant_initializer(0.01),
                                   trainable=False)
        self._new_lr = tf.placeholder(tf.float32, shape=(), name='new_learning_rate')
        self._lr_update = tf.assign(self._lr, self._new_lr, name='lr_update')

        # Configure optimizer
        optimizer_name = self._train_kwargs.get('optimizer', 'adam').lower()
        epsilon = float(self._train_kwargs.get('epsilon', 1e-3))
        optimizer = tf.train.AdamOptimizer(self._lr, epsilon=epsilon)
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._lr)
        elif optimizer_name == 'amsgrad':
            optimizer = AMSGrad(self._lr, epsilon=epsilon)

        # Calculate loss and metrics
        output_dim = self._model_kwargs.get('output_dim')
        preds = self._train_model.outputs
        labels = self._train_model.labels[..., :output_dim]

        null_val = 0.
        self._loss_fn = masked_mae_loss(scaler, null_val)
        self._rmse_fn = masked_rmse_loss(scaler, null_val)
        self._mape_fn = masked_mape_loss(scaler, null_val)
        self._train_loss = self._loss_fn(preds=preds, labels=labels)

        tvars = tf.trainable_variables()
        grads = tf.gradients(self._train_loss, tvars)
        max_grad_norm = kwargs['train'].get('max_grad_norm', 1.)  # 降低裁剪阈值
        grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
        global_step = tf.train.get_or_create_global_step()
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step, name='train_op')

        max_to_keep = self._train_kwargs.get('max_to_keep', 100)
        self._epoch = 0
        self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        self._best_model_path = None  # 添加变量以跟踪最佳模型路径

        # Log model statistics.
        total_trainable_parameter = utils.get_total_trainable_parameter_size()
        self._logger.info('Total number of trainable parameters: {:d}'.format(total_trainable_parameter))
        for var in tf.global_variables():
            self._logger.debug('{}, {}'.format(var.name, var.get_shape()))

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def run_epoch_generator(self, sess, model, data_generator, return_output=False, training=False, writer=None):
        losses = []
        maes = []
        rmses = []
        mapes = []
        outputs = []
        output_dim = self._model_kwargs.get('output_dim')
        preds = model.outputs
        labels = model.labels[..., :output_dim]
        loss = self._loss_fn(preds=preds, labels=labels)
        mae = self._loss_fn(preds=preds, labels=labels)  # MAE 作为损失函数
        rmse = self._rmse_fn(preds=preds, labels=labels)
        mape = self._mape_fn(preds=preds, labels=labels)
        fetches = {
            'loss': loss,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'global_step': tf.train.get_or_create_global_step()
        }
        if training:
            fetches.update({
                'train_op': self._train_op
            })
            merged = model.merged
            if merged is not None:
                fetches.update({'merged': merged})

        if return_output:
            fetches.update({
                'outputs': model.outputs
            })

        for step, (x, y) in enumerate(data_generator):
            # 检查输入和标签是否存在 nan 或 inf
            if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                self._logger.info("Step %d: Inputs contain NaN or Inf!", step)
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                self._logger.info("Step %d: Labels contain NaN or Inf!", step)

            feed_dict = {
                model.inputs: x,
                model.labels: y,
            }

            vals = sess.run(fetches, feed_dict=feed_dict)

            # 调试：打印输入、标签、预测值和损失
            if step == 0:  # 仅在第一个 batch 打印
                inputs = feed_dict[model.inputs]
                labels = feed_dict[model.labels]
                preds = vals['outputs'] if return_output else sess.run(model.outputs, feed_dict=feed_dict)
                self._logger.info("Step %d: Inputs mean: %s, min: %s, max: %s", step, np.mean(inputs), np.min(inputs), np.max(inputs))
                self._logger.info("Step %d: Labels mean: %s, min: %s, max: %s", step, np.mean(labels), np.min(labels), np.max(labels))
                self._logger.info("Step %d: Preds mean: %s, min: %s, max: %s", step, np.mean(preds), np.min(preds), np.max(preds))
                self._logger.info("Step %d: Loss: %s", step, vals['loss'])

            losses.append(vals['loss'])
            maes.append(vals['mae'])
            rmses.append(vals['rmse'])
            mapes.append(vals['mape'])
            if writer is not None and 'merged' in vals:
                writer.add_summary(vals['merged'], global_step=vals['global_step'])
            if return_output:
                outputs.append(vals['outputs'])

        results = {
            'loss': np.mean(losses),
            'mae': np.mean(maes),
            'rmse': np.mean(rmses),
            'mape': np.mean(mapes)
        }
        if return_output:
            results['outputs'] = outputs
        return results

    def get_lr(self, sess):
        return sess.run(self._lr)

    def set_lr(self, sess, lr):
        sess.run(self._lr_update, feed_dict={
            self._new_lr: lr
        })

    def train(self, sess, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(sess, **kwargs)

    def _train(self, sess, base_lr, epoch, steps, patience=50, epochs=100,
               min_learning_rate=2e-6, lr_decay_ratio=0.1, save_model=1,
               test_every_n_epochs=10, **train_kwargs):
        history = []
        min_val_loss = float('inf')
        wait = 0

        max_to_keep = train_kwargs.get('max_to_keep', 100)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)
        model_filename = train_kwargs.get('model_filename')
        if model_filename is not None:
            saver.restore(sess, model_filename)
            self._epoch = epoch + 1
        else:
            sess.run(tf.global_variables_initializer())
        self._logger.info('Start training ...')

        while self._epoch <= epochs:
            # Learning rate schedule.
            new_lr = max(min_learning_rate, base_lr * (lr_decay_ratio ** np.sum(self._epoch >= np.array(steps))))
            self.set_lr(sess=sess, lr=new_lr)

            start_time = time.time()
            train_results = self.run_epoch_generator(sess, self._train_model,
                                                     self._data['train_loader'].get_iterator(),
                                                     training=True,
                                                     writer=self._writer)
            train_loss, train_mae, train_rmse, train_mape = train_results['loss'], train_results['mae'], train_results['rmse'], train_results['mape']
            if train_loss > 1e5:
                self._logger.warning('Gradient explosion detected. Ending...')
                break

            global_step = sess.run(tf.train.get_or_create_global_step())
            # Compute validation error.
            val_results = self.run_epoch_generator(sess, self._test_model,
                                                   self._data['val_loader'].get_iterator(),
                                                   training=False)
            val_loss, val_mae, val_rmse, val_mape = val_results['loss'], val_results['mae'], val_results['rmse'], val_results['mape']

            utils.add_simple_summary(self._writer,
                                     ['loss/train_loss', 'metric/train_mae', 'metric/train_rmse', 'metric/train_mape',
                                      'loss/val_loss', 'metric/val_mae', 'metric/val_rmse', 'metric/val_mape'],
                                     [train_loss, train_mae, train_rmse, train_mape, val_loss, val_mae, val_rmse, val_mape],
                                     global_step=global_step)
            end_time = time.time()
            message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, train_rmse: {:.4f}, train_mape: {:.4f}, val_mae: {:.4f}, val_rmse: {:.4f}, val_mape: {:.4f}, lr:{:.6f} {:.1f}s'.format(
                self._epoch, epochs, global_step, train_mae, train_rmse, train_mape, val_mae, val_rmse, val_mape, new_lr, (end_time - start_time))
            self._logger.info(message)
            if self._epoch % test_every_n_epochs == test_every_n_epochs - 1:
                self.evaluate(sess)
            if val_loss <= min_val_loss:
                wait = 0
                if save_model > 0:
                    model_filename = self.save(sess, val_loss)
                    self._best_model_path = model_filename  # 更新最佳模型路径
                self._logger.info(
                    'Val loss decrease from %.4f to %.4f, saving to %s' % (min_val_loss, val_loss, model_filename))
                min_val_loss = val_loss
            else:
                wait += 1
                if wait > patience:
                    self._logger.warning('Early stopping at epoch: %d' % self._epoch)
                    break

            history.append(val_mae)
            # Increases epoch.
            self._epoch += 1

            sys.stdout.flush()
        return np.min(history)

    def evaluate(self, sess, **kwargs):
        global_step = sess.run(tf.train.get_or_create_global_step())
        test_results = self.run_epoch_generator(sess, self._test_model,
                                                self._data['test_loader'].get_iterator(),
                                                return_output=True,
                                                training=False)

        # y_preds:  a list of (batch_size, horizon, num_nodes, output_dim)
        test_loss, y_preds = test_results['loss'], test_results['outputs']
        utils.add_simple_summary(self._writer, ['loss/test_loss'], [test_loss], global_step=global_step)

        y_preds = np.concatenate(y_preds, axis=0)
        scaler = self._data['scaler']
        predictions = []
        y_truths = []
        mae_list = []
        rmse_list = []
        mape_list = []
        for horizon_i in range(self._data['y_test'].shape[1]):
            y_truth = scaler.inverse_transform(self._data['y_test'][:, horizon_i, :, 0])
            y_truths.append(y_truth)

            y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], horizon_i, :, 0])
            predictions.append(y_pred)

            mae = metrics.masked_mae_np(y_pred, y_truth, null_val=0)
            mape = metrics.masked_mape_np(y_pred, y_truth, null_val=0)
            rmse = metrics.masked_rmse_np(y_pred, y_truth, null_val=0)
            self._logger.info(
                "Horizon {:02d}, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                    horizon_i + 1, mae, mape, rmse
                )
            )
            mae_list.append(mae)
            rmse_list.append(rmse)
            mape_list.append(mape)

            utils.add_simple_summary(self._writer,
                                     ['%s_%d' % (item, horizon_i + 1) for item in
                                      ['metric/rmse', 'metric/mape', 'metric/mae']],
                                     [rmse, mape, mae],
                                     global_step=global_step)
        
        # 计算平均指标
        avg_mae = np.mean(mae_list)
        avg_rmse = np.mean(rmse_list)
        avg_mape = np.mean(mape_list)
        self._logger.info(
            "Average over {} horizons, MAE: {:.2f}, MAPE: {:.4f}, RMSE: {:.2f}".format(
                self._data['y_test'].shape[1], avg_mae, avg_mape, avg_rmse
            )
        )

        outputs = {
            'predictions': predictions,
            'groundtruth': y_truths,
            'mae': mae_list,
            'rmse': rmse_list,
            'mape': mape_list
        }
        return outputs

    def load(self, sess, model_filename):
        """
        Restore from saved model.
        :param sess:
        :param model_filename:
        :return:
        """
        self._saver.restore(sess, model_filename)

    def save(self, sess, val_loss):
        config = dict(self._kwargs)
        global_step = sess.run(tf.train.get_or_create_global_step())
        prefix = os.path.join(self._log_dir, 'models-{:.4f}'.format(val_loss))
        config['train']['epoch'] = self._epoch
        config['train']['global_step'] = global_step
        config['train']['log_dir'] = self._log_dir
        config['train']['model_filename'] = self._saver.save(sess, prefix, global_step=global_step,
                                                             write_meta_graph=False)
        config_filename = 'config_{}.yaml'.format(self._epoch)
        with open(os.path.join(self._log_dir, config_filename), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return config['train']['model_filename']

    def get_best_model_path(self):
        """
        Return the path to the best model based on validation loss.
        """
        return self._best_model_path