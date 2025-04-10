# @Time     : Jan. 13, 2019 20:16
# @Author   : Veritas YIN
# @FileName : trainer.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion
# @Modified : Mar. 31, 2025 by Grok 3 (xAI)

import os
from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
import logging

logger = logging.getLogger('stgcn')

def model_train(inputs, blocks, args, sum_path='./output/tensorboard'):
    '''
    训练基础模型。
    :param inputs: Dataset类的实例，训练数据源。
    :param blocks: list，ST-Conv块的通道配置。
    :param args: argparse类的实例，训练参数。
    :param sum_path: TensorBoard摘要的路径。
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    C_0 = args.features  # 输入特征数量

    # 模型训练的占位符
    x = tf.placeholder(tf.float32, [None, n_his + 1, n, C_0], name='data_input')  # 使用C_0作为通道维度
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # 定义模型损失
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    tf.summary.scalar('train_loss', train_loss)
    copy_loss_terms = tf.get_collection('copy_loss')
    copy_loss = tf.add_n(copy_loss_terms, name='copy_loss_sum') if copy_loss_terms else tf.constant(0.0)
    tf.summary.scalar('copy_loss', copy_loss)

    # 学习率设置
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    # 学习率每5个epoch衰减0.7
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    tf.summary.scalar('learning_rate', lr)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'错误：优化器 "{opt}" 未定义。')

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        sess.run(tf.global_variables_initializer())

        if inf_mode == 'sep':
            # 对于推断模式 'sep'，步长索引类型为int
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # 对于推断模式 'merge'，步长索引类型为np.ndarray
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            # Initialize min_val and min_va_val with shape (len(step_idx), 3)
            min_val = min_va_val = np.array([[4e1, 1e5, 1e5]] * len(step_idx))
        else:
            raise ValueError(f'错误：测试模式 "{inf_mode}" 未定义。')

        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                if j % 50 == 0:
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    log_msg = f'轮次 {i:2d}, 步长 {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]'
                    print(log_msg)
                    logger.info(log_msg)
            log_msg = f'轮次 {i:2d} 训练时间 {time.time() - start_time:.3f}秒'
            print(log_msg)
            logger.info(log_msg)

            start_time = time.time()
            min_va_val, min_val = \
                model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)

            for ix in tmp_idx:
                va, te = min_va_val[ix//3], min_val[ix//3]  # Adjust index for step_idx
                log_msg = (f'时间步长 {ix + 1}: '
                           f'MAPE {va[0]:7.3%}, {te[0]:7.3%}; '
                           f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                           f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
                print(log_msg)
                logger.info(log_msg)
            log_msg = f'轮次 {i:2d} 推断时间 {time.time() - start_time:.3f}秒'
            print(log_msg)
            logger.info(log_msg)

            if (i + 1) % args.save == 0:
                save_path = "output/models/"
                os.makedirs(save_path, exist_ok=True)
                model_save(sess, global_steps, 'STGCN', save_path)

        # 强制在训练结束时保存模型
        save_path = "output/models/"
        os.makedirs(save_path, exist_ok=True)
        model_save(sess, global_steps, 'STGCN', save_path)
        writer.close()
    print('模型训练完成！')
    logger.info('模型训练完成！')