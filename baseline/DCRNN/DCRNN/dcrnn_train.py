import argparse
import numpy as np
import os
import sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import yaml
import logging

from lib import utils
from lib.metrics import masked_mae_np
from model.dcrnn_supervisor import DCRNNSupervisor


def main(args):
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s,%(msecs)d - INFO - %(message)s')

    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.SafeLoader)

        tf_config = tf.ConfigProto()
        if args.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = utils.load_graph_data(graph_pkl_filename)

        with tf.Session(config=tf_config) as sess:
            # 创建 DCRNNSupervisor 实例
            supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)

            # 运行训练
            supervisor.train(sess=sess)

            # 获取最佳模型路径
            best_model_path = supervisor.get_best_model_path()
            if best_model_path is None:
                logging.info("No best model found. Please ensure training completed successfully.")
                return

            # 加载最佳模型
            logging.info(f"Loaded model from {best_model_path}")
            supervisor.load(sess, best_model_path)

            # 运行测试集评估
            logging.info("Starting evaluation on test set...")
            outputs = supervisor.evaluate(sess)

            # 完成评估
            logging.info("Evaluation completed.")

            # 保存预测结果
            output_filename = os.path.join(supervisor_config['base_dir'], 'dcrnn_predictions.npz')
            np.savez_compressed(output_filename, **outputs)
            logging.info(f"Predictions saved as {output_filename}")


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/dcrnn_la.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)