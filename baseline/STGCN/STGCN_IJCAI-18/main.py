import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = INFO, 2 = WARNING, 3 = ERROR
import tensorflow as tf
import logging

# Further suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from os.path import join as pjoin

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse
from datetime import datetime

# Setup logging
log_dir = "/root/python_on_hyy/logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'stgcn_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('stgcn')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='PEMS03', help='Dataset name')
parser.add_argument('--n_route', type=int, default=358, help='Number of nodes')
parser.add_argument('--n_his', type=int, default=12, help='Historical time steps')
parser.add_argument('--n_pred', type=int, default=12, help='Prediction time steps')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
parser.add_argument('--epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--save', type=int, default=10, help='Save model every N epochs')
parser.add_argument('--ks', type=int, default=3, help='Spatial kernel size')
parser.add_argument('--kt', type=int, default=3, help='Temporal kernel size')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--opt', type=str, default='RMSProp', help='Optimizer')
parser.add_argument('--graph', type=str, default='default', help='Graph weight matrix')
parser.add_argument('--inf_mode', type=str, default='merge', help='Inference mode')
parser.add_argument('--time_interval', type=int, default=5, help='Time interval in minutes')
parser.add_argument('--features', type=int, default=1, help='Number of input features')

args = parser.parse_args()
logger.info(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
C_0 = args.features
day_slot = int(24 * 60 / args.time_interval)
blocks = [[C_0, 32, 64], [64, 32, 128]]

# Load weighted adjacency matrix W
adj_path = f"/root/python_on_hyy/data_for_benchmark/{args.dataset}_adj.npy"
W = np.load(adj_path)

# Calculate graph kernel
L = scaled_laplacian(W)
Lk = cheb_poly_approx(L, Ks, n)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
PeMS = data_gen(args.dataset, n, n_his + n_pred, day_slot, C_0)
# Log mean and std for all features
mean_values = PeMS.mean[0, 0, 0, :]  # Shape: (C_0,)
std_values = PeMS.std[0, 0, 0, :]    # Shape: (C_0,)
mean_str = ", ".join([f"{val:.2f}" for val in mean_values])
std_str = ", ".join([f"{val:.2f}" for val in std_values])
logger.info(f'>> Loading dataset {args.dataset} with Mean: [{mean_str}], STD: [{std_str}]')

if __name__ == '__main__':
    model_train(PeMS, blocks, args)
    model_test(PeMS, PeMS.get_len('test'), n_his, n_pred, args.inf_mode)