import pickle
import numpy as np

# 加载 adj_mx.pkl 文件
file_path = "/root/python_on_hyy/data_for_benchmark/PEMS08/adj_mx.pkl"
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# 假设 data 是一个元组 (sensor_ids, sensor_id_to_ind, adj_mx)
adj_mx = data[2]  # 提取邻接矩阵

# 检查是否存在 NaN 或 Inf
print("Contains NaN:", np.any(np.isnan(adj_mx)))
print("Contains Inf:", np.any(np.isinf(adj_mx)))

# 打印矩阵的统计信息
print("Adjacency matrix shape:", adj_mx.shape)
print("Adjacency matrix mean:", np.mean(adj_mx))
print("Adjacency matrix min:", np.min(adj_mx))
print("Adjacency matrix max:", np.max(adj_mx))