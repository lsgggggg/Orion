import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch.distributed as dist  # 引入分布式训练支持
from torch.nn.parallel import DistributedDataParallel as DDP

class DynamicGAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_timesteps, adj_matrix, num_heads=4, dropout=0.1, k_neighbors=3):
        super(DynamicGAT, self).__init__()
        self.num_nodes = num_nodes
        self.num_timesteps = num_timesteps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.k_neighbors = k_neighbors

        # 线性变换层
        self.W = nn.Linear(in_channels, out_channels * num_heads)
        self.attn = nn.ModuleList([nn.Linear(out_channels * 2, 1, bias=False) for _ in range(num_heads)])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # 处理邻接矩阵，筛选 K=3 邻居并计算初始注意力权重
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.FloatTensor(adj_matrix)
        # 确保 adj_matrix 在正确的设备上（由模型参数决定）
        self.register_buffer('adj_matrix', adj_matrix)
        self.adj_mask, initial_weights = self._compute_adj_mask_and_weights()

        # 将 initial_weights 转换为可训练参数
        self.initial_weights = nn.Parameter(initial_weights, requires_grad=True)
        # 保存初始值，用于约束浮动范围
        self.register_buffer('initial_weights_base', initial_weights.clone().detach())
        # 浮动范围约束（±50%）
        self.weight_bound_factor = 0.5  # 允许浮动范围为初始值的 ±50%

        # 调试：确认初始化时的设备
        print(f"Initialized DynamicGAT with adj_matrix device: {self.adj_matrix.device}")
        print(f"self.initial_weights device: {self.initial_weights.device}")
        print(f"self.initial_weights_base device: {self.initial_weights_base.device}")

    def _compute_adj_mask_and_weights(self):
        """
        基于邻接矩阵筛选每个节点的 K=3 最近邻居，并计算初始注意力权重，全部在 GPU 上完成。
        
        Returns:
            adj_mask: 邻接掩码，形状为 (N, N)，值为 0 或 1。
            initial_weights: 初始注意力权重，形状为 (N, N)，基于距离倒数。
        """
        adj_matrix = self.adj_matrix  # 直接使用 GPU 上的邻接矩阵
        N = self.num_nodes

        # 初始化邻接掩码和初始权重
        adj_mask = torch.zeros((N, N), dtype=torch.float32, device=adj_matrix.device)
        initial_weights = torch.zeros((N, N), dtype=torch.float32, device=adj_matrix.device)

        # 对于每个节点，筛选 K=3 最近邻居
        for i in range(N):
            # 获取节点 i 到所有其他节点的距离
            distances = adj_matrix[i, :]
            # 仅考虑有连接的节点（距离 > 0）
            valid_indices = torch.where(distances > 0)[0]
            if valid_indices.numel() == 0:
                continue  # 如果没有邻居，跳过

            valid_distances = distances[valid_indices]
            # 按距离排序，选出最近的 K 个邻居
            _, sorted_indices = torch.sort(valid_distances)
            sorted_indices = sorted_indices[:min(self.k_neighbors, len(valid_indices))]
            neighbor_indices = valid_indices[sorted_indices]

            # 设置邻接掩码
            adj_mask[i, neighbor_indices] = 1.0

            # 计算初始权重（距离倒数）
            neighbor_distances = valid_distances[sorted_indices]
            # 避免除以 0，添加小值
            neighbor_weights = 1.0 / (neighbor_distances + 1e-6)
            # 归一化权重
            neighbor_weights = neighbor_weights / (torch.sum(neighbor_weights) + 1e-6)
            initial_weights[i, neighbor_indices] = neighbor_weights

        return adj_mask, initial_weights

    def constrain_weights(self):
        """
        约束 initial_weights 的浮动范围，使其不超过初始值的 ±50%。
        """
        with torch.no_grad():
            lower_bound = self.initial_weights_base * (1 - self.weight_bound_factor)
            upper_bound = self.initial_weights_base * (1 + self.weight_bound_factor)
            self.initial_weights.clamp_(min=lower_bound, max=upper_bound)
            # 确保非负
            self.initial_weights.clamp_(min=0)
            # 重新归一化（仅对 K=3 邻居）
            weights_sum = torch.sum(self.initial_weights * self.adj_mask, dim=1, keepdim=True)
            weights_sum = weights_sum + (weights_sum == 0).float() * 1e-6  # 避免除以 0
            self.initial_weights.data = (self.initial_weights * self.adj_mask) / weights_sum

    def forward(self, x):
        B, N, F_in, T = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, T, N, F_in]
        Wh = self.W(x).view(B, T, N, self.num_heads, self.out_channels)  # [B, T, N, H, C]

        # 约束 initial_weights 的浮动范围
        self.constrain_weights()

        # 扩展 adj_mask 和 initial_weights 以匹配批量和时间维度
        adj_mask = self.adj_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, N, 1]
        initial_weights = self.initial_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, N, 1]

        # 逐头计算注意力
        attention_scores = []
        for h in range(self.num_heads):
            Wh_h = Wh[:, :, :, h, :]  # [B, T, N, C]
            Wh1 = Wh_h.unsqueeze(3)  # [B, T, N, 1, C]
            Wh2 = Wh_h.unsqueeze(2)  # [B, T, 1, N, C]
            Wh1_exp = Wh1.expand(-1, -1, -1, N, -1)  # [B, T, N, N, C]
            Wh2_exp = Wh2.expand(-1, -1, N, -1, -1)  # [B, T, N, N, C]
            e_h = torch.cat([Wh1_exp, Wh2_exp], dim=-1)  # [B, T, N, N, 2C]
            score = self.attn[h](e_h).squeeze(-1)  # [B, T, N, N]
            score = self.leaky_relu(score)

            # 结合初始权重和邻接掩码
            score = score * adj_mask.squeeze(-1)  # [B, T, N, N]
            score = score + initial_weights.squeeze(-1)  # [B, T, N, N]
            attention_scores.append(score)

        attention = torch.stack(attention_scores, dim=-1)  # [B, T, N, N, H]
        attention = F.softmax(attention, dim=3)
        attention = self.dropout(attention)

        out = torch.einsum('btijh,btjhc->btihc', attention, Wh)  # [B, T, N, H, C]
        out = out.reshape(B, T, N, self.num_heads * self.out_channels)
        return out.permute(0, 2, 3, 1)  # [B, N, H*C, T]

    def get_attention_weights(self, x):
        B, N, F_in, T = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, T, N, F_in]
        with torch.no_grad():
            Wh = self.W(x).view(B, T, N, self.num_heads, self.out_channels)  # [B, T, N, H, C]
            adj_mask = self.adj_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, N, 1]
            initial_weights = self.initial_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, N, N, 1]
            attention_scores = []
            for h in range(self.num_heads):
                Wh_h = Wh[:, :, :, h, :]  # [B, T, N, C]
                Wh1 = Wh_h.unsqueeze(3)  # [B, T, N, 1, C]
                Wh2 = Wh_h.unsqueeze(2)  # [B, T, 1, N, C]
                Wh1_exp = Wh1.expand(-1, -1, -1, N, -1)  # [B, T, N, N, C]
                Wh2_exp = Wh2.expand(-1, -1, N, -1, -1)  # [B, T, N, N, C]
                e_h = torch.cat([Wh1_exp, Wh2_exp], dim=-1)  # [B, T, N, N, 2C]
                score = self.attn[h](e_h).squeeze(-1)  # [B, T, N, N]
                score = self.leaky_relu(score)
                score = score * adj_mask.squeeze(-1)  # [B, T, N, N]
                score = score + initial_weights.squeeze(-1)  # [B, T, N, N]
                attention_scores.append(score)
            attention = torch.stack(attention_scores, dim=-1)  # [B, T, N, N, H]
            attention = F.softmax(attention, dim=3)  # [B, T, N, N, H]
            return attention  # 返回张量，保持在 GPU 上，由调用者决定是否移到 CPU

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"

        self.d_k = out_channels // num_heads
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_k = nn.Linear(in_channels, out_channels)
        self.W_v = nn.Linear(in_channels, out_channels)
        self.W_o = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, N, F_in, T = x.shape
        assert F_in == self.in_channels, f"Input feature dimension {F_in} must match in_channels {self.in_channels}"
        x_reshaped = x.permute(0, 3, 1, 2)  # [B, T, N, F_in]

        Q = self.W_q(x_reshaped).view(batch_size, T, N, self.num_heads, self.d_k).permute(0, 2, 3, 1, 4)
        K = self.W_k(x_reshaped).view(batch_size, T, N, self.num_heads, self.d_k).permute(0, 2, 3, 1, 4)
        V = self.W_v(x_reshaped).view(batch_size, T, N, self.num_heads, self.d_k).permute(0, 2, 3, 1, 4)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V).permute(0, 3, 1, 2, 4).contiguous().view(batch_size, T, N, self.out_channels)
        output = self.W_o(context).permute(0, 2, 3, 1)
        return output

    def get_attention_weights(self, x):
        batch_size, N, F_in, T = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, T, N, F_in]
        with torch.no_grad():
            Q = self.W_q(x).view(batch_size, T, N, self.num_heads, self.d_k).permute(0, 2, 3, 1, 4)
            K = self.W_k(x).view(batch_size, T, N, self.num_heads, self.d_k).permute(0, 2, 3, 1, 4)
            scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)
            attn = F.softmax(scores, dim=-1)  # [B, N, H, T, T]
            return attn  # 返回张量，保持在 GPU 上，由调用者决定是否移到 CPU

class Belt_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_timesteps,
                 num_heads_spatial, num_heads_temporal, spatial_dropout,
                 temporal_dropout, ff_dropout, ff_hidden_dim, adj_matrix):
        super(Belt_Block, self).__init__()
        self.dynamic_gat = DynamicGAT(in_channels, out_channels, num_nodes, num_timesteps,
                                      adj_matrix=adj_matrix, num_heads=num_heads_spatial, dropout=spatial_dropout)

        self.gat_out_channels = out_channels * num_heads_spatial
        self.norm1 = nn.LayerNorm([num_nodes, self.gat_out_channels, num_timesteps])
        self.dropout1 = nn.Dropout(spatial_dropout)

        self.temporal_attention = MultiHeadSelfAttention(self.gat_out_channels, self.gat_out_channels,
                                                         num_heads=num_heads_temporal, dropout=temporal_dropout)
        self.norm2 = nn.LayerNorm([num_nodes, self.gat_out_channels, num_timesteps])
        self.dropout2 = nn.Dropout(temporal_dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(self.gat_out_channels, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, self.gat_out_channels)
        )
        self.norm3 = nn.LayerNorm([num_nodes, self.gat_out_channels, num_timesteps])
        self.dropout3 = nn.Dropout(ff_dropout)

    def forward(self, x):
        gat_out = self.dynamic_gat(x)

        if x.shape[-2] == gat_out.shape[-2]:
            gat_out = self.norm1(gat_out + x)
        else:
            gat_out = self.norm1(gat_out)
        gat_out = self.dropout1(gat_out)

        attn_out = self.temporal_attention(gat_out)
        attn_out = self.norm2(attn_out + gat_out)
        attn_out = self.dropout2(attn_out)

        B, N, F, T = attn_out.shape
        ff_out = self.feedforward(attn_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        ff_out = self.norm3(ff_out + attn_out)
        ff_out = self.dropout3(ff_out)

        return ff_out

    def interpret(self, x, save_path, epoch, config, visualize_nodes=-1, local_rank=0):
        # 在多 GPU 训练中，只有 rank 0 进行可视化
        if local_rank != 0:
            return

        # 提取注意力权重
        gat_out = self.dynamic_gat(x)
        gat_weights = self.dynamic_gat.get_attention_weights(x)  # [B, T, N, N, H]
        attn_weights = self.temporal_attention.get_attention_weights(gat_out)  # [B, N, H, T, T]

        # 从配置文件中读取可视化参数
        visualize_annot = config['Orion'].getboolean('visualize_annot', True)
        max_nodes_to_visualize_gat = config['Orion'].getint('max_nodes_to_visualize_gat', 10)
        max_nodes_to_visualize = config['Orion'].getint('max_nodes_to_visualize', 10)

        # 确定要可视化的节点（仅用于 Temporal Attention）
        num_nodes = x.shape[1]  # N
        if visualize_nodes == -1:
            nodes_to_visualize = range(min(num_nodes, max_nodes_to_visualize))  # 限制可视化节点数量
        elif 0 <= visualize_nodes < num_nodes:
            nodes_to_visualize = [visualize_nodes]
        else:
            print(f"Invalid visualize_nodes: {visualize_nodes}. Must be between 0 and {num_nodes-1} or -1 for all nodes.")
            return

        os.makedirs(save_path, exist_ok=True)

        # 可视化 GAT 注意力权重
        B, T, N, N, H_spatial = gat_weights.shape
        max_nodes_to_visualize_gat = min(max_nodes_to_visualize_gat, num_nodes)
        for t in range(T):
            for head in range(H_spatial):
                weights_subset = gat_weights[0, t, :max_nodes_to_visualize_gat, :max_nodes_to_visualize_gat, head]
                weights_subset = weights_subset.cpu().numpy()  # 仅在可视化时移到 CPU
                plt.figure(figsize=(10, 8))
                vmin = weights_subset.min()
                vmax = weights_subset.max()
                sns.heatmap(weights_subset, cmap="YlGnBu", annot=visualize_annot, fmt=".3f", cbar=True,
                            vmin=vmin, vmax=vmax)
                plt.title(f"GAT Attention at Timestep {t}, Head {head}, Epoch {epoch}", fontsize=16, pad=20)
                plt.xlabel("Node", fontsize=14)
                plt.ylabel("Node", fontsize=14)
                plt.savefig(os.path.join(save_path, f"gat_attention_t{t}_head{head}_epoch{epoch}_rank{local_rank}.png"), 
                            dpi=150, bbox_inches='tight')
                plt.close()

        # 可视化时间注意力权重
        B, N, H_temporal, T, T = attn_weights.shape
        for node in nodes_to_visualize:
            for head in range(H_temporal):
                weights = attn_weights[0, node, head, :, :]  # [T, T]
                weights = weights.cpu().numpy()  # 仅在可视化时移到 CPU
                plt.figure(figsize=(10, 8))
                sns.heatmap(weights, cmap="YlGnBu", annot=visualize_annot, fmt=".2f", cbar=True)
                plt.title(f"Temporal Attention, Node {node}, Head {head}, Epoch {epoch}", fontsize=16, pad=20)
                plt.xlabel("Time Step", fontsize=14)
                plt.ylabel("Time Step", fontsize=14)
                plt.savefig(os.path.join(save_path, f"temporal_attention_node{node}_head{head}_epoch{epoch}_rank{local_rank}.png"), 
                            dpi=150, bbox_inches='tight')
                plt.close()

class TimeReductionFFC(nn.Module):
    def __init__(self, in_timesteps, out_timesteps, in_features):
        super(TimeReductionFFC, self).__init__()
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.in_features = in_features
        
        self.reduction = nn.Linear(in_timesteps, out_timesteps)
        
    def forward(self, x):
        B, N, F, T = x.shape
        x = x.view(B * N * F, T)
        x = self.reduction(x)
        x = x.view(B, N, F, self.out_timesteps)
        return x

class Orion(nn.Module):
    def __init__(self, DEVICE, in_channels, out_channels, num_nodes, num_timesteps_h, num_timesteps_d, num_timesteps_w,
                 num_for_predict, nb_block, num_heads_spatial, num_heads_temporal, num_heads_fusion,
                 spatial_dropout, temporal_dropout, fusion_dropout, ff_dropout, ff_hidden_dim, adj_matrix):
        super(Orion, self).__init__()
        self.DEVICE = DEVICE
        self.num_nodes = num_nodes
        self.num_for_predict = num_for_predict

        self.recent_blocks = nn.ModuleList([
            Belt_Block(in_channels if i == 0 else out_channels * num_heads_spatial, out_channels, num_nodes,
                        num_timesteps_h, num_heads_spatial, num_heads_temporal, spatial_dropout,
                        temporal_dropout, ff_dropout, ff_hidden_dim, adj_matrix)
            for i in range(nb_block)
        ])
        self.daily_blocks = nn.ModuleList([
            Belt_Block(in_channels if i == 0 else out_channels * num_heads_spatial, out_channels, num_nodes,
                        num_timesteps_d, num_heads_spatial, num_heads_temporal, spatial_dropout,
                        temporal_dropout, ff_dropout, ff_hidden_dim, adj_matrix)
            for i in range(nb_block)
        ])
        self.weekly_blocks = nn.ModuleList([
            Belt_Block(in_channels if i == 0 else out_channels * num_heads_spatial, out_channels, num_nodes,
                        num_timesteps_w, num_heads_spatial, num_heads_temporal, spatial_dropout,
                        temporal_dropout, ff_dropout, ff_hidden_dim, adj_matrix)
            for i in range(nb_block)
        ])

        self.time_reduction = TimeReductionFFC(
            in_timesteps=num_timesteps_h,  # 12
            out_timesteps=1,               # 降维到 1
            in_features=out_channels * num_heads_spatial
        )

        self.fusion_attention = MultiHeadSelfAttention(
            in_channels=out_channels * num_heads_spatial,
            out_channels=out_channels,
            num_heads=num_heads_fusion,
            dropout=fusion_dropout
        )

        self.fusion_projection = nn.Linear(out_channels * num_heads_spatial, out_channels)
        self.fusion_norm = nn.LayerNorm([num_nodes, out_channels, num_for_predict])
        self.fusion_dropout = nn.Dropout(fusion_dropout)

        self.fc = nn.Linear(out_channels, out_channels)
        self.fc_norm = nn.LayerNorm([num_nodes, out_channels, num_for_predict])
        self.fc_dropout = nn.Dropout(ff_dropout)

        self.output_layer = nn.Linear(out_channels, 1)

        self.to(DEVICE)

    def forward(self, x_h, x_d, x_w):
        recent_out = x_h
        for block in self.recent_blocks:
            recent_out = block(recent_out)

        daily_out = x_d
        for block in self.daily_blocks:
            daily_out = block(daily_out)

        weekly_out = x_w
        for block in self.weekly_blocks:
            weekly_out = block(weekly_out)

        recent_out = self.time_reduction(recent_out)
        daily_out = self.time_reduction(daily_out)
        weekly_out = self.time_reduction(weekly_out)

        fused_input = torch.cat([recent_out, daily_out, weekly_out], dim=3)
        fused_out = self.fusion_attention(fused_input)

        fused_input_projected = self.fusion_projection(fused_input.permute(0, 1, 3, 2))
        fused_input_projected = fused_input_projected.permute(0, 1, 3, 2)
        fused_input_projected = fused_input_projected.repeat(1, 1, 1, self.num_for_predict // 3)

        fused_out = fused_out.repeat(1, 1, 1, self.num_for_predict // 3)
        fused_out = self.fusion_norm(fused_out + fused_input_projected)
        fused_out = self.fusion_dropout(fused_out)

        fc_out = self.fc(fused_out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        fc_out = self.fc_norm(fc_out + fused_out)
        fc_out = self.fc_dropout(fc_out)

        output = self.output_layer(fc_out.permute(0, 1, 3, 2))
        output = output.squeeze(-1)
        return output

    def interpret(self, x_h, x_d, x_w, save_path, epoch, config, visualize_nodes=-1, local_rank=0):
        # 在多 GPU 训练中，只有 rank 0 进行可视化
        if local_rank != 0:
            return

        self.eval()  # 确保模型处于评估模式
        with torch.no_grad():  # 禁用梯度计算，节省内存
            # 从配置文件中读取可视化参数
            visualize_annot = config['Orion'].getboolean('visualize_annot', True)
            max_nodes_to_visualize = config['Orion'].getint('max_nodes_to_visualize', 10)
            max_nodes_to_visualize_gat = config['Orion'].getint('max_nodes_to_visualize_gat', 10)

            # 确定要可视化的节点
            num_nodes = self.num_nodes
            if visualize_nodes == -1:
                nodes_to_visualize = range(min(num_nodes, max_nodes_to_visualize))
            elif 0 <= visualize_nodes < num_nodes:
                nodes_to_visualize = [visualize_nodes]
            else:
                print(f"Invalid visualize_nodes: {visualize_nodes}. Must be between 0 and {num_nodes-1} or -1 for all nodes.")
                return

            # 移动输入数据到正确设备
            x_h, x_d, x_w = x_h.to(self.DEVICE), x_d.to(self.DEVICE), x_w.to(self.DEVICE)

            # 调试：打印输入形状
            print(f"interpret: x_h shape: {x_h.shape}")
            print(f"interpret: x_d shape: {x_d.shape}")
            print(f"interpret: x_w shape: {x_w.shape}")

            # Recent blocks
            recent_out = x_h
            for i, block in enumerate(self.recent_blocks):
                block.interpret(
                    recent_out,
                    os.path.join(save_path, f"recent_block_{i}"),
                    epoch,
                    config,
                    visualize_nodes=visualize_nodes,
                    local_rank=local_rank
                )
                recent_out = block(recent_out)

            # Daily blocks
            daily_out = x_d
            for i, block in enumerate(self.daily_blocks):
                block.interpret(
                    daily_out,
                    os.path.join(save_path, f"daily_block_{i}"),
                    epoch,
                    config,
                    visualize_nodes=visualize_nodes,
                    local_rank=local_rank
                )
                daily_out = block(daily_out)

            # Weekly blocks
            weekly_out = x_w
            for i, block in enumerate(self.weekly_blocks):
                block.interpret(
                    weekly_out,
                    os.path.join(save_path, f"weekly_block_{i}"),
                    epoch,
                    config,
                    visualize_nodes=visualize_nodes,
                    local_rank=local_rank
                )
                weekly_out = block(weekly_out)

            # 时间降维
            recent_out = self.time_reduction(recent_out)
            daily_out = self.time_reduction(daily_out)
            weekly_out = self.time_reduction(weekly_out)

            # 调试：打印降维后的形状
            print(f"interpret: recent_out shape after reduction: {recent_out.shape}")
            print(f"interpret: daily_out shape after reduction: {daily_out.shape}")
            print(f"interpret: weekly_out shape after reduction: {weekly_out.shape}")

            # 融合注意力
            fused_input = torch.cat([recent_out, daily_out, weekly_out], dim=3)  # [B, N, F, 3]
            fusion_attn_weights = self.fusion_attention.get_attention_weights(fused_input)  # [B, N, H, T, T], T=3

            # 调试：打印融合注意力权重的形状
            print(f"interpret: fusion_attn_weights shape: {fusion_attn_weights.shape}")

            # 可视化融合注意力
            os.makedirs(save_path, exist_ok=True)
            B, N, H_fusion, T, T = fusion_attn_weights.shape  # T=3
            for node in nodes_to_visualize:
                for head in range(H_fusion):
                    weights = fusion_attn_weights[0, node, head, :, :]  # [T, T], T=3
                    weights = weights.cpu().numpy()  # 仅在可视化时移到 CPU
                    plt.figure(figsize=(6, 6))  # 融合注意力矩阵较小，调整图形大小
                    sns.heatmap(weights, cmap="YlGnBu", annot=visualize_annot, fmt=".2f", cbar=True,
                                xticklabels=['Recent', 'Daily', 'Weekly'],
                                yticklabels=['Recent', 'Daily', 'Weekly'])
                    plt.title(f"Fusion Attention, Node {node}, Head {head}, Epoch {epoch}", fontsize=16, pad=20)
                    plt.xlabel("Period", fontsize=14)
                    plt.ylabel("Period", fontsize=14)
                    plt.savefig(os.path.join(save_path, f"fusion_attention_node{node}_head{head}_epoch{epoch}_rank{local_rank}.png"), 
                                dpi=150, bbox_inches='tight')
                    plt.close()

def make_model(DEVICE, in_channels, out_channels, num_nodes, num_timesteps_h, num_timesteps_d, num_timesteps_w,
               num_for_predict, nb_block, num_heads_spatial, num_heads_temporal, num_heads_fusion,
               spatial_dropout, temporal_dropout, fusion_dropout, ff_dropout, ff_hidden_dim, adj_matrix,
               use_distributed=False, local_rank=0):
    # 确保 adj_matrix 在正确的设备上
    adj_matrix = adj_matrix.to(DEVICE)

    # 创建 Orion 模型
    model = Orion(DEVICE, in_channels, out_channels, num_nodes, num_timesteps_h, num_timesteps_d, num_timesteps_w,
                    num_for_predict, nb_block, num_heads_spatial, num_heads_temporal, num_heads_fusion,
                    spatial_dropout, temporal_dropout, fusion_dropout, ff_dropout, ff_hidden_dim, adj_matrix)
    
    # 初始化参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 将模型移动到正确的设备
    model = model.to(DEVICE)

    # 调试：打印模型参数的设备
    if local_rank == 0:
        for name, param in model.named_parameters():
            print(f"Parameter {name} is on device: {param.device}")
        print(f"adj_matrix is on device: {adj_matrix.device}")
        print(f"Model device: {next(model.parameters()).device}")
        # 调试：检查模型是否有 interpret 方法
        print(f"Model has interpret method before DDP: {hasattr(model, 'interpret')}")

    # 使用 DistributedDataParallel（DDP）包装模型
    if use_distributed and torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False
        )
        if local_rank == 0:
            print(f"Using {torch.distributed.get_world_size()} GPUs with DistributedDataParallel!")
            # 调试：检查 DDP 包装后 model.module 是否有 interpret 方法
            print(f"Model.module has interpret method after DDP: {hasattr(model.module, 'interpret')}")
            if not hasattr(model.module, 'interpret'):
                print("Available methods in model.module:", [attr for attr in dir(model.module) if not attr.startswith('_')])
    else:
        if local_rank == 0:
            print("Using a single GPU!")

    return model