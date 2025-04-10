import torch
from torch import nn

from .mlp import MultiLayerPerceptron

class STID(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
            int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
            self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # 动态调整时间索引，确保适配不同数据集
            # 假设 t_i_d_data 是原始时间步索引，转换为一天内的索引
            time_in_day_indices = (t_i_d_data[:, -1, :] % self.time_of_day_size).type(torch.LongTensor)
            if time_in_day_indices.max().item() >= self.time_of_day_size or time_in_day_indices.min().item() < 0:
                raise ValueError(f"Time in day indices out of bounds: min {time_in_day_indices.min().item()}, max {time_in_day_indices.max().item()}, expected [0, {self.time_of_day_size-1}]")
            time_in_day_emb = self.time_in_day_emb[time_in_day_indices]
        else:
            time_in_day_emb = None

        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            # 确保 d_i_w_data 在 [0, 1] 范围内，并映射到 [0, 6]
            # 假设 d_i_w_data 的原始值是 [0, 6]（表示一周的 7 天），需要归一化到 [0, 1] 再重新映射
            # 如果原始数据已经是 [0, 1] 以外的范围，我们需要先将其归一化
            d_i_w_data = d_i_w_data - d_i_w_data.min()  # 确保最小值为 0
            d_i_w_data = d_i_w_data / (d_i_w_data.max() + 1e-8)  # 归一化到 [0, 1]
            day_in_week_indices = (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            # 确保索引在 [0, 6] 范围内
            day_in_week_indices = torch.clamp(day_in_week_indices, 0, self.day_of_week_size - 1)
            if day_in_week_indices.max().item() >= self.day_of_week_size or day_in_week_indices.min().item() < 0:
                raise ValueError(f"Day in week indices out of bounds: min {day_in_week_indices.min().item()}, max {day_in_week_indices.max().item()}, expected [0, {self.day_of_week_size-1}]")
            day_in_week_emb = self.day_in_week_emb[day_in_week_indices]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb_expanded = self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
            node_emb.append(node_emb_expanded)

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb_expanded = time_in_day_emb.transpose(1, 2).unsqueeze(-1)
            tem_emb.append(tem_emb_expanded)
        if day_in_week_emb is not None:
            tem_emb_expanded = day_in_week_emb.transpose(1, 2).unsqueeze(-1)
            tem_emb.append(tem_emb_expanded)

        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction