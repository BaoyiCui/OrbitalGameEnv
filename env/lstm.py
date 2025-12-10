
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TrajectoryPredictor(nn.Module):
    """
    一个基于LSTM的轨迹预测器，用于缓解部分可观测性问题。

    该模型接收一个智能体过去一段时间的绝对状态（位置和速度）序列，
    并预测其未来相对于当前位置的**相对位移**轨迹。

    绝对坐标与相对坐标的转换逻辑将在环境的观测函数和数据准备阶段处理。
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 30, num_layers: int = 2):
        """
        初始化模型。

        Args:
            input_dim (int): 输入特征的维度。默认为6 (3D绝对位置 + 3D速度)。
            hidden_dim (int): LSTM隐藏层的维度。
            output_dim (int): 输出特征的维度。默认为30 (10个未来时间步 * 3D相对位移)。
            num_layers (int): LSTM的层数。
        """
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # LSTM遗忘门偏置初始化为1，更容易记住长期信息
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, historical_data_abs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播（方案1：输入相对于序列的最后一个点进行归一化）。

        Args:
            historical_data_abs (torch.Tensor): 历史轨迹数据（绝对位置和速度）。
                                             形状: (batch_size, sequence_length, 6)
            mask (torch.Tensor, optional): 一个布尔或0/1的掩码，标记有效的时间步。
                                        形状: (batch_size, sequence_length)

        Returns:
            torch.Tensor: 预测出的未来10步轨迹的**相对位移**（已展平）。
                          形状: (batch_size, 30)
        """
        lstm_input = historical_data_abs
        batch_size = historical_data_abs.shape[0]
        
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if mask is not None and mask.any():
            lengths = torch.clamp(mask.sum(dim=1).cpu(), min=1).long()
            last_seq_idxs = lengths - 1
            # 获取每个批次项的最后一个有效状态
            last_states = historical_data_abs[torch.arange(batch_size), last_seq_idxs] # (batch_size, 6)
            last_pos = last_states[:, :3].unsqueeze(1) # (batch_size, 1, 3)
            
            # 创建相对于最后一个点的输入
            relative_pos = historical_data_abs[:, :, :3] - last_pos
            # 保持速度为绝对值
            velocities = historical_data_abs[:, :, 3:]
            lstm_input = torch.cat([relative_pos, velocities], dim=-1)

            packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=historical_data_abs.shape[1])
            
            last_timestep_output = lstm_out[torch.arange(batch_size), last_seq_idxs]
        else:
            # 无掩码的简化路径
            last_pos = historical_data_abs[:, -1:, :3] # (batch_size, 1, 3)
            relative_pos = historical_data_abs[:, :, :3] - last_pos
            velocities = historical_data_abs[:, :, 3:]
            lstm_input = torch.cat([relative_pos, velocities], dim=-1)
            
            lstm_out, _ = self.lstm(lstm_input)
            last_timestep_output = lstm_out[:, -1, :]
            
        # 全连接层预测未来的相对位移轨迹
        predicted_trajectory_relative = self.fc(last_timestep_output)
        
        return predicted_trajectory_relative
