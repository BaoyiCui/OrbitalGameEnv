
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TrajectoryPredictor(nn.Module):
    """
    一个基于LSTM的轨迹预测器，用于缓解部分可观测性问题。

    该模型接收一个智能体过去一段时间的绝对状态（位置和速度）序列，
    并直接预测其未来的绝对位置轨迹。

    坐标系转换的逻辑将在环境的观测函数中处理，以保持模型的简洁性。
    """
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 30, num_layers: int = 2):
        """
        初始化模型。

        Args:
            input_dim (int): 输入特征的维度。默认为6 (3D绝对位置 + 3D速度)。
            hidden_dim (int): LSTM隐藏层的维度。
            output_dim (int): 输出特征的维度。默认为30 (10个未来时间步 * 3D绝对位置)。
            num_layers (int): LSTM的层数。
        """
        super(TrajectoryPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, historical_data_abs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播（绝对 -> 绝对）。

        Args:
            historical_data_abs (torch.Tensor): 历史轨迹数据（绝对位置和速度）。
                                                 形状: (batch_size, sequence_length, 6)
            mask (torch.Tensor, optional): 一个布尔或0/1的掩码，标记有效的时间步。
                                            形状: (batch_size, sequence_length)

        Returns:
            torch.Tensor: 预测出的未来10步轨迹的绝对坐标（已展平）。
                          形状: (batch_size, 30)
        """
        # 模型直接在绝对坐标系下学习
        lstm_input = historical_data_abs

        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if mask is not None and mask.any():
            lengths = torch.clamp(mask.sum(dim=1).cpu(), min=1)
            packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=historical_data_abs.shape[1])
            
            batch_size = lstm_out.shape[0]
            last_seq_idxs = (lengths - 1).long()
            last_timestep_output = lstm_out[torch.arange(batch_size), last_seq_idxs]
        else:
            lstm_out, _ = self.lstm(lstm_input)
            last_timestep_output = lstm_out[:, -1, :]
            
        # 全连接层预测出未来的绝对坐标轨迹
        predicted_trajectory_abs = self.fc(last_timestep_output)
        
        return predicted_trajectory_abs
