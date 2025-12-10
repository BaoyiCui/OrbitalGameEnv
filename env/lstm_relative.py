import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RelativeTrajectoryPredictor(nn.Module):
    """
    一个基于LSTM的轨迹预测器，用于缓解部分可观测性问题。
    该模型根据不同的方案（scheme）处理相对坐标。

    该模型接收一个智能体过去一段时间的轨迹序列，并预测其未来相对于当前位置的**相对位移**轨迹。
    模型的输出接口与原始TrajectoryPredictor保持一致，以便于集成。
    """
    def __init__(self, scheme: int, input_dim: int = 6, hidden_dim: int = 128, output_dim: int = 30, num_layers: int = 2):
        """
        初始化模型。

        Args:
            scheme (int): 使用的方案 (2 或 3)。
            input_dim (int): 输入特征的维度。默认为6 (3D位置 + 3D速度)。
            hidden_dim (int): LSTM隐藏层的维度。
            output_dim (int): 输出特征的维度。默认为30 (10个未来时间步 * 3D相对位移)。
            num_layers (int): LSTM的层数。
        """
        super(RelativeTrajectoryPredictor, self).__init__()
        if scheme not in [2, 3]:
            raise ValueError("Scheme must be 2 or 3.")
        self.scheme = scheme
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
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, historical_data: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播。

        Args:
            historical_data (torch.Tensor): 历史轨迹数据。
                                            对于方案2，这应该是相对于虚拟星的相对状态。
                                            对于方案3，这应该是绝对状态。
                                            形状: (batch_size, sequence_length, 6)
            mask (torch.Tensor, optional): 一个布尔或0/1的掩码，标记有效的时间步。
                                            形状: (batch_size, sequence_length)

        Returns:
            torch.Tensor: 预测出的未来10步轨迹的**相对位移**（相对于当前位置）。
                          形状: (batch_size, 30)
        """
        if self.scheme == 2:
            # 方案2：输入是相对于动态传播的虚拟星的相对状态。
            # 这个转换逻辑非常复杂，因为它需要轨道动力学传播，
            # 这必须在环境（environment）中完成。
            # 因此，此模型假定 `historical_data` 已经是计算好的相对状态。
            lstm_input = historical_data
        
        elif self.scheme == 3:
            # 方案3：输入相对于序列的第一个点进行归一化。
            batch_size = historical_data.shape[0]
            ref_pos = torch.zeros(batch_size, 1, 3, device=historical_data.device)

            if mask is not None and mask.any():
                # 在有掩码的情况下，第一个有效点是参考点
                # 注意：这假定掩码至少有一个有效点
                first_valid_indices = torch.zeros(batch_size, dtype=torch.long) # 简化处理，取第一个
                ref_states = historical_data[torch.arange(batch_size), first_valid_indices]
                ref_pos = ref_states[:, :3].unsqueeze(1)
            else:
                # 无掩码时，取序列的第一个点
                ref_pos = historical_data[:, 0:1, :3]

            relative_pos = historical_data[:, :, :3] - ref_pos
            velocities = historical_data[:, :, 3:] # 速度保持绝对值
            lstm_input = torch.cat([relative_pos, velocities], dim=-1)

        # --- 通用LSTM处理逻辑 ---
        batch_size = lstm_input.shape[0]
        
        if mask is not None and mask.dim() == 1:
            mask = mask.unsqueeze(0)

        if mask is not None and mask.any():
            lengths = torch.clamp(mask.sum(dim=1).cpu(), min=1).long()
            packed_input = pack_padded_sequence(lstm_input, lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=lstm_input.shape[1])
            
            last_seq_idxs = (lengths - 1).long()
            last_timestep_output = lstm_out[torch.arange(batch_size), last_seq_idxs]
        else:
            lstm_out, _ = self.lstm(lstm_input)
            last_timestep_output = lstm_out[:, -1, :]
            
        # 全连接层预测未来的相对位移轨迹
        # 模型的学习目标始终是相对于当前位置的位移，
        # 因此无论输入参考系如何，输出都应符合环境的期望。
        predicted_trajectory_relative = self.fc(last_timestep_output)
        
        return predicted_trajectory_relative