
import torch
import torch.nn as nn

class AttentionBasedEncoder(nn.Module):
    """
    基于注意力的观测编码器 (Attention-based Observation Encoder)。

    该模块用于处理POMDP（部分可观测马尔可夫决策过程）环境下的结构化观测信息。
    其核心是利用多头自注意力机制对来自不同源（自身状态、其他智能体、环境）
    的信息进行动态加权和融合，为策略网络提供信息丰富的特征表示。

    架构流程:
    1. 独立嵌入 (Independent Embedding): 对观测的不同组成部分使用独立的MLP进行编码。
    2. 关系建模 (Relational Modeling): 使用多头自注意力机制捕捉各嵌入向量间的关系。
    3. 特征融合 (Feature Fusion): 融合原始自身嵌入和经过注意力加权的上下文向量，并输出最终特征。
    """
    def __init__(self, self_dim, other_dim, ob_dim, lstm_pred_dim, embed_dim=128, nhead=4):
        """
        初始化编码器。

        Args:
            self_dim (int): 智能体自身状态信息的维度。
            other_dim (int): 其他追击者智能体信息的维度。
            ob_dim (int): 环境障碍物信息的维度 (若无则为0)。
            lstm_pred_dim (int): 预测的逃逸者轨迹信息的维度。
            embed_dim (int): 内部嵌入向量的维度。
            nhead (int): 多头注意力机制的头数。
        """
        super().__init__()
        self.self_dim = self_dim
        self.other_dim = other_dim
        self.ob_dim = ob_dim
        self.lstm_pred_dim = lstm_pred_dim

        # --- 1. 独立嵌入层 ---
        # 自身信息编码器 (输入包含自身状态 + LSTM预测)
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim + lstm_pred_dim, embed_dim),
            nn.Tanh()
        )
        
        # 其他智能体信息编码器
        self.other_encoder = None
        if self.other_dim > 0:
            self.other_encoder = nn.Sequential(
                nn.Linear(other_dim, embed_dim),
                nn.Tanh()
            )
        
        # 障碍物信息编码器 (仅在 ob_dim > 0 时创建)
        self.ob_encoder = None
        if self.ob_dim > 0:
            self.ob_encoder = nn.Sequential(
                nn.Linear(ob_dim, embed_dim),
                nn.Tanh()
            )

        # --- 2. 关系建模层 ---
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True)
        self.layer_norm_attn = nn.LayerNorm(embed_dim)

        # --- 3. 特征融合与输出层 ---
        # 输入维度为 h_self 和 h_attn 拼接后的维度 (embed_dim * 2)
        self.final_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 256)
        )
        self.layer_norm_final = nn.LayerNorm(256)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        编码器前向传播。

        Args:
            obs (torch.Tensor): 原始观测张量，应按 [o_self, o_lstm_pred, o_other, o_ob] 顺序拼接。

        Returns:
            torch.Tensor: 编码后的特征向量 h。
        """
        # --- 输入切片 ---
        current_offset = 0
        o_self_state = obs[:, current_offset : current_offset + self.self_dim]
        current_offset += self.self_dim
        o_lstm_pred = obs[:, current_offset : current_offset + self.lstm_pred_dim]
        current_offset += self.lstm_pred_dim
        
        # 将预测轨迹拼接到自身信息中，形成完整的自身相关信息
        full_self_info = torch.cat([o_self_state, o_lstm_pred], dim=-1)

        # --- 1. 独立嵌入 ---
        h_self = self.self_encoder(full_self_info)
        
        # 收集所有实体的嵌入向量，用于构建注意力序列
        entity_embeddings_list = [h_self]
        
        if self.other_encoder is not None:
            o_other = obs[:, current_offset : current_offset + self.other_dim]
            current_offset += self.other_dim
            h_other = self.other_encoder(o_other)
            entity_embeddings_list.append(h_other)

        if self.ob_encoder is not None:
            o_ob = obs[:, current_offset:]
            h_ob = self.ob_encoder(o_ob)
            entity_embeddings_list.append(h_ob)
        
        # --- 2. 关系建模 ---
        # 将实体嵌入列表堆叠成送入注意力机制的序列, shape: (batch_size, num_entities, embed_dim)
        attn_sequence = torch.stack(entity_embeddings_list, dim=1)
        
        # 多头自注意力机制，Q, K, V 均为序列本身
        attn_output, _ = self.attention(attn_sequence, attn_sequence, attn_sequence)
        
        # 提取与自身信息（序列第一个元素）相关的上下文特征向量 h_attn
        h_attn = self.layer_norm_attn(attn_output[:, 0, :])

        # --- 3. 特征融合与输出 ---
        # 强调自身信息：将注意力输出 h_attn 与原始自身嵌入 h_self 拼接
        final_mlp_input = torch.cat([h_self, h_attn], dim=-1)
        
        # 通过最终的MLP得到最终决策特征向量 h
        h = self.final_mlp(final_mlp_input)
        h = self.layer_norm_final(h)
        
        return h
