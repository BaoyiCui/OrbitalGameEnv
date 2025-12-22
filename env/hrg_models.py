import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Base_Student_Encoder(nn.Module):
    """
    基类：处理 Obs 的切分和基础 Embedding。
    所有模型（HAFN, LSTM, MLP）都应基于对观测的相同理解。
    """
    def __init__(self, env_cfg, student_obs_dim, hidden_dim=128):
        super().__init__()
        self.num_teammates = env_cfg.num_p - 1
        self.hidden_dim = hidden_dim
        
        # 维度定义
        # self_dim is calculated based on the observation structure in mpe_pomdp_env.py
        # Self(8) + TargetObs(3*Ne) + Orbital(3)
        self.self_obs_base_dim = 8
        self.target_obs_dim = 3 * env_cfg.num_e
        self.orbital_feat_dim = 3
        self.self_dim = self.self_obs_base_dim + self.target_obs_dim + self.orbital_feat_dim
        self.teammate_dim = 7
        
        # 基础编码器 (MLP)
        self.self_embed = nn.Sequential(
            nn.Linear(self.self_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        if self.num_teammates > 0:
            self.teammate_embed = nn.Sequential(
                nn.Linear(self.teammate_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

    def split_and_embed_obs(self, obs):
        """
        输入: [Batch, Total_Obs_Dim]
        输出: 
           self_emb: [Batch, 1, Hidden]
           team_embs: [Batch, Num_Teammates, Hidden] (如果有队友)
        """
        # The observation structure is:
        # [self_base(8), targets(3*Ne), teammates(7*(Np-1)), orbital(3)]
        # The base encoder combines self_base, targets, and orbital features.
        
        self_part = torch.cat([
            obs[:, :self.self_obs_base_dim],
            obs[:, self.self_obs_base_dim : self.self_obs_base_dim + self.target_obs_dim],
            obs[:, self.self_obs_base_dim + self.target_obs_dim + self.teammate_dim * self.num_teammates:]
        ], dim=1)

        self_emb = self.self_embed(self_part).unsqueeze(1) # [B, 1, H]
        
        team_embs = None
        if self.num_teammates > 0:
            teammates_part = obs[:, self.self_obs_base_dim + self.target_obs_dim : self.self_obs_base_dim + self.target_obs_dim + self.teammate_dim * self.num_teammates]
            teammates_in = teammates_part.view(-1, self.num_teammates, self.teammate_dim)
            team_embs = self.teammate_embed(teammates_in) # [B, N_t, H]
            
        return self_emb, team_embs

# ==========================================
# 1. HAFN Encoder (Proposed Method)
# ==========================================
class HAFN_Encoder(Base_Student_Encoder):
    def __init__(self, env_cfg, student_obs_dim, history_input_dim=6, hidden_dim=128):
        super().__init__(env_cfg, student_obs_dim, hidden_dim)
        
        # --- Stage 1: Entity Attention (处理空间/编队关系) ---
        self.entity_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm_entity = nn.LayerNorm(hidden_dim)
        
        # --- Stage 2: History Cross Attention (处理时序/不完美观测) ---
        self.history_embed = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # [Reverted]: Using a learnable, zero-initialized positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, env_cfg.history_len, hidden_dim))
        
        # Cross Attention: Query来自当前状态, Key/Value来自历史
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim)
        
        # [Reverted]: No FFN layer
        
        # The output is a simple concatenation of the two contexts
        self.output_dim = hidden_dim * 2 

    def forward(self, obs, history, history_mask):
        # === 1. Entity Self-Attention (Pre-LN) ===
        self_emb, team_embs = self.split_and_embed_obs(obs)
        
        if team_embs is not None:
            tokens = torch.cat([self_emb, team_embs], dim=1)
            normed_tokens = self.norm_entity(tokens)
            attn_out, _ = self.entity_attn(normed_tokens, normed_tokens, normed_tokens)
            entity_ctx = (tokens + attn_out)[:, 0:1, :]
        else:
            entity_ctx = self_emb
            
        # === 2. History Processing (Attention only) ===
        B, T, _ = history.shape
        hist_emb = self.history_embed(history)
        # [Reverted]: Add the learnable position embedding
        hist_emb = hist_emb + self.pos_embed[:, :T, :]
        
        key_padding_mask = (history_mask == 0)
        
        # Cross-Attention (Post-LN style, as it was in that version)
        attn_out, attn_weights = self.cross_attn(
            query=entity_ctx,
            key=hist_emb,
            value=hist_emb,
            key_padding_mask=key_padding_mask
        )
        # [Reverted]: Simple Add & Norm, no FFN
        history_ctx = self.norm_cross(entity_ctx + attn_out)
        
        # === 3. Final Output ===
        output = torch.cat([entity_ctx.squeeze(1), history_ctx.squeeze(1)], dim=-1)
        return output, attn_weights

# ==========================================
# 2. LSTM Encoder (Strong Baseline)
# ==========================================
class LSTM_Encoder(Base_Student_Encoder):
    def __init__(self, env_cfg, student_obs_dim, history_input_dim=6, hidden_dim=128):
        super().__init__(env_cfg, student_obs_dim, hidden_dim)
        
        # LSTM to process history
        self.lstm = nn.LSTM(input_size=history_input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        
        # Fusion layer to combine self and team context
        self.obs_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = hidden_dim * 2

    def forward(self, obs, history, history_mask):
        # 1. Process current observation
        self_emb, team_embs = self.split_and_embed_obs(obs) # self_emb: [B, 1, H]
        
        if team_embs is not None:
            # Pool teammates info
            team_ctx = team_embs.mean(dim=1) # [B, H]
        else:
            # If no teammates, use zeros
            team_ctx = torch.zeros(self_emb.shape[0], self.hidden_dim, device=obs.device)
            
        # Fuse self and team context while preserving self's independence
        obs_ctx_combined = torch.cat([self_emb.squeeze(1), team_ctx], dim=1) # [B, 2*H]
        obs_ctx = self.obs_fusion(obs_ctx_combined) # [B, H]

        # 2. Process history with LSTM
        _, (h_n, c_n) = self.lstm(history)
        hist_ctx = h_n[-1] # [B, H]
        
        # 3. Final output
        output = torch.cat([obs_ctx, hist_ctx], dim=-1)
        return output, None

# ==========================================
# 3. MLP Encoder (Simple Baseline)
# ==========================================
class MLP_Encoder(nn.Module):
    def __init__(self, env_cfg, student_obs_dim, history_input_dim=6, hidden_dim=128):
        super().__init__()
        input_dim = student_obs_dim + env_cfg.history_len * history_input_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU()
        )
        self.output_dim = hidden_dim * 2

    def forward(self, obs, history, history_mask):
        B = obs.shape[0]
        hist_flat = history.view(B, -1)
        combined = torch.cat([obs, hist_flat], dim=1)
        output = self.net(combined)
        return output, None

# ==========================================
# 4. Aligned Teacher (For Distillation)
# ==========================================
class Aligned_Teacher(nn.Module):
    """
    维度对齐的教师网络
    接收上帝视角的特权信息，输出一个与学生网络输出维度完全相同的“理想态势感知向量”。
    """
    def __init__(self, priv_obs_dim, student_out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(priv_obs_dim),
            nn.Linear(priv_obs_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, student_out_dim) 
        )
    
    def forward(self, priv_obs):
        return self.net(priv_obs)