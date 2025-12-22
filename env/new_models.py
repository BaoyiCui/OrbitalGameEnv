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
        
        self.self_obs_base_dim = 8
        self.target_obs_dim = 3 * env_cfg.num_e
        self.orbital_feat_dim = 3
        self.self_dim = self.self_obs_base_dim + self.target_obs_dim + self.orbital_feat_dim
        self.teammate_dim = 7
        
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
        self_part = torch.cat([
            obs[:, :self.self_obs_base_dim],
            obs[:, self.self_obs_base_dim : self.self_obs_base_dim + self.target_obs_dim],
            obs[:, self.self_obs_base_dim + self.target_obs_dim + self.teammate_dim * self.num_teammates:]
        ], dim=1)

        self_emb = self.self_embed(self_part).unsqueeze(1)
        
        team_embs = None
        if self.num_teammates > 0:
            teammates_part = obs[:, self.self_obs_base_dim + self.target_obs_dim : self.self_obs_base_dim + self.target_obs_dim + self.teammate_dim * self.num_teammates]
            teammates_in = teammates_part.view(-1, self.num_teammates, self.teammate_dim)
            team_embs = self.teammate_embed(teammates_in)
            
        return self_emb, team_embs

class HAFN_Encoder(Base_Student_Encoder):
    def __init__(self, env_cfg, student_obs_dim, history_input_dim=6, hidden_dim=128):
        super().__init__(env_cfg, student_obs_dim, hidden_dim)
        
        self.entity_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.norm_entity = nn.LayerNorm(hidden_dim)
        
        self.history_embed = nn.Sequential(
            nn.Linear(history_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.register_buffer('pos_embed', self._get_sinusoidal_encoding(env_cfg.history_len, hidden_dim))
        
        self.norm_cross = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid() 
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = hidden_dim

    def _get_sinusoidal_encoding(self, length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, obs, history, history_mask):
        self_emb, team_embs = self.split_and_embed_obs(obs)
        if team_embs is not None:
            tokens = torch.cat([self_emb, team_embs], dim=1)
            normed_tokens = self.norm_entity(tokens)
            attn_out, _ = self.entity_attn(normed_tokens, normed_tokens, normed_tokens)
            current_feat = (tokens + attn_out)[:, 0, :]
        else:
            current_feat = self_emb.squeeze(1)

        B, T, _ = history.shape
        hist_emb = self.history_embed(history) + self.pos_embed[:, :T, :]
        key_padding_mask = (history_mask == 0)
        
        query = current_feat.unsqueeze(1)

        normed_query = self.norm_cross(query)
        attn_out, attn_weights = self.cross_attn(normed_query, hist_emb, hist_emb, key_padding_mask=key_padding_mask)
        hist_feat = query + attn_out

        normed_hist_feat = self.norm_ffn(hist_feat)
        ffn_out = self.ffn(normed_hist_feat)
        hist_feat_processed = (hist_feat + ffn_out).squeeze(1)

        combined_input = torch.cat([current_feat, hist_feat_processed], dim=-1)
        gate = self.fusion_gate(combined_input)
        
        final_feat = current_feat + gate * hist_feat_processed
        
        return self.output_head(final_feat), attn_weights

class LSTM_Encoder(Base_Student_Encoder):
    def __init__(self, env_cfg, student_obs_dim, history_input_dim=6, hidden_dim=128):
        super().__init__(env_cfg, student_obs_dim, hidden_dim)
        
        self.lstm = nn.LSTM(input_size=history_input_dim, hidden_size=hidden_dim, 
                            num_layers=1, batch_first=True)
        
        self.obs_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        self.output_dim = hidden_dim * 2

    def forward(self, obs, history, history_mask):
        self_emb, team_embs = self.split_and_embed_obs(obs)
        
        if team_embs is not None:
            team_ctx = team_embs.mean(dim=1)
        else:
            team_ctx = torch.zeros(self_emb.shape[0], self.hidden_dim, device=obs.device)
            
        obs_ctx_combined = torch.cat([self_emb.squeeze(1), team_ctx], dim=1)
        obs_ctx = self.obs_fusion(obs_ctx_combined)

        _, (h_n, c_n) = self.lstm(history)
        hist_ctx = h_n[-1]
        
        output = torch.cat([obs_ctx, hist_ctx], dim=-1)
        return output, None

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

class Aligned_Teacher(nn.Module):
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
