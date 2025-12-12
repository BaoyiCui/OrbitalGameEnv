import sys
import os
import math
import signal
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import random
from gymnasium import spaces

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from env.hrg_models import HRG_Student_Encoder, Aligned_Teacher

class TrainConfig:
    """训练超参数配置"""
    # 模型与架构
    distil_coef: float = 1.0
    hls_weight_decay: float = 1e-4

    # PPO 核心参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.011
    vf_coef: float = 0.5
    
    # 学习率与优化器
    lr: float = 3e-4
    anneal_lr: bool = True  # [新增] 是否进行学习率衰减
    anneal_ent: bool = True
    ent_anneal_start_frac: float = 0.3
    final_ent_coef: float = 0.0001
    
    # 训练流程
    total_timesteps: int = 5_000_000
    num_steps: int = 2048 # 每个环境采集的步数
    num_envs: int = 4     # [修改] 增加并行环境数，稳定梯度
    num_mini_batches: int = 4
    update_epochs: int = 5
    
    # === 课程学习参数修改 ===
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.8
    
    # [新增] 课程稳定性参数
    curriculum_stability_required: int = 3  # 需要连续3次达标才升级
    curriculum_rollback_threshold: float = 0.4 # 成功率低于0.4时回退

    # 距离定义：m = 距离捕获边界的距离
    initial_m: float = 2000.0
    target_m: float = 80000.0
    ring_width_delta: float = 5000.0
    m_increment: float = 2000.0
    
    initial_dist_cap: float = 50000.0
    min_dist_cap: float = 30000.0
    dist_cap_decrement: float = 500.0

    # 燃料课程
    initial_p_init_dv: float = 500.0
    min_p_init_dv: float = 350.0 # [修改] 提高下限，确保物理可行
    p_init_dv_decrement: float = 20.0

    # 调试与杂项
    debug_critic: bool = False 
    debug_observation: bool = False
    resume_from_checkpoint: str = None
    checkpoint_interval: int = 50
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"hrg_maddpg_robust_{int(time.time())}"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class HRG_ActorCritic(nn.Module):
    def __init__(self, env_cfg: MPE_POMDP_EnvCfg, act_dim: int, priv_obs_dim: int, student_obs_dim: int):
        super().__init__()
        
        d_model = 128
        history_input_dim = 6
        
        # --- 新增: 输入归一化层 ---
        self.history_norm = nn.LayerNorm(history_input_dim)
        
        self.history_embedding = layer_init(nn.Linear(history_input_dim, d_model))
        self.pos_encoder = PositionalEncoding(d_model, max_len=env_cfg.history_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # [修改] 显式传递维度参数，确保 Student 知道 Obs 结构
        # Obs 结构: [Self | Target | Teammates]
        # 注意: 这里的维度需要与环境中的观测空间定义严格对应
        self.student_enc = HRG_Student_Encoder(
            env_cfg,
            student_obs_dim=student_obs_dim,
            self_input_dim=8, 
            target_input_dim=3,   # <--- 必须是 3，根据环境的观测空间定义 (仅位置)
            teammate_input_dim=7, 
            hidden_dim=d_model
        )
        
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(self.student_enc.output_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01)
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        self.teacher_enc = Aligned_Teacher(
            priv_obs_dim, 
            student_out_dim=self.student_enc.output_dim
        )
        
        self.critic = nn.Sequential(
            nn.LayerNorm(priv_obs_dim), # 在输入端对特权信息进行归一化
            layer_init(nn.Linear(priv_obs_dim, 512)), nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

        action_space = spaces.Box(-env_cfg.p_dv_step, env_cfg.p_dv_step, shape=(3,))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def get_history_feats(self, history, history_mask=None):
        normed_history = self.history_norm(history)
        embedded_history = self.history_embedding(normed_history)
        pos_encoded_history = self.pos_encoder(embedded_history.permute(1, 0, 2)).permute(1, 0, 2)
        src_key_padding_mask = (history_mask == 0) if history_mask is not None else None
        transformer_output = self.transformer_encoder(pos_encoded_history, src_key_padding_mask=src_key_padding_mask)
        
        if src_key_padding_mask is not None:
            mask_expanded = ~src_key_padding_mask.unsqueeze(-1).expand_as(transformer_output)
            sum_features = (transformer_output * mask_expanded).sum(dim=1)
            num_unmasked = mask_expanded.sum(dim=1)
            history_features = sum_features / torch.clamp(num_unmasked, min=1e-9)
        else:
            history_features = transformer_output.mean(dim=1)
        return history_features

    def get_value(self, privileged_obs):
        return self.critic(privileged_obs)

    def get_action_and_value(self, obs, privileged_obs, history, history_mask=None, action=None, deterministic=False):
        hist_feats = self.get_history_feats(history, history_mask)
        student_features, attn_weights = self.student_enc(obs, hist_feats)
        action_mean = self.actor_head(student_features)
        
        clipped_logstd = torch.clamp(self.actor_logstd, -2, 1)
        action_std = torch.exp(clipped_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            pre_tanh_action = probs.rsample() if not deterministic else action_mean
            tanh_action = torch.tanh(pre_tanh_action)
            final_action = self.action_bias + self.action_scale * tanh_action
        else:
            unscaled_action = (action - self.action_bias) / self.action_scale
            pre_tanh_action = torch.atanh(torch.clamp(unscaled_action, -1.0 + 1e-6, 1.0 - 1e-6))
            final_action = action
            tanh_action = torch.tanh(pre_tanh_action)

        log_prob_pre_tanh = probs.log_prob(pre_tanh_action).sum(1)
        log_prob_correction = torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6).sum(1)
        log_prob = log_prob_pre_tanh - log_prob_correction
        entropy = probs.entropy().sum(1)
        value = self.get_value(privileged_obs)
        
        return final_action, log_prob, entropy, value, student_features, attn_weights

class CentralizedRolloutBuffer:
    def __init__(self, num_steps, num_envs, num_agents, student_obs_dim, privileged_obs_dim, act_dim, device, history_cfg):
        self.num_steps = num_steps
        self.num_envs = num_envs # [新增]
        self.num_agents = num_agents
        self.device = device
        
        # 增加 num_envs 维度: [step, env, agent, dim]
        self.obs = torch.zeros((num_steps, num_envs, num_agents, student_obs_dim)).to(device)
        self.privileged_obs = torch.zeros((num_steps, num_envs, num_agents, privileged_obs_dim)).to(device)
        self.actions = torch.zeros((num_steps, num_envs, num_agents, act_dim)).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs, num_agents)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs, num_agents)).to(device)
        self.dones = torch.zeros((num_steps, num_envs, num_agents)).to(device)
        self.values = torch.zeros((num_steps, num_envs, num_agents)).to(device)
        
        self.history = torch.zeros((num_steps, num_envs, num_agents, history_cfg.history_len, 6)).to(device)
        self.history_masks = torch.zeros((num_steps, num_envs, num_agents, history_cfg.history_len)).to(device)
        self.step = 0

    def add(self, obs, privileged_obs, actions, logprobs, rewards, dones, values, history, history_masks):
        # 期望输入维度: obs [num_envs, num_agents, dim], 等等
        self.obs[self.step] = obs
        self.privileged_obs[self.step] = privileged_obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        self.history[self.step] = history
        self.history_masks[self.step] = history_masks
        
        self.step = (self.step + 1) % self.num_steps

    def compute_returns(self, next_value, next_done, gamma, gae_lambda):
        # next_value shape: [num_envs, num_agents] (broadcast or mean logic handled outside, but here we expect per agent)
        # next_done shape: [num_envs, num_agents]
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
            
            # 广播计算
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def get(self, mini_batch_size):
        # 将 [step, env, agent] 展平为 [batch_size, agent] 或 [batch_size * agent]
        # 这里我们保持 Agent 维度独立，以便于后续处理 (或者你可以完全展平)
        # PPO通常的做法: Flatten (Step * Env) -> Batch
        # Agent维度通常也Flatten，除非你有特定的 Multi-Agent 结构
        
        # 展平前两个维度: steps 和 envs
        b_obs = self.obs.reshape((-1, self.num_agents, self.obs.shape[-1]))
        b_priv_obs = self.privileged_obs.reshape((-1, self.num_agents, self.privileged_obs.shape[-1]))
        b_actions = self.actions.reshape((-1, self.num_agents, self.actions.shape[-1]))
        b_logprobs = self.logprobs.reshape((-1, self.num_agents))
        b_advantages = self.advantages.reshape((-1, self.num_agents))
        b_returns = self.returns.reshape((-1, self.num_agents))
        b_history = self.history.reshape((-1, self.num_agents, self.history.shape[-2], self.history.shape[-1]))
        b_history_masks = self.history_masks.reshape((-1, self.num_agents, self.history_masks.shape[-1]))
        
        # 完全展平为 samples (Steps * Envs * Agents) 
        # 注意: Transformer 和 Student Encoder 是按 Agent 处理的，所以 Batch 中每个样本应该是一个 Agent 的数据
        batch_size = b_obs.shape[0] * self.num_agents
        
        # 使用 yield 节省内存
        # 需要展平所有数据以进行打乱
        flat_obs = b_obs.reshape(-1, b_obs.shape[-1])
        flat_priv = b_priv_obs.reshape(-1, b_priv_obs.shape[-1])
        flat_act = b_actions.reshape(-1, b_actions.shape[-1])
        flat_log = b_logprobs.reshape(-1)
        flat_adv = b_advantages.reshape(-1)
        flat_ret = b_returns.reshape(-1)
        flat_hist = b_history.reshape(-1, b_history.shape[-2], b_history.shape[-1])
        flat_mask = b_history_masks.reshape(-1, b_history_masks.shape[-1])
        
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = indices[start:end]
            
            yield (
                flat_obs[mb_inds],
                flat_priv[mb_inds],
                flat_act[mb_inds],
                flat_log[mb_inds],
                flat_adv[mb_inds],
                flat_ret[mb_inds],
                flat_hist[mb_inds],
                flat_mask[mb_inds]
            )

def train(cfg: TrainConfig, env_cfg: MPE_POMDP_EnvCfg, all_params: dict):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path("runs") / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 日志记录
    params_path = run_dir / "all_params.txt"
    with open(params_path, "w") as f:
        f.write("--- TrainConfig ---\n")
        for key, value in vars(cfg).items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- MPE_POMDP_EnvCfg ---\n")
        for key, value in vars(env_cfg).items():
            f.write(f"{key}: {value}\n")
        f.write("\n--- All CLI Params ---\n")
        for key, value in sorted(all_params.items()):
            f.write(f"{key}: {value}\n")
    writer = SummaryWriter(str(run_dir))
    progress_path = run_dir / "curriculum_progress_log.txt"

    # === [关键修改] 创建并行环境列表 (简单列表实现，非SubprocVecEnv以避免复杂性) ===
    # 注意: MPE_POMDP_Env 不是线程安全的，如果用多线程需要加锁。这里用单线程循环模拟并行步进。
    envs = [MPE_POMDP_Env(env_cfg) for _ in range(cfg.num_envs)]
    
    # 初始化所有环境的难度
    current_m = cfg.initial_m
    current_dist_cap = cfg.initial_dist_cap
    current_p_init_dv = cfg.initial_p_init_dv
    
    for env in envs:
        env.set_difficulty_parameters(
            m_distance=current_m,
            ring_width_delta=cfg.ring_width_delta,
            p_init_dv=current_p_init_dv, 
            dist_cap=current_dist_cap
        )
    print(f"Init Difficulty: m={current_m}, Cap={current_dist_cap}, Fuel={current_p_init_dv}")

    # 获取维度信息
    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    # 动态获取 Student 观测维度 (已包含新增的轨道特征)
    student_obs_dim = envs[0].observation_spaces[pursuer_ids[0]].shape[0]
    
    # === [修改] 计算 Privileged Observation 维度 ===
    # 结构: [Anchor(6)] + [(Num_P + Num_E - 1) * (Rel_State(6) + Orb_Feat(3))]
    # 假设 Num_E = 1 (Anchor), 那么其余所有 Agent (Num_P) 都有 9 维特征
    # 如果有多个 Evader，除 Anchor 外的 Evader 也有 9 维
    num_others = env_cfg.num_p + (env_cfg.num_e - 1)
    priv_obs_dim = 6 + num_others * 9 
    
    act_dim = envs[0].action_spaces[pursuer_ids[0]].shape[0]

    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim).to(cfg.device)
    
    hls_params, other_params = [], []
    for name, param in agent.named_parameters():
        if not param.requires_grad: continue
        if "hls_" in name: hls_params.append(param)
        else: other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': hls_params, 'weight_decay': cfg.hls_weight_decay},
        {'params': other_params, 'weight_decay': 0.0}
    ], lr=cfg.lr, eps=1e-5)

    # Buffer 包含 num_envs 维度
    buffer = CentralizedRolloutBuffer(
        cfg.num_steps, cfg.num_envs, env_cfg.num_p, 
        student_obs_dim, priv_obs_dim, act_dim, cfg.device, env_cfg
    )

    # [新增] Ctrl+C 信号处理逻辑
    sigint_state = {'count': 0, 'update': 0}
    def sigint_handler(sig, frame):
        sigint_state['count'] += 1
        if sigint_state['count'] == 1:
            print(f"\n[Ctrl+C] 第一次按下。保存当前模型 (update {sigint_state['update']})... 再次按下可强制退出。")
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_interrupt_{sigint_state['update']}.pth"
            checkpoint_data = { 'agent_state_dict': agent.state_dict(), 'env_cfg': vars(env_cfg) }
            torch.save(checkpoint_data, ckpt_path)
            print(f"--- 模型已保存至 {ckpt_path} ---")
        else:
            print("\n[Ctrl+C] 第二次按下。强制退出。")
            sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    recent_episode_stats = deque(maxlen=cfg.curriculum_check_episodes)
    curriculum_stability_counter = 0 # [新增] 课程稳定性计数器
    
    # 记录每个环境的当前Episode回报
    current_ep_returns = np.zeros(cfg.num_envs)
    current_ep_lens = np.zeros(cfg.num_envs)
    
    # Reset all envs
    obss = []
    infos_list = []
    for env in envs:
        o, i = env.reset()
        obss.append(o)
        infos_list.append(i)
        
    global_step = 0
    
    for update in range(1, num_updates + 1):
        sigint_state['update'] = update # 告知信号处理器当前的 update 数

        # === [关键修改] 学习率衰减 (Linear Annealing) ===
        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[1]["lr"] = lrnow # HLS group
        
        # 熵系数衰减
        if cfg.anneal_ent:
            anneal_start_update = cfg.ent_anneal_start_frac * num_updates
            progress = max(0.0, (update - anneal_start_update) / (num_updates - anneal_start_update))
            current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef

        agent.eval() # 设置为评估模式
        
        # Collection Loop
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            current_ep_lens += 1
            
            # --- 1. 整理所有环境的 Observation ---
            # 目标形状: [num_envs, num_agents, dim]
            env_obs_batch = []
            env_priv_batch = []
            env_hist_batch = []
            env_mask_batch = []
            
            for i, (obs, info) in enumerate(zip(obss, infos_list)):
                # 提取每个Agent的数据
                p_obs = [torch.Tensor(obs[pid]) for pid in pursuer_ids]
                p_priv = [torch.Tensor(info[pid]['privileged_state']) for pid in pursuer_ids]
                p_hist = [torch.from_numpy(info[pid][f'history_input_{evader_ids[0]}']).float() for pid in pursuer_ids]
                p_mask = [torch.from_numpy(info[pid][f'history_mask_{evader_ids[0]}']).float() for pid in pursuer_ids]
                
                env_obs_batch.append(torch.stack(p_obs))     # [num_agents, dim]
                env_priv_batch.append(torch.stack(p_priv))
                env_hist_batch.append(torch.stack(p_hist))
                env_mask_batch.append(torch.stack(p_mask))
                
            # Stack 到 Tensor 并移到 GPU
            obs_tensor = torch.stack(env_obs_batch).to(cfg.device) # [num_envs, num_agents, dim]
            priv_tensor = torch.stack(env_priv_batch).to(cfg.device)
            hist_tensor = torch.stack(env_hist_batch).to(cfg.device)
            mask_tensor = torch.stack(env_mask_batch).to(cfg.device)
            
            # [新增] 调试观测信息
            if cfg.debug_observation and ((global_step - cfg.num_envs) // 200 < global_step // 200):
                print(f"\n--- Debug Observation at Step ~{global_step} (p_0, env_0) ---")
                print(f"  - Student Obs: {obs_tensor[0, 0].cpu().numpy().tolist()}")
                print(f"  - History Input (shape): {hist_tensor[0, 0].shape}")
                print(f"  - History Mask (sum): {mask_tensor[0, 0].sum().item()}")
                print("----------------------------------------------------")
            
            # --- 2. Inference (Batch Processing) ---
            with torch.no_grad():
                # 此时输入包含 num_envs 维度，需要小心处理
                # HRG_ActorCritic 的 forward 可以处理任意 batch size
                # 展平 [num_envs, num_agents] -> [N, ...]
                flat_obs = obs_tensor.view(-1, *obs_tensor.shape[2:])
                flat_priv = priv_tensor.view(-1, *priv_tensor.shape[2:])
                flat_hist = hist_tensor.view(-1, *hist_tensor.shape[2:])
                flat_mask = mask_tensor.view(-1, *mask_tensor.shape[2:])
                
                flat_actions, flat_logprob, _, flat_values, _, _ = agent.get_action_and_value(
                    flat_obs, flat_priv, flat_hist, flat_mask
                )
                
                # 恢复维度 [num_envs, num_agents, ...]
                actions_tensor = flat_actions.view(cfg.num_envs, env_cfg.num_p, -1)
                logprob_tensor = flat_logprob.view(cfg.num_envs, env_cfg.num_p)
                values_tensor = flat_values.view(cfg.num_envs, env_cfg.num_p)

            # --- 3. Step Environments ---
            next_obss = []
            next_infos_list = []
            rewards_list = []
            dones_list = []
            
            for i, env in enumerate(envs):
                # 获取该环境的 Action
                action_np = actions_tensor[i].cpu().numpy()
                action_dict = {pid: action_np[j] for j, pid in enumerate(pursuer_ids)}
                # 补全 Evader 动作
                action_dict.update(env.get_evader_actions())
                
                next_o, rew, term, trunc, next_i = env.step(action_dict)
                
                # [修改] 打印详细奖励
                if env_cfg.debug_rewards: # 打印所有环境的信息
                    p0_info = next_i.get(pursuer_ids[0])
                    if p0_info and 'reward_components' in p0_info:
                        rew_info = p0_info['reward_components']
                        rew_str = ", ".join([f"{k}: {v:.3f}" for k, v in rew_info.items()])
                        print(f"Upd {update}, Step {step}, Env {i}, p0 Rewards: {rew_str}")
                
                # 记录数据
                p_rewards = np.array([rew[pid] for pid in pursuer_ids])
                rewards_list.append(p_rewards)
                
                # 处理 Done
                any_done = any(term.values()) or any(trunc.values())
                dones_list.append([any_done] * env_cfg.num_p) # 所有Agent同一done
                
                current_ep_returns[i] += np.mean(p_rewards)
                
                if any_done:
                    # 记录统计
                    final_info = next(iter(next_i.values()), None)
                    if final_info:
                        # [修改点 A] 不再存储累计字典，而是记录单次成功与否 (1.0 或 0.0)
                        is_success = final_info.get('termination_reason') == 'capture_success'
                        recent_episode_stats.append(1.0 if is_success else 0.0)

                        if 'episode_statistics' in final_info:
                            writer.add_scalar(f"charts/ep_return_env_{i}", current_ep_returns[i], global_step)
                            # 为了减少日志刷屏，只打印部分
                            if i == 0:
                                print(f"Upd {update}, Env {i}: Ret={current_ep_returns[i]:.2f}, Len={current_ep_lens[i]}, Reason={final_info.get('termination_reason')}")
                    
                    next_o, next_i = env.reset()
                    current_ep_returns[i] = 0.0
                    current_ep_lens[i] = 0
                
                next_obss.append(next_o)
                next_infos_list.append(next_i)
            
            # --- 4. Store in Buffer ---
            # 转换为 Tensor
            rew_tensor = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(cfg.device)
            done_tensor = torch.tensor(np.array(dones_list), dtype=torch.float32).to(cfg.device)
            
            # 保存 History 需要从之前的 info 中取 (因为是 s_t 的 history)
            buffer.add(
                obs_tensor, priv_tensor, actions_tensor, logprob_tensor, 
                rew_tensor, done_tensor, values_tensor, 
                hist_tensor, mask_tensor
            )
            
            obss = next_obss
            infos_list = next_infos_list
        
        # --- Value Bootstrap ---
        with torch.no_grad():
            # 计算最后一个 Next Value (所有环境)
            next_priv_batch = []
            for i, info in enumerate(infos_list):
                 p_priv = [torch.Tensor(info[pid]['privileged_state']) for pid in pursuer_ids]
                 next_priv_batch.append(torch.stack(p_priv))
            next_priv_tensor = torch.stack(next_priv_batch).to(cfg.device) # [num_envs, num_agents, dim]
            
            flat_next_priv = next_priv_tensor.view(-1, *next_priv_tensor.shape[2:])
            flat_next_vals = agent.get_value(flat_next_priv).view(cfg.num_envs, env_cfg.num_p)
            
            # Done 状态 (假设 reset 后不是 done)
            next_dones = torch.zeros((cfg.num_envs, env_cfg.num_p)).to(cfg.device)
            
            buffer.compute_returns(flat_next_vals, next_dones, cfg.gamma, cfg.gae_lambda)

        # --- Training Update ---
        agent.train()
        b_inds = np.arange(cfg.num_steps * cfg.num_envs * env_cfg.num_p)
        minibatch_size = int(len(b_inds) // cfg.num_mini_batches)
        
        for epoch in range(cfg.update_epochs):
            for b_obs, b_priv, b_act, b_log, b_adv, b_ret, b_hist, b_mask in buffer.get(minibatch_size):
                
                _, new_logprob, entropy, new_value, s_feat, _ = agent.get_action_and_value(
                    b_obs, b_priv, b_hist, b_mask, b_act
                )
                new_value = new_value.view(-1)
                
                logratio = new_logprob - b_log
                ratio = logratio.exp()
                
                # Norm Adv
                mb_adv = b_adv
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1-cfg.clip_coef, 1+cfg.clip_coef)).mean()
                v_loss = 0.5 * ((new_value - b_ret) ** 2).mean()
                entropy_loss = entropy.mean()
                
                # Distillation
                with torch.no_grad():
                    teacher_targets = agent.teacher_enc(b_priv)
                distil_loss = F.mse_loss(s_feat, teacher_targets)
                
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss + cfg.distil_coef * distil_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", entropy_loss.item(), global_step)
        writer.add_scalar("losses/distillation_loss", distil_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/entropy_coef", current_ent_coef, global_step)

        # [新增] 保存检查点
        if update % cfg.checkpoint_interval == 0:
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"ckpt_{update}.pth"
            
            checkpoint_data = {
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'global_step': global_step,
                'env_cfg': vars(env_cfg),
                'train_cfg': vars(cfg),
                'curriculum': {
                    'current_m': current_m,
                    'current_dist_cap': current_dist_cap,
                    'current_p_init_dv': current_p_init_dv,
                }
            }
            
            torch.save(checkpoint_data, ckpt_path)
            print(f"--- Checkpoint saved to {ckpt_path} at update {update} ---")

        # === [关键修改] 鲁棒的课程学习逻辑 ===
        # [修改点 B] 采用滑动平均值计算成功率，避免多环境数据混淆
        if len(recent_episode_stats) >= cfg.curriculum_check_episodes:
            current_sr = np.mean(recent_episode_stats)
            writer.add_scalar("charts/success_rate", current_sr, global_step)

            # --- 1. 升级逻辑 (增加 Stability 检查) ---
            if current_sr >= cfg.success_rate_threshold:
                curriculum_stability_counter += 1
            else:
                curriculum_stability_counter = 0 # 中断则重置
            
            # 只有连续 N 次达标才升级
            if curriculum_stability_counter >= cfg.curriculum_stability_required:
                changed = False
                if current_m < cfg.target_m:
                    current_m = min(current_m + cfg.m_increment, cfg.target_m)
                    changed = True
                if current_p_init_dv > cfg.min_p_init_dv:
                    current_p_init_dv = max(current_p_init_dv - cfg.p_init_dv_decrement, cfg.min_p_init_dv)
                    changed = True
                if current_dist_cap > cfg.min_dist_cap:
                    current_dist_cap = max(current_dist_cap - cfg.dist_cap_decrement, cfg.min_dist_cap)
                    changed = True
                
                if changed:
                    print(f"\n*** [UPGRADE] SR={current_sr:.2f} (Stable {curriculum_stability_counter}). New: m={current_m}, Fuel={current_p_init_dv} ***")
                    # 应用到所有环境
                    for env in envs:
                        env.set_difficulty_parameters(m_distance=current_m, p_init_dv=current_p_init_dv, dist_cap=current_dist_cap)
                    recent_episode_stats.clear()
                    curriculum_stability_counter = 0 # 升级后重置计数
                    
                    # 记录日志
                    with open(progress_path, "a") as f:
                        f.write(f"{update}\t{global_step}\t{current_sr:.3f}\t{current_m}\t{current_dist_cap}\t{current_p_init_dv}\tUPGRADE\n")

                # [新增] 早停检查: 无论本次是否升级(changed)，只要满足课程目标就检查
                if current_m >= cfg.target_m and \
                   current_p_init_dv <= cfg.min_p_init_dv and \
                   current_dist_cap <= cfg.min_dist_cap:
                    print(f"\n--- [EARLY STOPPING] Curriculum target reached at update {update}. Stopping training. ---")
                    break # 退出主训练循环

            # --- 2. 降级逻辑 (救命回退) ---
            elif current_sr < cfg.curriculum_rollback_threshold and current_m > cfg.initial_m:
                print(f"\n!!! [ROLLBACK] SR={current_sr:.2f} < {cfg.curriculum_rollback_threshold}. DETECTED COLLAPSE !!!")
                
                # 回退操作：难度大幅降低
                current_m = max(current_m - cfg.m_increment * 2, cfg.initial_m)
                current_p_init_dv = min(current_p_init_dv + cfg.p_init_dv_decrement * 2, cfg.initial_p_init_dv)
                current_dist_cap = min(current_dist_cap + cfg.dist_cap_decrement * 2, cfg.initial_dist_cap)
                
                for env in envs:
                    env.set_difficulty_parameters(m_distance=current_m, p_init_dv=current_p_init_dv, dist_cap=current_dist_cap)
                recent_episode_stats.clear()
                curriculum_stability_counter = 0
                
                with open(progress_path, "a") as f:
                        f.write(f"{update}\t{global_step}\t{current_sr:.3f}\t{current_m}\t{current_dist_cap}\t{current_p_init_dv}\tROLLBACK\n")

    # [新增] 保存训练完成的最终模型
    print(f"\n--- 训练结束于 update {update}。保存最终模型... ---")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_final_{update}.pth"
    checkpoint_data = { 'agent_state_dict': agent.state_dict(), 'env_cfg': vars(env_cfg) }
    torch.save(checkpoint_data, ckpt_path)
    print(f"--- 最终模型已保存至 {ckpt_path} ---")

    for env in envs: env.close()
    writer.close()

if __name__ == "__main__":
    train(TrainConfig(), MPE_POMDP_EnvCfg(), {})