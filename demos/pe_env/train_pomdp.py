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
from env.mpe_env import GlobalTrainManager
from env.hrg_models import HAFN_Encoder, LSTM_Encoder, MLP_Encoder, Aligned_Teacher

class TrainConfig:
    """训练超参数配置"""
    # 模型与架构
    student_model_type: str = 'hafn'
    use_distillation: bool = True
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
    anneal_lr: bool = True
    anneal_ent: bool = True
    ent_anneal_start_frac: float = 0.3
    final_ent_coef: float = 0.0001
    
    # 训练流程
    total_timesteps: int = 5_000_000
    num_steps: int = 2048
    num_envs: int = 1
    num_mini_batches: int = 4
    update_epochs: int = 5
    
    # === 课程学习参数修改 ===
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.8
    
    curriculum_stability_required: int = 5
    curriculum_rollback_threshold: float = 0.4

    # 距离定义：m = 距离捕获边界的距离
    initial_m: float = 2000.0
    target_m: float = 80000.0
    ring_width_delta: float = 5000.0
    m_increment: float = 2000.0

    # 燃料课程
    initial_p_init_dv: float = 500.0
    min_p_init_dv: float = 350.0
    p_init_dv_decrement: float = 20.0

    # 调试与杂项
    debug_critic: bool = False 
    debug_observation: bool = False
    resume_from_checkpoint: str = None
    checkpoint_interval: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"hrg_maddpg_robust_{int(time.time())}"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class HRG_ActorCritic(nn.Module):
    def __init__(self, env_cfg: MPE_POMDP_EnvCfg, act_dim: int, priv_obs_dim: int, student_obs_dim: int, student_model_type: str = 'hafn'):
        super().__init__()
        self.student_model_type = student_model_type
        self.env_cfg = env_cfg
        
        hidden_dim = 128
        history_input_dim = 6
        
        # === 1. 初始化 Student Encoder (根据类型选择) ===
        if student_model_type == 'hafn':
            self.student_encoder = HAFN_Encoder(
                env_cfg, student_obs_dim, history_input_dim, hidden_dim
            )
        elif student_model_type == 'lstm':
            self.student_encoder = LSTM_Encoder(
                env_cfg, student_obs_dim, history_input_dim, hidden_dim
            )
        elif student_model_type == 'mlp':
            self.student_encoder = MLP_Encoder(
                env_cfg, student_obs_dim, history_input_dim, hidden_dim
            )
        else:
            raise ValueError(f"Unknown student_model_type: {student_model_type}")
            
        student_output_dim = self.student_encoder.output_dim

        # === 2. Actor Head ===
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(student_output_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01)
        )
        
        # Actor 噪声参数
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        # === 3. Critic (Value Function) ===
        # Critic 输入的是特权信息 (Privileged Obs)
        self.critic = nn.Sequential(
            nn.LayerNorm(priv_obs_dim),
            layer_init(nn.Linear(priv_obs_dim, 512)), nn.LayerNorm(512), nn.ReLU(),
            layer_init(nn.Linear(512, 256)), nn.LayerNorm(256), nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

        # === 4. Distillation Teacher (用于辅助学生学习) ===
        self.teacher_enc = Aligned_Teacher(priv_obs_dim, student_out_dim=student_output_dim)
        
        # 动作缩放 (Action Scaling)
        action_space = spaces.Box(-env_cfg.p_dv_step, env_cfg.p_dv_step, shape=(3,))
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

    def get_student_features(self, obs, history, history_mask):
        # 所有的逻辑都委托给具体的 encoder
        return self.student_encoder(obs, history, history_mask)

    def get_value(self, privileged_obs):
        return self.critic(privileged_obs)

    def get_action_and_value(self, obs, privileged_obs, history, history_mask=None, action=None, deterministic=False):
        # 1. 获取特征
        student_features, attn_weights = self.get_student_features(obs, history, history_mask)
        
        # 2. 计算动作分布
        action_mean = self.actor_head(student_features)
        clipped_logstd = torch.clamp(self.actor_logstd, -2, 1)
        action_std = torch.exp(clipped_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        # 3. 采样动作
        if action is None:
            pre_tanh_action = probs.rsample() if not deterministic else action_mean
            tanh_action = torch.tanh(pre_tanh_action)
            final_action = self.action_bias + self.action_scale * tanh_action
        else:
            unscaled_action = (action - self.action_bias) / self.action_scale
            pre_tanh_action = torch.atanh(torch.clamp(unscaled_action, -1.0 + 1e-6, 1.0 - 1e-6))
            final_action = action
            tanh_action = torch.tanh(pre_tanh_action)

        log_prob = probs.log_prob(pre_tanh_action).sum(1) - torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6).sum(1)
        entropy = probs.entropy().sum(1)
        
        # 4. 计算价值
        value = self.get_value(privileged_obs)
        
        # 5. 教师蒸馏目标
        with torch.no_grad():
            distil_target = self.teacher_enc(privileged_obs)

        return final_action, log_prob, entropy, value, student_features, distil_target

class CentralizedRolloutBuffer:
    def __init__(self, num_steps, num_envs, num_agents, student_obs_dim, privileged_obs_dim, act_dim, device, history_cfg):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.device = device
        
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
        self.advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1].float()
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            self.advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        self.returns = self.advantages + self.values

    def get(self, mini_batch_size):
        batch_size = self.num_steps * self.num_envs * self.num_agents
        
        flat_obs = self.obs.reshape(-1, self.obs.shape[-1])
        flat_priv = self.privileged_obs.reshape(-1, self.privileged_obs.shape[-1])
        flat_act = self.actions.reshape(-1, self.actions.shape[-1])
        flat_log = self.logprobs.reshape(-1)
        flat_adv = self.advantages.reshape(-1)
        flat_ret = self.returns.reshape(-1)
        flat_hist = self.history.reshape(-1, self.history.shape[-2], self.history.shape[-1])
        flat_mask = self.history_masks.reshape(-1, self.history_masks.shape[-1])
        
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
    with open(progress_path, "w") as f:
        f.write("Update(Upd),GlobalStep(GS),SuccessRate(SR),Distance_m(M),Fuel_dv(DV),RingRatio(RR),EventType(Event)\n")

    envs = [MPE_POMDP_Env(env_cfg) for _ in range(cfg.num_envs)]
    
    current_m = cfg.initial_m
    current_p_init_dv = cfg.initial_p_init_dv
    
    for env in envs:
        env.set_difficulty_parameters(
            m_distance=current_m,
            ring_width_delta=cfg.ring_width_delta,
            p_init_dv=current_p_init_dv
        )
    print(f"Init Difficulty: m={current_m}, Fuel={current_p_init_dv}")

    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    student_obs_dim = envs[0].observation_spaces[pursuer_ids[0]].shape[0]
    
    num_others = env_cfg.num_p + (env_cfg.num_e - 1)
    priv_obs_dim = 6 + num_others * 9 
    act_dim = envs[0].action_spaces[pursuer_ids[0]].shape[0]

    agent = HRG_ActorCritic(env_cfg, act_dim, priv_obs_dim, student_obs_dim, student_model_type=cfg.student_model_type).to(cfg.device)
    
    hls_params, other_params = [], []
    for name, param in agent.named_parameters():
        if not param.requires_grad: continue
        
        # [FIXED]: Recognize both "hls_" and the new "fusion_gate"
        if "hls_" in name or "fusion_gate" in name: 
            hls_params.append(param)
        else: 
            other_params.append(param)

    optimizer = torch.optim.Adam([
        {'params': hls_params, 'weight_decay': cfg.hls_weight_decay},
        {'params': other_params, 'weight_decay': 0.0}
    ], lr=cfg.lr, eps=1e-5)

    buffer = CentralizedRolloutBuffer(
        cfg.num_steps, cfg.num_envs, env_cfg.num_p, 
        student_obs_dim, priv_obs_dim, act_dim, cfg.device, env_cfg
    )

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
    curriculum_stability_counter = 0
    
    current_ep_returns = np.zeros(cfg.num_envs)
    current_ep_lens = np.zeros(cfg.num_envs)
    
    obss = []
    infos_list = []
    for env in envs:
        o, i = env.reset()
        obss.append(o)
        infos_list.append(i)
        
    global_step = 0
    
    for update in range(1, num_updates + 1):
        sigint_state['update'] = update

        GlobalTrainManager.update_ratios(global_step, env_cfg)
        
        if global_step > 0 and (update % 100 == 0):
             print(f"Curriculum Update: Ring Ratio = {GlobalTrainManager.current_ring_ratio:.3f}, Failed Pool Size = {len(GlobalTrainManager.failed_seeds)}")

        if cfg.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * cfg.lr
            optimizer.param_groups[0]["lr"] = lrnow
            optimizer.param_groups[1]["lr"] = lrnow
        
        if cfg.anneal_ent:
            anneal_start_update = cfg.ent_anneal_start_frac * num_updates
            progress = max(0.0, (update - anneal_start_update) / (num_updates - anneal_start_update))
            current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef

        agent.eval()
        
        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            current_ep_lens += 1
            
            env_obs_batch = []
            env_priv_batch = []
            env_hist_batch = []
            env_mask_batch = []
            
            for i, (obs, info) in enumerate(zip(obss, infos_list)):
                p_obs = [torch.Tensor(obs[pid]) for pid in pursuer_ids]
                p_priv = [torch.Tensor(info[pid]['privileged_state']) for pid in pursuer_ids]
                p_hist = [torch.from_numpy(info[pid][f'history_input_{evader_ids[0]}']).float() for pid in pursuer_ids]
                p_mask = [torch.from_numpy(info[pid][f'history_mask_{evader_ids[0]}']).float() for pid in pursuer_ids]
                
                env_obs_batch.append(torch.stack(p_obs))
                env_priv_batch.append(torch.stack(p_priv))
                env_hist_batch.append(torch.stack(p_hist))
                env_mask_batch.append(torch.stack(p_mask))
                
            obs_tensor = torch.stack(env_obs_batch).to(cfg.device)
            priv_tensor = torch.stack(env_priv_batch).to(cfg.device)
            hist_tensor = torch.stack(env_hist_batch).to(cfg.device)
            mask_tensor = torch.stack(env_mask_batch).to(cfg.device)
            
            if cfg.debug_observation and ((global_step - cfg.num_envs) // 200 < global_step // 200):
                print(f"\n--- Debug Observation at Step ~{global_step} (p_0, env_0) ---")
                print(f"  - Student Obs: {obs_tensor[0, 0].cpu().numpy().tolist()}")
                print(f"  - History Input (shape): {hist_tensor[0, 0].shape}")
                print(f"  - History Mask (sum): {mask_tensor[0, 0].sum().item()}")
                print("----------------------------------------------------")
            
            with torch.no_grad():
                flat_obs = obs_tensor.view(-1, *obs_tensor.shape[2:])
                flat_priv = priv_tensor.view(-1, *priv_tensor.shape[2:])
                flat_hist = hist_tensor.view(-1, *hist_tensor.shape[2:])
                flat_mask = mask_tensor.view(-1, *mask_tensor.shape[2:])
                
                flat_actions, flat_logprob, _, flat_values, _, _ = agent.get_action_and_value(
                    flat_obs, flat_priv, flat_hist, flat_mask
                )
                
                actions_tensor = flat_actions.view(cfg.num_envs, env_cfg.num_p, -1)
                logprob_tensor = flat_logprob.view(cfg.num_envs, env_cfg.num_p)
                values_tensor = flat_values.view(cfg.num_envs, env_cfg.num_p)

            next_obss = []
            next_infos_list = []
            rewards_list = []
            dones_list = []
            
            for i, env in enumerate(envs):
                action_np = actions_tensor[i].cpu().numpy()
                action_dict = {pid: action_np[j] for j, pid in enumerate(pursuer_ids)}
                action_dict.update(env.get_evader_actions())
                
                next_o, rew, term, trunc, next_i = env.step(action_dict)
                
                if env_cfg.debug_rewards:
                    p0_info = next_i.get(pursuer_ids[0])
                    if p0_info and 'reward_components' in p0_info:
                        rew_info = p0_info['reward_components']
                        rew_str = ", ".join([f"{k}: {v:.3f}" for k, v in rew_info.items()])
                        print(f"Upd {update}, Step {step}, Env {i}, p0 Rewards: {rew_str}")
                
                p_rewards = np.array([rew[pid] for pid in pursuer_ids])
                rewards_list.append(p_rewards)
                
                any_done = any(term.values()) or any(trunc.values())
                dones_list.append([any_done] * env_cfg.num_p)
                
                current_ep_returns[i] += np.mean(p_rewards)
                
                if any_done:
                    final_info = next(iter(next_i.values()), None)
                    if final_info:
                        is_success = final_info.get('termination_reason') == 'capture_success'
                        recent_episode_stats.append(1.0 if is_success else 0.0)

                        if 'episode_statistics' in final_info:
                            writer.add_scalar(f"charts/ep_return_env_{i}", current_ep_returns[i], global_step)
                            if i == 0:
                                print(f"Upd {update}, Env {i}: Ret={current_ep_returns[i]:.2f}, Len={current_ep_lens[i]}, Reason={final_info.get('termination_reason')}")
                    
                    next_o, next_i = env.reset()
                    current_ep_returns[i] = 0.0
                    current_ep_lens[i] = 0
                
                next_obss.append(next_o)
                next_infos_list.append(next_i)
            
            rew_tensor = torch.tensor(np.array(rewards_list), dtype=torch.float32).to(cfg.device)
            done_tensor = torch.tensor(np.array(dones_list), dtype=torch.float32).to(cfg.device)
            
            buffer.add(
                obs_tensor, priv_tensor, actions_tensor, logprob_tensor, 
                rew_tensor, done_tensor, values_tensor, 
                hist_tensor, mask_tensor
            )
            
            obss = next_obss
            infos_list = next_infos_list
        
        with torch.no_grad():
            next_priv_batch = []
            for i, info in enumerate(infos_list):
                 p_priv = [torch.Tensor(info[pid]['privileged_state']) for pid in pursuer_ids]
                 next_priv_batch.append(torch.stack(p_priv))
            next_priv_tensor = torch.stack(next_priv_batch).to(cfg.device)
            
            flat_next_priv = next_priv_tensor.view(-1, *next_priv_tensor.shape[2:])
            flat_next_vals = agent.get_value(flat_next_priv).view(cfg.num_envs, env_cfg.num_p)
            
            next_dones = torch.zeros((cfg.num_envs, env_cfg.num_p)).to(cfg.device)
            
            buffer.compute_returns(flat_next_vals, next_dones, cfg.gamma, cfg.gae_lambda)

        agent.train()
        
        for epoch in range(cfg.update_epochs):
            for b_obs, b_priv, b_act, b_log, b_adv, b_ret, b_hist, b_mask in buffer.get(cfg.num_mini_batches):
                
                _, new_logprob, entropy, new_value, s_feat, distil_target = agent.get_action_and_value(
                    b_obs, b_priv, b_hist, b_mask, b_act
                )
                new_value = new_value.view(-1)
                
                logratio = new_logprob - b_log
                ratio = logratio.exp()
                
                mb_adv = b_adv
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1-cfg.clip_coef, 1+cfg.clip_coef)).mean()
                v_loss = 0.5 * ((new_value - b_ret) ** 2).mean()
                entropy_loss = entropy.mean()
                
                if cfg.use_distillation:
                    distil_loss = F.mse_loss(s_feat, distil_target)
                else:
                    distil_loss = torch.tensor(0.0).to(cfg.device)
                
                loss = pg_loss + cfg.vf_coef * v_loss - current_ent_coef * entropy_loss + cfg.distil_coef * distil_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/entropy_loss", (-current_ent_coef * entropy_loss).item(), global_step)
        writer.add_scalar("losses/distillation_loss", distil_loss.item(), global_step)
        writer.add_scalar("losses/total_loss", loss.item(), global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/entropy_coef", current_ent_coef, global_step)
        writer.add_scalar("policy/mean_entropy", entropy_loss.item(), global_step)

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
                    'current_p_init_dv': current_p_init_dv,
                }
            }
            
            torch.save(checkpoint_data, ckpt_path)
            print(f"--- Checkpoint saved to {ckpt_path} at update {update} ---")

        if len(recent_episode_stats) >= cfg.curriculum_check_episodes:
            current_sr = np.mean(recent_episode_stats)
            writer.add_scalar("charts/success_rate", current_sr, global_step)

            physical_difficulty_reached = (current_m >= cfg.target_m and 
                                          current_p_init_dv <= cfg.min_p_init_dv)
            formation_curriculum_reached = (global_step >= env_cfg.formation_curriculum_end_step)
            high_success_rate = current_sr > 0.8

            if physical_difficulty_reached and formation_curriculum_reached and high_success_rate:
                print(f"\n--- [EARLY STOPPING] Conditions met (SR > 0.8, Physical & Formation curriculum finished). Stopping training at update {update}. ---")
                ckpt_dir = run_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f"ckpt_early_stop_{update}.pth"
                checkpoint_data = { 'agent_state_dict': agent.state_dict(), 'env_cfg': vars(env_cfg) }
                torch.save(checkpoint_data, ckpt_path)
                print(f"--- Early stopping checkpoint saved to {ckpt_path} ---")
                break 
            
            elif current_sr >= cfg.success_rate_threshold and not physical_difficulty_reached:
                curriculum_stability_counter += 1
                print(f"--- [Progress] SR={current_sr:.2f} >= {cfg.success_rate_threshold}. Stability count: {curriculum_stability_counter}/{cfg.curriculum_stability_required} ---")
                
                if curriculum_stability_counter >= cfg.curriculum_stability_required:
                    print(f"\n+++ [UPGRADE] SR={current_sr:.2f} is stable. Increasing difficulty. +++")
                    current_m = min(current_m + cfg.m_increment, cfg.target_m)
                    current_p_init_dv = max(current_p_init_dv - cfg.p_init_dv_decrement, cfg.min_p_init_dv)
                    
                    for env in envs:
                        env.set_difficulty_parameters(m_distance=current_m, p_init_dv=current_p_init_dv)
                    
                    print(f"    New Difficulty: m={current_m}, Fuel={current_p_init_dv}")
                    recent_episode_stats.clear()
                    curriculum_stability_counter = 0
                    
                    with open(progress_path, "a") as f:
                        f.write(f"{update},{global_step},{current_sr:.3f},{current_m},{current_p_init_dv},{GlobalTrainManager.current_ring_ratio:.3f},UPGRADE\n")

            elif current_sr < cfg.curriculum_rollback_threshold:
                if physical_difficulty_reached:
                    print(f"\n!!! [ROLLBACK-Formation] SR={current_sr:.2f} < {cfg.curriculum_rollback_threshold}. Physical curriculum finished. Rolling back formation only. !!!")
                    GlobalTrainManager.current_ring_ratio = min(GlobalTrainManager.current_ring_ratio + 0.1, env_cfg.start_ring_ratio)
                    print(f"    New Ring Ratio: {GlobalTrainManager.current_ring_ratio:.2f}")
                    
                    with open(progress_path, "a") as f:
                        f.write(f"{update},{global_step},{current_sr:.3f},{current_m},{current_p_init_dv},{GlobalTrainManager.current_ring_ratio:.3f},ROLLBACK_FORMATION\n")

                elif current_m > cfg.initial_m:
                    print(f"\n!!! [ROLLBACK-Physical] SR={current_sr:.2f} < {cfg.curriculum_rollback_threshold}. DETECTED COLLAPSE !!!")
                    current_m = max(current_m - cfg.m_increment * 2, cfg.initial_m)
                    current_p_init_dv = min(current_p_init_dv + cfg.p_init_dv_decrement * 2, cfg.initial_p_init_dv)
                    
                    print(f"    New Difficulty: m={current_m}, Fuel={current_p_init_dv}")
                    GlobalTrainManager.current_ring_ratio = env_cfg.start_ring_ratio
                    print(f"    Formation curriculum reset to Ring Ratio: {GlobalTrainManager.current_ring_ratio:.2f}")

                    for env in envs:
                        env.set_difficulty_parameters(m_distance=current_m, p_init_dv=current_p_init_dv)
                    
                    with open(progress_path, "a") as f:
                        f.write(f"{update},{global_step},{current_sr:.3f},{current_m},{current_p_init_dv},{GlobalTrainManager.current_ring_ratio:.3f},ROLLBACK_PHYSICAL\n")

                recent_episode_stats.clear()
                curriculum_stability_counter = 0
            
            else:
                if curriculum_stability_counter > 0:
                    print(f"--- [Maintain] SR={current_sr:.2f} is not stable enough. Resetting stability counter. ---")
                    curriculum_stability_counter = 0

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