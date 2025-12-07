import sys
import os

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import signal
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import random

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from env.encoder import AttentionBasedEncoder

class TrainConfig:
    """训练超参数配置"""
    use_encoder: bool = True  # 是否在Actor网络中使用Attention Encoder
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    anneal_ent: bool = True  # 是否对熵系数进行退火
    ent_anneal_start_frac: float = 0.3  # 从总训练步数的哪个百分比开始退火
    final_ent_coef: float = 0.0001  # 熵系数最终衰减到的值
    sl_coef: float = 0.5  # 监督学习损失的权重
    lr: float = 3e-4
    sl_lr: float = 5e-4 # LSTM的学习率
    num_mini_batches: int = 4
    update_epochs: int = 5
    total_timesteps: int = 5_000_000
    num_steps: int = 2048
    num_envs: int = 1
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.78
    initial_dist_cap: float = 60e3
    initial_p_init_dv: float = 200.0
    dist_cap_decrement: float = 1e3
    p_init_dv_decrement: float = 20.0
    min_dist_cap: float = 30e3
    min_p_init_dv: float = 200.0
    debug_critic: bool = False 
    resume_from_checkpoint: str = None # 从指定检查点恢复训练
    checkpoint_interval: int = 50 # 每隔N个update保存一次检查点
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"mpe_pomdp_lstm_{int(time.time())}"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, actor_obs_dim, critic_obs_dim, act_dim, env, env_cfg, train_cfg):
        super().__init__()
        self.use_encoder = train_cfg.use_encoder
        if self.use_encoder:
            self_dim = 7
            lstm_pred_dim = env_cfg.lstm_future_len * 3 * env_cfg.num_e
            other_dim = 7 * (env_cfg.num_p - 1)
            ob_dim = 0
            self.encoder = AttentionBasedEncoder(
                self_dim=self_dim,
                other_dim=other_dim,
                ob_dim=ob_dim,
                lstm_pred_dim=lstm_pred_dim,
                embed_dim=128,
                nhead=4
            )
            encoder_output_dim = 256
            self.actor_head = layer_init(nn.Linear(encoder_output_dim, act_dim), std=0.01)
        else:
            self.actor_net = nn.Sequential(
                layer_init(nn.Linear(actor_obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, act_dim), std=0.01)
            )
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0)
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
        action_space = env.action_spaces[pursuer_ids[0]]
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        if env_cfg.lstm_scheme == 1:
            from env.lstm import TrajectoryPredictor
            self.lstm = TrajectoryPredictor(input_dim=6, hidden_dim=128, output_dim=env_cfg.lstm_future_len * 3, num_layers=2)
        elif env_cfg.lstm_scheme in [2, 3]:
            from env.lstm_relative import RelativeTrajectoryPredictor
            self.lstm = RelativeTrajectoryPredictor(scheme=env_cfg.lstm_scheme, input_dim=6, hidden_dim=128, output_dim=env_cfg.lstm_future_len * 3, num_layers=2)

    def get_value(self, central_obs):
        return self.critic_net(central_obs)

    def get_action_and_value(self, obs, central_obs, action=None, deterministic=False):
        if self.use_encoder:
            h = self.encoder(obs)
            action_mean = self.actor_head(h)
        else:
            action_mean = self.actor_net(obs)
        clipped_logstd = torch.clamp(self.actor_logstd, -2, 1)
        action_std = torch.exp(clipped_logstd).expand_as(action_mean)
        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            if deterministic:
                pre_tanh_action = action_mean
            else:
                pre_tanh_action = probs.rsample()
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
        value = self.critic_net(central_obs)
        return final_action, log_prob, entropy, value

class CentralizedRolloutBuffer:
    def __init__(self, num_steps, num_agents, actor_obs_dim, critic_obs_dim, act_dim, device, lstm_cfg):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device
        self.obs = torch.zeros((num_steps, num_agents, actor_obs_dim)).to(device)
        self.central_obs = torch.zeros((num_steps, critic_obs_dim)).to(device)
        self.actions = torch.zeros((num_steps, num_agents, act_dim)).to(device)
        self.logprobs = torch.zeros((num_steps, num_agents)).to(device)
        self.rewards = torch.zeros((num_steps, num_agents)).to(device)
        self.dones = torch.zeros((num_steps, num_agents)).to(device)
        self.values = torch.zeros((num_steps, num_agents)).to(device)
        self.sl_history_inputs = torch.zeros((num_steps, num_agents, lstm_cfg.lstm_history_len, 6)).to(device)
        self.sl_future_gts = torch.zeros((num_steps, num_agents, lstm_cfg.lstm_future_len, 3)).to(device)
        self.step = 0

    def add(self, obs, central_obs, actions, logprobs, rewards, dones, values, infos, pursuer_ids):
        self.obs[self.step] = obs
        self.central_obs[self.step] = central_obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values
        for i, agent_id in enumerate(pursuer_ids):
            if agent_id in infos and f'sl_history_input_e_0' in infos[agent_id]:
                self.sl_history_inputs[self.step, i] = torch.from_numpy(infos[agent_id][f'sl_history_input_e_0']).to(self.device)
                self.sl_future_gts[self.step, i] = torch.from_numpy(infos[agent_id][f'sl_future_ground_truth_e_0']).to(self.device)
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

    def get(self, batch_size, mini_batch_size):
        num_samples = batch_size
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, mini_batch_size):
            end = start + mini_batch_size
            batch_indices = indices[start:end]
            step_indices = batch_indices // self.num_agents
            agent_indices = batch_indices % self.num_agents
            yield (
                self.obs[step_indices, agent_indices],
                self.central_obs[step_indices],
                self.actions[step_indices, agent_indices],
                self.logprobs[step_indices, agent_indices],
                self.advantages[step_indices, agent_indices],
                self.returns[step_indices, agent_indices],
                self.sl_history_inputs[step_indices, agent_indices],
                self.sl_future_gts[step_indices, agent_indices],
            )

def train(cfg: TrainConfig, env_cfg: MPE_POMDP_EnvCfg, all_params: dict):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # 为保证路径稳健，基于脚本自身位置构建路径
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    run_dir = base_dir / "runs" / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    # 定义日志文件路径
    params_path = run_dir / "all_params.txt"
    progress_path = run_dir / "curriculum_progress_log.txt"

    with open(params_path, "w", encoding="utf-8") as f:
        f.write("--- All Run Parameters ---")
        for key, value in sorted(all_params.items()):
            f.write(f"{key}: {value}\n")
    print(f"All run parameters saved to {params_path}")

    shutdown_requested = False
    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            print("\nCtrl+C received! Finishing current update and saving checkpoint...")
            shutdown_requested = True
        else:
            print("\nSecond Ctrl+C received! Forcing exit.")
            sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    writer.add_text("hyperparameters", f"<pre>{vars(cfg)}</pre>")

    env = MPE_POMDP_Env(env_cfg)
    
    curriculum_max_episode_length = cfg.initial_episode_length
    current_dist_cap = cfg.initial_dist_cap
    current_p_init_dv = cfg.initial_p_init_dv
    env.set_difficulty_parameters(episode_length=curriculum_max_episode_length, dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
    print(f"POMDP Mode: {env_cfg.use_partial_obs}, Obs Interval: {env_cfg.obs_interval}")
    print(f"任务时长固定: {curriculum_max_episode_length}s, 初始捕获距离: {current_dist_cap}m, 初始燃料: {current_p_init_dv}m/s")

    pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
    evader_ids = [f'e_{i}' for i in range(env_cfg.num_e)]
    
    actor_obs_dim = env.observation_spaces[pursuer_ids[0]].shape[0]
    critic_obs_dim = sum(env.observation_spaces[p_id].shape[0] for p_id in pursuer_ids)
    act_dim = env.action_spaces[pursuer_ids[0]].shape[0]

    agent = ActorCritic(actor_obs_dim, critic_obs_dim, act_dim, env, env_cfg, cfg).to(cfg.device)
    
    if cfg.use_encoder:
        ac_params = list(agent.encoder.parameters()) + list(agent.actor_head.parameters()) + list(agent.critic_net.parameters()) + [agent.actor_logstd]
    else:
        ac_params = list(agent.actor_net.parameters()) + list(agent.critic_net.parameters()) + [agent.actor_logstd]

    optimizer = torch.optim.Adam(ac_params, lr=cfg.lr, eps=1e-5)
    lstm_optimizer = torch.optim.Adam(agent.lstm.parameters(), lr=cfg.sl_lr, eps=1e-5)
    sl_loss_fn = nn.SmoothL1Loss()

    env.set_policy_lstm(agent.lstm)

    buffer = CentralizedRolloutBuffer(cfg.num_steps, env_cfg.num_p, actor_obs_dim, critic_obs_dim, act_dim, cfg.device, env_cfg)

    start_update = 1
    global_step = 0
    if cfg.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location=cfg.device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lstm_optimizer.load_state_dict(checkpoint['lstm_optimizer_state_dict'])
        start_update = checkpoint['update'] + 1
        global_step = checkpoint['global_step']
        curriculum_max_episode_length = checkpoint.get('current_episode_length', cfg.initial_episode_length)
        current_dist_cap = checkpoint.get('current_dist_cap', cfg.initial_dist_cap)
        current_p_init_dv = checkpoint.get('current_p_init_dv', cfg.initial_p_init_dv)
        env.set_difficulty_parameters(episode_length=curriculum_max_episode_length, dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
        print(f"Resumed at update {start_update}, global_step {global_step}")
        print(f"Resumed difficulty: Ep_Len={curriculum_max_episode_length}, Dist_Cap={current_dist_cap}, Fuel={current_p_init_dv}")

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)

    # --- 学习率退火参数 ---
    anneal_lr_start_update = 500  # 从第500个Update开始退火
    final_lr_fraction = 0.1       # 最终学习率降为初始值的10%
    
    recent_episode_stats = deque(maxlen=cfg.curriculum_check_episodes) 
    
    current_episode_return = 0.0
    ep_len_counter = 0
    current_episode_components = {}

    obs, infos = env.reset() 
    last_sma_perturb_km = env.current_sma_perturb_km # 跟踪SMA扰动值以记录变化
    
    for update in range(start_update, num_updates + 1):
        # --- [新] SMA扰动课程学习逻辑（从env中移入） ---
        start = env_cfg.sma_perturb_start_update
        end = env_cfg.sma_perturb_end_update
        max_perturb = env_cfg.sma_perturb_km_max

        if update < start:
            new_perturb = 0.0
        elif update >= end:
            new_perturb = max_perturb
        else:
            progress = (update - start) / (end - start)
            new_perturb = progress * max_perturb
        
        env.set_sma_perturb(new_perturb)

        # --- 记录SMA扰动课程学习进度 ---
        if env.current_sma_perturb_km > last_sma_perturb_km:
            print(f"*** 课程学习: SMA扰动提升 -> 新扰动: {env.current_sma_perturb_km:.2f} km ***")
            file_exists = os.path.exists(progress_path)
            with open(progress_path, "a", encoding="utf-8") as f:
                if not file_exists:
                    f.write("--- Curriculum Progress Log ---\n")
                    f.write("Abbreviations:\n")
                    f.write("  U: Update\n")
                    f.write("  GS: Global Step\n")
                    f.write("  Trig: Trigger Type (SR=Success Rate, SMA=SMA Perturbation)\n")
                    f.write("  SR_trig: Trigger Success Rate\n")
                    f.write("  DistCap: New Capture Distance Cap (m)\n")
                    f.write("  Fuel: New Pursuer Initial Fuel (m/s)\n")
                    f.write("  SMA_km: New SMA Perturbation (km)\n\n")
                f.write("--------------------------------------------------\n")
                f.write(f"U: {update} | GS: {global_step}\n")
                f.write(f"Trig: SMA\n")
                f.write(f"SMA_km: {env.current_sma_perturb_km:.2f}\n")
            print(f"Curriculum progress logged to {progress_path}")
            last_sma_perturb_km = env.current_sma_perturb_km # 更新跟踪值

        # --- 学习率退火逻辑 ---
        if update < anneal_lr_start_update:
            # 阶段一: 保持恒定学习率
            current_lr = cfg.lr
        else:
            # 阶段二: 线性衰减
            decay_steps = num_updates - anneal_lr_start_update
            progress = (update - anneal_lr_start_update) / decay_steps
            progress = min(1.0, max(0.0, progress))
            
            lr_decay_factor = 1.0 - (1.0 - final_lr_fraction) * progress
            current_lr = cfg.lr * lr_decay_factor

        # 应用新学习率到优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        # 记录到 TensorBoard
        writer.add_scalar("charts/learning_rate", current_lr, global_step)

        # --- 熵系数退火逻辑 ---
        if cfg.anneal_ent:
            anneal_start_update = cfg.ent_anneal_start_frac * num_updates
            if update < anneal_start_update:
                current_ent_coef = cfg.ent_coef
            else:
                progress = (update - anneal_start_update) / (num_updates - anneal_start_update)
                progress = min(1.0, progress)
                current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef

        start_time = time.time()
        
        agent.eval()
        for step in range(cfg.num_steps):

            global_step += 1
            ep_len_counter += 1
            
            pursuer_obs_list = [torch.Tensor(obs[name]).to(cfg.device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            central_obs_tensor = torch.cat(pursuer_obs_list, dim=-1)

            with torch.no_grad():
                actions_tensor, log_prob, _, values = agent.get_action_and_value(pursuer_obs_tensor, central_obs_tensor.unsqueeze(0))
                values = values.flatten()

            # 获取逃逸者动作（支持 0:Drift, 1:Random, 2:APF）
            evader_actions = env.get_evader_actions()
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
            
            pursuer_rewards_sum = sum(rewards.get(name, 0) for name in pursuer_ids)
            current_episode_return += pursuer_rewards_sum

            if cfg.debug_critic:
                for name in pursuer_ids:
                    if name in infos and 'reward_components' in infos[name]:
                        for key, value in infos[name]['reward_components'].items():
                            current_episode_components[key] = current_episode_components.get(key, 0.0) + value

            pursuer_rewards_tensor = torch.tensor([rewards[name] for name in pursuer_ids]).to(cfg.device)
            pursuer_dones = torch.tensor([terminations.get(name, False) or truncations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            
            buffer.add(pursuer_obs_tensor, central_obs_tensor, actions_tensor, log_prob, pursuer_rewards_tensor, pursuer_dones, values, infos, pursuer_ids)
            
            obs = next_obs
            
            if any(terminations.values()) or any(truncations.values()):
                final_info = next(iter(infos.values()), None)
                if final_info:
                    stats = final_info.get('episode_statistics', {})
                    if stats: recent_episode_stats.append(stats)
                    
                    print(f"G_Step:{global_step}, Ep_Done, Reason:{final_info.get('termination_reason', 'Unknown')}, Ep_Return:{current_episode_return:.2f}, Ep_Len:{ep_len_counter}")
                    writer.add_scalar("charts/episodic_return", current_episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", ep_len_counter, global_step)

                    if cfg.debug_critic and current_episode_components:
                        print(f"--- Episode End Breakdown ---")
                        for key, value in sorted(current_episode_components.items()):
                            print(f"  Sum {key}: {value:.2f}")
                        print("-----------------------------")

                obs, infos = env.reset() 
                current_episode_return = 0.0
                ep_len_counter = 0
                current_episode_components = {}

        with torch.no_grad():
            next_pursuer_obs = [torch.Tensor(obs[name]).to(cfg.device) for name in pursuer_ids if name in obs]
            if next_pursuer_obs:
                next_central_obs = torch.cat(next_pursuer_obs, dim=-1).unsqueeze(0)
                next_value = agent.get_value(next_central_obs).reshape(1, -1)
            else: 
                next_value = torch.zeros(1, 1).to(cfg.device)
            next_done = torch.tensor([terminations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            buffer.compute_returns(next_value, next_done, cfg.gamma, cfg.gae_lambda)

        batch_size = cfg.num_steps * env_cfg.num_p
        mini_batch_size = batch_size // cfg.num_mini_batches
        
        agent.train()
        avg_v_loss, avg_pg_loss, avg_sl_loss, avg_entropy_loss = 0, 0, 0, 0
        num_minibatches_processed = 0

        for epoch in range(cfg.update_epochs):
            for b_obs, b_central_obs, b_actions, b_logprobs, b_advantages, b_returns, b_sl_hist, b_sl_gt in buffer.get(batch_size, mini_batch_size):
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs, b_central_obs, b_actions)
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()
                logratio = new_logprob - b_logprobs
                ratio = logratio.exp()
                adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                rl_loss = pg_loss - current_ent_coef * entropy_loss + v_loss * cfg.vf_coef

                sl_mask = b_sl_hist.abs().sum(dim=-1) > 1e-6
                sl_pred = agent.lstm(b_sl_hist, mask=sl_mask)
                sl_loss = sl_loss_fn(sl_pred, b_sl_gt.view(b_sl_gt.shape[0], -1))

                optimizer.zero_grad()
                rl_loss.backward()
                if cfg.use_encoder:
                    grad_params = list(agent.encoder.parameters()) + list(agent.actor_head.parameters()) + list(agent.critic_net.parameters())
                else:
                    grad_params = list(agent.actor_net.parameters()) + list(agent.critic_net.parameters())
                
                if cfg.debug_critic:
                    critic_params = list(agent.critic_net.parameters())
                    if critic_params:
                        critic_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in critic_params if p.grad is not None]), 2)
                        writer.add_scalar("info/critic_grad_norm", critic_grad_norm, global_step)

                nn.utils.clip_grad_norm_(grad_params, 0.5)
                optimizer.step()

                lstm_optimizer.zero_grad()
                weighted_sl_loss = sl_loss * cfg.sl_coef
                weighted_sl_loss.backward()
                nn.utils.clip_grad_norm_(agent.lstm.parameters(), 0.5)
                lstm_optimizer.step()

                avg_v_loss += v_loss.item()
                avg_pg_loss += pg_loss.item()
                avg_sl_loss += sl_loss.item()
                avg_entropy_loss += entropy_loss.item()
                num_minibatches_processed += 1

        y_pred, y_true = buffer.values.cpu().numpy(), buffer.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if cfg.debug_critic:
            print("\n" + "-" * 30)
            print(f"--- Critic Diagnosis (Update {update}) ---")
            print(f"Values (Pred) | Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}, Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
            print(f"Returns (True)| Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f}, Range: [{y_true.min():.4f}, {y_true.max():.4f}]")
            print(f"Diff (L2 Loss)| Mean Abs Err: {np.abs(y_true - y_pred).mean():.4f}")
            if len(y_pred.flatten()) > 3:
                y_pred_flat = y_pred.flatten()
                y_true_flat = y_true.flatten()
                indices = np.random.choice(len(y_pred_flat), 3, replace=False)
                print("  --- Samples ---")
                for i in indices:
                    pred_val = y_pred_flat[i]
                    true_val = y_true_flat[i]
                    print(f"  Sample {i}: Pred_Value: {pred_val:.4f} vs True_Return: {true_val:.4f} | Error: {pred_val-true_val:.4f}")
            print("-" * 30)

        sps = int(cfg.num_steps / (time.time() - start_time))
        print(f"Update: {update}/{num_updates}, SPS: {sps}, Explained Variance: {explained_var:.2f}")
        
        writer.add_scalar("losses/value_loss", avg_v_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/policy_loss", avg_pg_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/entropy_loss", avg_entropy_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/sl_loss", avg_sl_loss / num_minibatches_processed, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("info/explained_variance", explained_var, global_step)
        writer.add_scalar("info/ent_coef", current_ent_coef, global_step)

        if (update % cfg.checkpoint_interval == 0) or shutdown_requested:
            checkpoint_path = checkpoint_dir / f"update_{update}.pt"
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            torch.save({
                'update': update,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lstm_optimizer_state_dict': lstm_optimizer.state_dict(),
                'current_episode_length': curriculum_max_episode_length,
                'current_dist_cap': current_dist_cap,
                'current_p_init_dv': current_p_init_dv,
            }, checkpoint_path)
            print("Checkpoint saved.")
        
        if shutdown_requested:
            print("Graceful shutdown complete.")
            break

        if len(recent_episode_stats) >= 2:
            if update > 10 and len(recent_episode_stats) > 10:
                total_eps = recent_episode_stats[-1]['total_episodes'] - recent_episode_stats[0]['total_episodes']
                if total_eps > 0:
                    successes = recent_episode_stats[-1]['success_count'] - recent_episode_stats[0]['success_count']
                    timeouts = recent_episode_stats[-1]['timeout_count'] - recent_episode_stats[0]['timeout_count']
                    fuelouts = recent_episode_stats[-1]['fuelout_count'] - recent_episode_stats[0]['fuelout_count']
                    
                    current_success_rate = successes / total_eps
                    current_timeout_rate = timeouts / total_eps
                    current_fuelout_rate = fuelouts / total_eps

                    writer.add_scalar("charts/success_rate", current_success_rate, global_step)
                    writer.add_scalar("charts/timeout_rate", current_timeout_rate, global_step)
                    writer.add_scalar("charts/fuelout_rate", current_fuelout_rate, global_step)

                    if current_success_rate >= cfg.success_rate_threshold:
                        changed = False
                        if current_dist_cap > cfg.min_dist_cap:
                            current_dist_cap -= cfg.dist_cap_decrement
                            changed = True
                        if current_p_init_dv > cfg.min_p_init_dv:
                            current_p_init_dv -= cfg.p_init_dv_decrement
                            changed = True
                        if changed:
                            env.set_difficulty_parameters(dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
                            print(f"*** 课程学习: 成功率 {current_success_rate:.2f}, 提升难度 -> 新捕获距离:{current_dist_cap}m, 新燃料:{current_p_init_dv}m/s ***")
                            
                            # --- 记录课程学习进度 ---
                            file_exists = os.path.exists(progress_path)
                            with open(progress_path, "a", encoding="utf-8") as f:
                                if not file_exists:
                                    f.write("--- Curriculum Progress Log ---\n")
                                    f.write("Abbreviations:\n")
                                    f.write("  U: Update\n")
                                    f.write("  GS: Global Step\n")
                                    f.write("  Trig: Trigger Type (SR=Success Rate, SMA=SMA Perturbation)\n")
                                    f.write("  SR_trig: Trigger Success Rate\n")
                                    f.write("  DistCap: New Capture Distance Cap (m)\n")
                                    f.write("  Fuel: New Pursuer Initial Fuel (m/s)\n")
                                    f.write("  SMA_km: New SMA Perturbation (km)\n\n")
                                f.write("--------------------------------------------------\n")
                                f.write(f"U: {update} | GS: {global_step}\n")
                                f.write(f"Trig: SR\n")
                                f.write(f"SR_trig: {current_success_rate:.2f}\n")
                                f.write(f"DistCap: {current_dist_cap}\n")
                                f.write(f"Fuel: {current_p_init_dv}\n")
                            print(f"Curriculum progress logged to {progress_path}")

                            recent_episode_stats.clear()

    env.close()
    writer.close()
    print("训练完成!")


if __name__ == "__main__":
    print("使用默认设置参数进行训练")
    train_cfg = TrainConfig()
    env_cfg = MPE_POMDP_EnvCfg()
    train(train_cfg, env_cfg)
