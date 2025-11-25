
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

# --- 1. 超参数配置 ---
class TrainConfig:
    """训练超参数配置"""
    # --- Encoder 配置 ---
    use_encoder: bool = True  # 选择是否使用新的Encoder结构

    # --- 算法超参数 ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    anneal_ent: bool = True  # 是否对熵系数进行退火
    ent_anneal_start_frac: float = 0.3  # 从总训练步数的哪个百分比开始退火
    final_ent_coef: float = 0.001  # 熵系数最终衰减到的值
    sl_coef: float = 0.5  # 监督学习损失的权重
    lr: float = 3e-4
    sl_lr: float = 5e-4 # LSTM的学习率
    num_mini_batches: int = 4
    update_epochs: int = 5

    # --- 训练过程超参数 ---
    total_timesteps: int = 5_000_000
    num_steps: int = 2048
    num_envs: int = 1

    # --- 课程学习超参数 ---
    initial_episode_length: int = 3600 * 10
    curriculum_check_episodes: int = 50
    success_rate_threshold: float = 0.7
    initial_dist_cap: float = 60e3
    initial_p_init_dv: float = 200.0
    dist_cap_decrement: float = 1e3
    p_init_dv_decrement: float = 100.0
    min_dist_cap: float = 30e3
    min_p_init_dv: float = 200.0

    # --- 其他 ---
    debug_critic: bool = False # 是否打印Critic诊断信息
    resume_from_checkpoint: str = None # 从指定检查点恢复训练
    checkpoint_interval: int = 50 # 每隔N个update保存一次检查点
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"mpe_pomdp_lstm_{int(time.time())}"


# --- 2. Actor-Critic 网络 ---

# --- 辅助函数：正交初始化 ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, actor_obs_dim, critic_obs_dim, act_dim, env, env_cfg, train_cfg):
        super().__init__()
        self.use_encoder = train_cfg.use_encoder
        
        # --- 1. Encoder / Actor 主干 ---
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
            
            # 修改：Actor Head 使用正交初始化，增益为 0.01 (让初始动作接近 0)
            self.actor_head = layer_init(nn.Linear(encoder_output_dim, act_dim), std=0.01)
        else:
            # 修改：使用正交初始化构建 MLP
            self.actor_net = nn.Sequential(
                layer_init(nn.Linear(actor_obs_dim, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, 256)),
                nn.Tanh(),
                layer_init(nn.Linear(256, act_dim), std=0.01) # 最后一层增益 0.01
            )

        # --- 2. Critic 网络 (使用正交初始化) ---
        self.critic_net = nn.Sequential(
            layer_init(nn.Linear(critic_obs_dim, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0) # Critic 输出层增益 1.0
        )
        
        # 初始 LogStd 设置为 -0.5 (std ≈ 0.6)，比 0 (std=1) 更适合精细操作
        self.actor_logstd = nn.Parameter(torch.ones(1, act_dim) * -0.5)
        
        # --- 3. 动作缩放参数 ---
        pursuer_ids = [f'p_{i}' for i in range(env_cfg.num_p)]
        action_space = env.action_spaces[pursuer_ids[0]]
        # 注册为 buffer，这样 device 会自动管理，且不会被视为模型参数更新
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))

        # --- 4. LSTM (保持不变) ---
        if env_cfg.lstm_scheme == 1:
            from env.lstm import TrajectoryPredictor
            self.lstm = TrajectoryPredictor(input_dim=6, hidden_dim=128, output_dim=env_cfg.lstm_future_len * 3, num_layers=2)
        elif env_cfg.lstm_scheme in [2, 3]:
            from env.lstm_relative import RelativeTrajectoryPredictor
            self.lstm = RelativeTrajectoryPredictor(scheme=env_cfg.lstm_scheme, input_dim=6, hidden_dim=128, output_dim=env_cfg.lstm_future_len * 3, num_layers=2)

    def get_value(self, central_obs):
        return self.critic_net(central_obs)

    def get_action_and_value(self, obs, central_obs, action=None, deterministic=False):
        """
        新增 deterministic 参数：用于评估时只输出均值
        """
        # 1. 计算均值
        if self.use_encoder:
            h = self.encoder(obs)
            action_mean = self.actor_head(h)
        else:
            action_mean = self.actor_net(obs)

        # 2. 限制 LogStd 范围 (改进点：数值稳定性)
        # 将 logstd 限制在 [-20, 2] 之间，防止 std 过小导致 NaN 或 过大导致完全随机
        clipped_logstd = torch.clamp(self.actor_logstd, -20, 2)
        action_std = torch.exp(clipped_logstd).expand_as(action_mean)
        
        # 3. 构建分布
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                # 评估模式：直接使用均值，不再采样
                pre_tanh_action = action_mean
            else:
                # 训练模式：采样
                pre_tanh_action = probs.rsample()
                
            tanh_action = torch.tanh(pre_tanh_action)
            final_action = self.action_bias + self.action_scale * tanh_action
        else:
            # 训练更新阶段：从 buffer 中的实际 action 反推
            # 反归一化
            unscaled_action = (action - self.action_bias) / self.action_scale
            
            # 安全的 atanh (防止数值越界出现 NaN)
            # 这里的 clamp 极其重要，因为浮点数误差可能导致 unscaled_action 稍微超过 1
            pre_tanh_action = torch.atanh(torch.clamp(unscaled_action, -1.0 + 1e-6, 1.0 - 1e-6))
            
            final_action = action
            # 对于反推的情况，tanh_action 需要重新计算以便用于 correction
            tanh_action = torch.tanh(pre_tanh_action)

        # 4. 计算 Log Probability (包含 Tanh 修正)
        log_prob_pre_tanh = probs.log_prob(pre_tanh_action).sum(1)
        
        # 修正公式： log(p(y)) = log(p(x)) - sum(log(1 - tanh(x)^2))
        # 使用 1e-6 保证稳定性
        log_prob_correction = torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6).sum(1)
        
        log_prob = log_prob_pre_tanh - log_prob_correction
        
        entropy = probs.entropy().sum(1)
        value = self.critic_net(central_obs)
        
        return final_action, log_prob, entropy, value


# --- 3. Rollout Buffer ---
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

# --- 4. 训练主函数 ---
def train(cfg: TrainConfig, env_cfg: MPE_POMDP_EnvCfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    run_dir = Path(f"OrbitalGameEnv/runs/{cfg.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir))

    # 将本次运行的奖励参数保存到文件中
    params_path = run_dir / "reward_params.txt"
    with open(params_path, "w") as f:
        params_content = f"""--- Reward Parameters ---
lambert_reward_weight: {env_cfg.lambert_reward_weight}
reward_dist_weight: {env_cfg.reward_dist_weight}
reward_time_weight: {env_cfg.reward_time_weight}
reward_formation_weight: {env_cfg.reward_formation_weight}
reward_fuel_weight: {env_cfg.reward_fuel_weight}
reward_advantage_weight: {env_cfg.reward_advantage_weight}
capture_reward: {env_cfg.capture_reward}
reward_timeout_penalty: {env_cfg.reward_timeout_penalty}
reward_fuelout_penalty: {env_cfg.reward_fuelout_penalty}
"""
        f.write(params_content)
    print(f"Reward parameters saved to {params_path}")

    # --- 优雅关闭与检查点 ---
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

    
    current_episode_length = cfg.initial_episode_length
    current_dist_cap = cfg.initial_dist_cap
    current_p_init_dv = cfg.initial_p_init_dv
    env.set_difficulty_parameters(episode_length=current_episode_length, dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
    print(f"POMDP Mode: {env_cfg.use_partial_obs}, Obs Interval: {env_cfg.obs_interval}")
    print(f"任务时长固定: {current_episode_length}s, 初始捕获距离: {current_dist_cap}m, 初始燃料: {current_p_init_dv}m/s")

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

    # --- 检查点加载 ---
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
        # 加载课程学习状态
        current_episode_length = checkpoint.get('current_episode_length', cfg.initial_episode_length)
        current_dist_cap = checkpoint.get('current_dist_cap', cfg.initial_dist_cap)
        current_p_init_dv = checkpoint.get('current_p_init_dv', cfg.initial_p_init_dv)
        env.set_difficulty_parameters(episode_length=current_episode_length, dist_cap=current_dist_cap, p_init_dv=current_p_init_dv)
        print(f"Resumed at update {start_update}, global_step {global_step}")
        print(f"Resumed difficulty: Ep_Len={current_episode_length}, Dist_Cap={current_dist_cap}, Fuel={current_p_init_dv}")

    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    
    # 用于记录最近N个回合的统计数据
    recent_episode_stats = deque(maxlen=cfg.curriculum_check_episodes) 
    
    # 用于记录当前回合的数据
    current_episode_return = 0.0
    current_episode_length = 0
    current_episode_components = {}

    obs, infos = env.reset()
    
    for update in range(start_update, num_updates + 1):
        # --- 新增：熵系数退火逻辑 ---
        if cfg.anneal_ent:
            anneal_start_update = cfg.ent_anneal_start_frac * num_updates
            if update < anneal_start_update:
                current_ent_coef = cfg.ent_coef
            else:
                # 线性衰减
                progress = (update - anneal_start_update) / (num_updates - anneal_start_update)
                progress = min(1.0, progress)  # 确保 progress 不会超过 1
                current_ent_coef = cfg.ent_coef - progress * (cfg.ent_coef - cfg.final_ent_coef)
        else:
            current_ent_coef = cfg.ent_coef
        # --- 熵系数退火逻辑结束 ---

        start_time = time.time()
        
        agent.eval()
        for step in range(cfg.num_steps):

            global_step += 1
            current_episode_length += 1
            
            pursuer_obs_list = [torch.Tensor(obs[name]).to(cfg.device) for name in pursuer_ids]
            pursuer_obs_tensor = torch.stack(pursuer_obs_list)
            central_obs_tensor = torch.cat(pursuer_obs_list, dim=-1)

            with torch.no_grad():
                actions_tensor, log_prob, _, values = agent.get_action_and_value(pursuer_obs_tensor, central_obs_tensor.unsqueeze(0))
                values = values.flatten()

            evader_actions = {name: env.action_spaces[name].sample() for name in evader_ids}
            actions_to_step = {name: actions_tensor[i].cpu().numpy() for i, name in enumerate(pursuer_ids)}
            actions_to_step.update(evader_actions)

            next_obs, rewards, terminations, truncations, infos = env.step(actions_to_step)
            

            # 注意：这里我们累加的是所有追击者奖励的总和
            pursuer_rewards_sum = sum(rewards.get(name, 0) for name in pursuer_ids)
            current_episode_return += pursuer_rewards_sum

            # 如果开启诊断模式，则累加奖励分量
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
                    
                    print(f"G_Step:{global_step}, Ep_Done, Reason:{final_info.get('termination_reason', 'Unknown')}, Ep_Return:{current_episode_return:.2f}, Ep_Len:{current_episode_length}")
                    writer.add_scalar("charts/episodic_return", current_episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", current_episode_length, global_step)

                    # 如果开启诊断模式，则打印奖励分量
                    if cfg.debug_critic and current_episode_components:
                        print(f"--- Episode End Breakdown ---")
                        for key, value in sorted(current_episode_components.items()):
                            print(f"  Sum {key}: {value:.2f}")
                        print("-----------------------------")

                # Reset episode-specific trackers
                obs, infos = env.reset()
                current_episode_return = 0.0
                current_episode_length = 0
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
        # 初始化用于记录整个 update 周期内的 loss
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

                # 累加 loss
                avg_v_loss += v_loss.item()
                avg_pg_loss += pg_loss.item()
                avg_sl_loss += sl_loss.item()
                avg_entropy_loss += entropy_loss.item()
                num_minibatches_processed += 1

        # 计算 Explained Variance
        y_pred, y_true = buffer.values.cpu().numpy(), buffer.returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if cfg.debug_critic:
            print("\n" + "-" * 30)
            print(f"--- Critic Diagnosis (Update {update}) ---")
            print(f"Values (Pred) | Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}, Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
            print(f"Returns (True)| Mean: {y_true.mean():.4f}, Std: {y_true.std():.4f}, Range: [{y_true.min():.4f}, {y_true.max():.4f}]")
            print(f"Diff (L2 Loss)| Mean Abs Err: {np.abs(y_true - y_pred).mean():.4f}")
            # 随机抽 3 个样本看看
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
        
        # 记录平均 loss
        writer.add_scalar("losses/value_loss", avg_v_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/policy_loss", avg_pg_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/entropy_loss", avg_entropy_loss / num_minibatches_processed, global_step)
        writer.add_scalar("losses/sl_loss", avg_sl_loss / num_minibatches_processed, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("info/explained_variance", explained_var, global_step)
        writer.add_scalar("info/ent_coef", current_ent_coef, global_step)

        # --- 检查点保存 ---
        if (update % cfg.checkpoint_interval == 0) or shutdown_requested:
            checkpoint_path = checkpoint_dir / f"update_{update}.pt"
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            torch.save({
                'update': update,
                'global_step': global_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lstm_optimizer_state_dict': lstm_optimizer.state_dict(),
                'current_episode_length': current_episode_length,
                'current_dist_cap': current_dist_cap,
                'current_p_init_dv': current_p_init_dv,
            }, checkpoint_path)
            print("Checkpoint saved.")
        
        if shutdown_requested:
            print("Graceful shutdown complete.")
            break

        # 课程学习与统计日志
        if len(recent_episode_stats) >= 2:
            # 确保我们有足够的数据来进行有意义的统计
            # 这里可以根据需要调整检查的最小 episode 数量
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
                            recent_episode_stats.clear()

    env.close()
    writer.close()
    print("训练完成!")



if __name__ == "__main__":
    print("使用默认设置参数进行训练")
    train_cfg = TrainConfig()
    env_cfg = MPE_POMDP_EnvCfg()
    train(train_cfg, env_cfg)
