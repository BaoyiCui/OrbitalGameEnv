
import sys
import os

# 将项目根目录添加到Python路径中
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import random

from env.mpe_pomdp_env import MPE_POMDP_Env, MPE_POMDP_EnvCfg
from env.lstm import TrajectoryPredictor
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
    initial_p_init_dv: float = 500.0
    dist_cap_decrement: float = 1e3
    p_init_dv_decrement: float = 10.0
    min_dist_cap: float = 30e3
    min_p_init_dv: float = 200.0

    # --- 其他 ---
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_name: str = f"mpe_pomdp_lstm_{int(time.time())}"


# --- 2. Actor-Critic 网络 ---
class ActorCritic(nn.Module):
    def __init__(self, actor_obs_dim, critic_obs_dim, act_dim, env_cfg, train_cfg):
        super().__init__()
        self.use_encoder = train_cfg.use_encoder
        
        if self.use_encoder:
            # --- 使用注意力编码器 ---
            self_dim = 6  # 自身状态维度
            lstm_pred_dim = env_cfg.lstm_future_len * 3 * env_cfg.num_e
            other_dim = 3 * (env_cfg.num_p - 1)
            ob_dim = 0 # 当前环境没有障碍物, 设置为0
            
            self.encoder = AttentionBasedEncoder(
                self_dim=self_dim,
                other_dim=other_dim,
                ob_dim=ob_dim,
                lstm_pred_dim=lstm_pred_dim,
                embed_dim=128,
                nhead=4
            )
            encoder_output_dim = 256 # Encoder的输出维度
            
            self.actor_head = nn.Linear(encoder_output_dim, act_dim)

        else:
            # --- 不使用Encoder，维持原有简单MLP结构 ---
            self.actor_net = nn.Sequential(
                nn.Linear(actor_obs_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, act_dim)
            )

        # --- Critic网络 ---
        self.critic_net = nn.Sequential(
            nn.Linear(critic_obs_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        
        # --- 轨迹预测网络 ---
        self.lstm = TrajectoryPredictor(
            input_dim=6, 
            hidden_dim=128, 
            output_dim=env_cfg.lstm_future_len * 3, 
            num_layers=2
        )

    def get_value(self, central_obs):
        return self.critic_net(central_obs)

    def get_action_and_value(self, obs, central_obs, action=None):
        if self.use_encoder:
            h = self.encoder(obs)
            action_mean = self.actor_head(h)
        else:
            action_mean = self.actor_net(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
            
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        value = self.critic_net(central_obs)
        
        return action, log_prob, entropy, value


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

    writer = SummaryWriter(f"runs/{cfg.run_name}")
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

    agent = ActorCritic(actor_obs_dim, critic_obs_dim, act_dim, env_cfg, cfg).to(cfg.device)
    
    if cfg.use_encoder:
        ac_params = list(agent.encoder.parameters()) + list(agent.actor_head.parameters()) + list(agent.critic_net.parameters()) + [agent.actor_logstd]
    else:
        ac_params = list(agent.actor_net.parameters()) + list(agent.critic_net.parameters()) + [agent.actor_logstd]

    optimizer = torch.optim.Adam(ac_params, lr=cfg.lr, eps=1e-5)
    lstm_optimizer = torch.optim.Adam(agent.lstm.parameters(), lr=cfg.sl_lr, eps=1e-5)
    sl_loss_fn = nn.MSELoss()

    env.set_policy_lstm(agent.lstm)

    buffer = CentralizedRolloutBuffer(cfg.num_steps, env_cfg.num_p, actor_obs_dim, critic_obs_dim, act_dim, cfg.device, env_cfg)

    global_step = 0
    num_updates = cfg.total_timesteps // (cfg.num_steps * cfg.num_envs)
    episode_stats = deque(maxlen=cfg.curriculum_check_episodes)

    obs, infos = env.reset()
    episode_lambert_reward = 0.0
    
    for update in range(1, num_updates + 1):
        start_time = time.time()
        
        agent.eval()
        for step in range(cfg.num_steps):
            global_step += 1
            
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
            
            pursuer_rewards = torch.tensor([rewards[name] for name in pursuer_ids]).to(cfg.device)
            pursuer_dones = torch.tensor([terminations.get(name, False) or truncations.get(name, False) for name in pursuer_ids]).to(cfg.device)
            
            buffer.add(pursuer_obs_tensor, central_obs_tensor, actions_tensor, log_prob, pursuer_rewards, pursuer_dones, values, infos, pursuer_ids)
            
            obs = next_obs
            
            if any(terminations.values()) or any(truncations.values()):
                final_info = next(iter(infos.values()), None)
                if final_info:
                    stats = final_info.get('episode_statistics', {})
                    if stats: episode_stats.append(stats)
                    pursuer_total_reward = sum(rewards.get(name, 0) for name in pursuer_ids)
                    print(f"G_Step:{global_step}, Ep_Done, Reason:{final_info.get('termination_reason', 'Unknown')}, Pursuer_Reward:{pursuer_total_reward:.2f}")
                    writer.add_scalar("charts/episode_reward", pursuer_total_reward, global_step)
                obs, infos = env.reset()
                episode_lambert_reward = 0.0

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
                rl_loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                sl_mask = b_sl_hist.abs().sum(dim=-1) > 1e-6
                sl_pred = agent.lstm(b_sl_hist, mask=sl_mask)
                sl_loss = sl_loss_fn(sl_pred, b_sl_gt.view(b_sl_gt.shape[0], -1))

                optimizer.zero_grad()
                rl_loss.backward()
                if cfg.use_encoder:
                    grad_params = list(agent.encoder.parameters()) + list(agent.actor_head.parameters()) + list(agent.critic_net.parameters())
                else:
                    grad_params = list(agent.actor_net.parameters()) + list(agent.critic_net.parameters())
                nn.utils.clip_grad_norm_(grad_params, 0.5)
                optimizer.step()

                lstm_optimizer.zero_grad()
                weighted_sl_loss = sl_loss * cfg.sl_coef
                weighted_sl_loss.backward()
                nn.utils.clip_grad_norm_(agent.lstm.parameters(), 0.5)
                lstm_optimizer.step()

        sps = int(cfg.num_steps / (time.time() - start_time))
        print(f"Update: {update}/{num_updates}, SPS: {sps}")
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/sl_loss", sl_loss.item(), global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        if update > 0 and update % (cfg.curriculum_check_episodes // (cfg.num_steps // 200)) == 0:
            if not episode_stats: continue
            total_eps = episode_stats[-1]['total_episodes'] - episode_stats[0]['total_episodes']
            if total_eps > 0:
                successes = episode_stats[-1]['success_count'] - episode_stats[0]['success_count']
                current_success_rate = successes / total_eps
                writer.add_scalar("curriculum/success_rate", current_success_rate, global_step)
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
                        episode_stats.clear()

    env.close()
    writer.close()
    print("训练完成!")



if __name__ == "__main__":
    print("Running training with default configurations.")
    train_cfg = TrainConfig()
    env_cfg = MPE_POMDP_EnvCfg()
    train(train_cfg, env_cfg)
