#此处是部分可观的环境，对应的训练脚本是train_pomdp.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces
from collections import deque
import torch
import datetime

# 导入父类环境和配置
from .mpe_env import MPEEnv, MPEEnvCfg
# 导入LSTM模型
from .lstm import TrajectoryPredictor

# 导入新的lambert求解器
try:
    from lambert_solver import solve_lambert
except ImportError:
    print("\033[93mWarning: C++ Lambert solver not found. Lambert-based rewards will be disabled.\033[0m")
    solve_lambert = None

@dataclass
class MPE_POMDP_EnvCfg(MPEEnvCfg):
    """继承并扩展MPE配置，加入POMDP和LSTM的特定参数"""
    use_partial_obs: bool = True
    obs_interval: int = 2
    lstm_history_len: int = 20
    lstm_future_len: int = 10

    # Lambert奖励配置 
    use_lambert_reward: bool = False
    lambert_reward_weight: float = 5
    lambert_transfer_time: float = 36000
    mu: float = 3.986004418e14

class MPE_POMDP_Env(MPEEnv):
    """
    继承自MPEEnv，通过覆盖和添加方法来实现部分可观测性功能。
    """
    def __init__(self, config: MPE_POMDP_EnvCfg = MPE_POMDP_EnvCfg()):
        # 首先调用父类的构造函数，完成大部分初始化
        super().__init__(config)
        self._config: MPE_POMDP_EnvCfg = config

        self.lstm_model = None
        self.device = None # 新增设备属性
        if self._config.use_partial_obs:
            # 初始化POMDP相关的状态变量
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.lstm_history_len) for i in range(self._config.num_e)}
            self.pursuer_predictions = {f'e_{i}': torch.zeros(self._config.lstm_future_len * 3) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}

            # 重新定义观测空间以适应LSTM的输出
            self.observation_spaces = {}
            for a in self.possible_agents:
                if a.startswith('p_'):
                    self_obs_dim = 6
                    evader_obs_dim = self._config.lstm_future_len * 3 * self._config.num_e
                    other_pursuers_obs_dim = 3 * (self._config.num_p - 1)
                    obs_shape = (self_obs_dim + evader_obs_dim + other_pursuers_obs_dim,)
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
                else:
                    self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

    def set_policy_lstm(self, lstm_model: TrajectoryPredictor):
        self.lstm_model = lstm_model
        try:
            self.device = next(self.lstm_model.parameters()).device
            print(f"LSTM model has been set in the POMDP environment on device: {self.device}")
        except StopIteration:
            print("Warning: LSTM model has no parameters. Cannot determine device.")

    def reset(self, seed=None, options=None):
        observations, infos = super().reset(seed, options)

        if self._config.use_partial_obs:
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}
            self.pursuer_predictions = {f'e_{i}': torch.zeros(self._config.lstm_future_len * 3, device=self.device) for i in range(self._config.num_e)}
            for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
                self.evader_history_buffers[evader_id].clear()
                if evader_id in self.states:
                    initial_state = self.states[evader_id]
                    for _ in range(self._config.lstm_history_len):
                        self.evader_history_buffers[evader_id].append(initial_state)
            
            self._update_predictions_and_prepare_sl_data()
            observations = self._get_observations()

        return observations, infos

    def step(self, actions):
        if self._config.use_partial_obs:
            self._update_predictions_and_prepare_sl_data()
        
        # 动作应用和状态传播 
        for a in self.agents:
            if a.startswith('p_'): dv_step = self._config.p_dv_step
            else: dv_step = self._config.e_dv_step
            action = actions.get(a, np.zeros(3))
            if np.linalg.norm(action) > dv_step: action = action / np.linalg.norm(action) * dv_step
            if np.linalg.norm(action) > self.remain_Dvs[a]: action = action / np.linalg.norm(action) * self.remain_Dvs[a]
            self.states[a][3:] += action
            self.remain_Dvs[a] -= np.linalg.norm(action)

        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(self._time, self.states[a], self._config.dt, self._config.hpop_in)
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        observations = self._get_observations()
        # 注意在这里重载了奖励计算
        rewards = self._get_rewards(actions)
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()
        self.terminations, self.truncations = terminations, truncations

        current_infos = {a: self.infos.get(a, {}) for a in self.possible_agents if a in self.agents}
        if any(terminations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            if reason == 'capture_success': self.episode_statistics['success_count'] += 1
            elif reason == 'timeout': self.episode_statistics['timeout_count'] += 1
            elif reason == 'fuel_out': self.episode_statistics['fuelout_count'] += 1
            if self.episode_statistics['total_episodes'] > 0: self.episode_statistics['success_rate'] = self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']

        for agent in self.agents:
            current_infos[agent]['termination_reason'] = termination_reasons.get(agent, None)
            current_infos[agent]['episode_statistics'] = self.episode_statistics.copy()

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                if agent in observations: current_infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, current_infos

    def _get_rewards(self, actions):
        # 首先获取父类计算的基础奖励
        rewards = super()._get_rewards(actions)

        # --- Lambert引导奖励 ---
        if self._config.use_lambert_reward and solve_lambert is not None:
            for p_id in self.pursuer_ids:
                if p_id not in self.states or not self.evader_ids:
                    continue
                
                e_id = self.evader_ids[0]
                if e_id not in self.states:
                    continue

                p_state = self.states[p_id]
                e_state = self.states[e_id]

                r1_vec = p_state[:3].tolist()
                r2_vec = e_state[:3].tolist()
                v1_current = p_state[3:]
                action = actions.get(p_id, np.zeros(3))

                try:
                    v1_lambert = np.array(solve_lambert(r1_vec, r2_vec, self._config.lambert_transfer_time, self._config.mu))
                    
                    delta_v_ideal = v1_lambert - v1_current
                    delta_v_ideal_norm = np.linalg.norm(delta_v_ideal)

                    if delta_v_ideal_norm > 1e-6:
                        # 获取理想推力的单位方向向量
                        ideal_direction = delta_v_ideal / delta_v_ideal_norm
                        
                        action_norm = np.linalg.norm(action)
                        if action_norm > 1e-6:
                            # 计算实际推力的单位方向
                            action_direction = action / action_norm
                            # 计算余弦相似度作为奖励 (范围在[-1, 1])
                            cosine_similarity = np.dot(action_direction, ideal_direction)
                            lambert_reward = cosine_similarity
                        else:
                            # 如果没有施加推力，则没有奖励
                            lambert_reward = 0.0
                        
                        # 将lambert奖励的原始值存入info字典，用于可视化
                        if p_id not in self.infos:
                            self.infos[p_id] = {}
                        self.infos[p_id]['raw_lambert_reward'] = lambert_reward

                        rewards[p_id] += self._config.lambert_reward_weight * lambert_reward

                except RuntimeError:
                    pass # Lambert不收敛则跳过

        return rewards

    def _get_observations(self):
        if not self._config.use_partial_obs:
            return super()._get_observations()

        observations = {}
        all_positions = {agent_id: self.states[agent_id][:3] for agent_id in self.possible_agents if agent_id in self.states}
        
        for agent_id in self.possible_agents:
            if agent_id not in self.states: continue
            
            if agent_id.startswith('p_'):
                obs_components = []
                current_pos = all_positions[agent_id]
                
                obs_components.append(self.states[agent_id])
                
                for i in range(self._config.num_e):
                    evader_id = f'e_{i}'
                    abs_prediction = self.pursuer_predictions[evader_id].reshape(self._config.lstm_future_len, 3)
                    rel_prediction = abs_prediction.to(self.device) - torch.from_numpy(current_pos).to(self.device).float()
                    obs_components.append(rel_prediction.cpu().flatten().numpy())
                
                other_pursuer_rel_positions = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id and pursuer_id in all_positions:
                        rel_pos = all_positions[pursuer_id] - current_pos
                        other_pursuer_rel_positions.append(rel_pos)
                
                if other_pursuer_rel_positions:
                    obs_components.append(np.concatenate(other_pursuer_rel_positions))
                elif self._config.num_p > 1:
                    obs_components.append(np.zeros(3 * (self._config.num_p - 1)))

                observations[agent_id] = np.concatenate(obs_components)
            else:
                observations[agent_id] = self.states[agent_id]
        
        return observations

    def _update_predictions_and_prepare_sl_data(self):
        if not self.lstm_model: return

        self.lstm_model.eval()

        for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            history_buffer = self.evader_history_buffers[evader_id]
            
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                true_state = self.states[evader_id]
                history_buffer.append(true_state)
            else:
                if len(self.pursuer_predictions[evader_id].flatten().nonzero()) > 0:
                    last_pred_pos = self.pursuer_predictions[evader_id].reshape(self._config.lstm_future_len, 3)[0].cpu().numpy()
                    if len(history_buffer) > 0:
                        prev_pos = history_buffer[-1][:3]
                        estimated_vel = (last_pred_pos - prev_pos) / self._config.dt
                    else:
                        estimated_vel = np.zeros(3)
                    pseudo_state = np.concatenate([last_pred_pos, estimated_vel])
                    history_buffer.append(pseudo_state)
                else: 
                     history_buffer.append(self.states[evader_id])

            if len(history_buffer) > 0:
                input_seq = torch.tensor(np.array(list(history_buffer)), dtype=torch.float32).unsqueeze(0).to(self.device)
                mask = torch.ones(1, len(history_buffer)).to(self.device)
                with torch.no_grad():
                    new_prediction = self.lstm_model(input_seq, mask=mask).squeeze(0)
                self.pursuer_predictions[evader_id] = new_prediction

            temp_e_state = np.copy(self.states[evader_id])
            future_gt_traj = []
            for step in range(self._config.lstm_future_len):
                current_sim_time = self._time + datetime.timedelta(seconds=(step + 1) * self._config.dt)
                _, temp_e_state = self._orbit_lib.orbit_hpop(current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in)
                future_gt_traj.append(temp_e_state[:3])
            
            sl_data = {
                f'sl_history_input_{evader_id}': np.array(list(history_buffer)),
                f'sl_future_ground_truth_{evader_id}': np.array(future_gt_traj)
            }
            for p_agent_id in [f'p_{i}' for i in range(self._config.num_p)]:
                 if p_agent_id in self.infos:
                    self.infos[p_agent_id].update(sl_data)
