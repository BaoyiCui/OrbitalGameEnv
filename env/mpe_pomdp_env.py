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
    """加入POMDP和LSTM的特定参数"""
    use_partial_obs: bool = True
    obs_interval: int = 2
    lstm_history_len: int = 20
    lstm_future_len: int = 10
    lstm_scheme: int = 1  # 1:rel-to-last, 2:rel-to-virtual-star, 3:rel-to-first

    # Lambert奖励配置 
    use_lambert_reward: bool = True
    lambert_reward_weight: float = 5
    lambert_transfer_time: float = 7200#Lambert转移时间（秒）
    mu: float = 3.986004418e14

class MPE_POMDP_Env(MPEEnv):
    """
    继承自MPEEnv，通过覆盖和添加方法来实现部分可观测性功能。
    """
    def __init__(self, config: MPE_POMDP_EnvCfg = MPE_POMDP_EnvCfg()):
        # 首先调用父类的构造函数，完成大部分初始化
        super().__init__(config)
        self._config: MPE_POMDP_EnvCfg = config

        # 添加归一化参数
        self.position_scale = 1e7  # 10000 km, a common scale for orbital mechanics
        self.velocity_scale = 1e4  # 10 km/s

        self.lstm_model = None
        self.device = None # 新增设备属性
        if self._config.use_partial_obs:
            # 初始化POMDP相关的状态变量
            self.evader_history_buffers = {f'e_{i}': deque(maxlen=self._config.lstm_history_len) for i in range(self._config.num_e)}
            self.pursuer_predictions = {f'e_{i}': torch.zeros(self._config.lstm_future_len * 3) for i in range(self._config.num_e)}
            self.obs_counters = {f'e_{i}': 0 for i in range(self._config.num_e)}

            # 为方案2初始化特定状态
            if self._config.lstm_scheme == 2:
                self.virtual_star_states = {f'e_{i}': np.zeros(6) for i in range(self._config.num_e)}
                self.virtual_star_history_buffers = {f'e_{i}': deque(maxlen=self._config.lstm_history_len) for i in range(self._config.num_e)}

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

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """归一化状态数据"""
        normalized = state.copy()
        normalized[:3] /= self.position_scale
        normalized[3:] /= self.velocity_scale
        return normalized

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
                    initial_state_normalized = self.normalize_state(self.states[evader_id])
                    for _ in range(self._config.lstm_history_len):
                        self.evader_history_buffers[evader_id].append(initial_state_normalized)
                    
                    # 为方案2重置虚拟星状态和历史
                    if self._config.lstm_scheme == 2:
                        self.virtual_star_states[evader_id] = np.copy(self.states[evader_id])
                        self.virtual_star_history_buffers[evader_id].clear()
                        vs_initial_state_normalized = self.normalize_state(self.virtual_star_states[evader_id])
                        for _ in range(self._config.lstm_history_len):
                            self.virtual_star_history_buffers[evader_id].append(vs_initial_state_normalized)

            self._update_predictions_and_prepare_sl_data()
            observations = self._get_observations()

        return observations, infos

    def step(self, actions):
        if self._config.use_partial_obs:
            self._update_predictions_and_prepare_sl_data()
        
        # 动作应用和状态传播 (这部分逻辑保持不变)
        for a in self.agents:
            if a.startswith('p_'): dv_step = self._config.p_dv_step
            else: dv_step = self._config.e_dv_step
            action = actions.get(a, np.zeros(3))
            if np.linalg.norm(action) > dv_step: 
                action = action / np.linalg.norm(action) * dv_step
            if np.linalg.norm(action) > self.remain_Dvs[a]: 
                action = action / np.linalg.norm(action) * self.remain_Dvs[a]
            self.states[a][3:] += action
            self.remain_Dvs[a] -= np.linalg.norm(action)

        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(self._time, self.states[a], self._config.dt, self._config.hpop_in)
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        # 为方案2传播虚拟星
        if self._config.use_partial_obs and self._config.lstm_scheme == 2:
            for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
                if evader_id in self.virtual_star_states:
                    # 虚拟星从回合开始时的状态独立传播，不受动作影响
                    vs_time = self._time - datetime.timedelta(seconds=self._config.dt)
                    _, new_vs_state = self._orbit_lib.orbit_hpop(vs_time, self.virtual_star_states[evader_id], self._config.dt, self._config.hpop_in)
                    self.virtual_star_states[evader_id] = new_vs_state

        observations = self._get_observations()
        rewards = self._get_rewards(actions)
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()
        self.terminations, self.truncations = terminations, truncations

        current_infos = {a: self.infos.get(a, {}) for a in self.possible_agents if a in self.agents}
        if any(terminations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success':
                self.episode_statistics['success_count'] += 1
            elif reason == 'timeout':
                self.episode_statistics['timeout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_timeout_penalty
            elif reason == 'fuel_out':
                self.episode_statistics['fuelout_count'] += 1
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_fuelout_penalty
            
            if self.episode_statistics['total_episodes'] > 0:
                self.episode_statistics['success_rate'] = self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']

        for agent in self.agents:
            current_infos[agent]['termination_reason'] = termination_reasons.get(agent, None)
            current_infos[agent]['episode_statistics'] = self.episode_statistics.copy()

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                if agent in observations: current_infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, current_infos

    def _get_rewards(self, actions):
        # 这部分逻辑保持不变，奖励计算不依赖于归一化
        return super()._get_rewards(actions)

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
                
                # 追击者自身状态也进行归一化
                obs_components.append(self.normalize_state(self.states[agent_id]))
                
                for i in range(self._config.num_e):
                    evader_id = f'e_{i}'
                    evader_current_pos = all_positions.get(evader_id)
                    if evader_current_pos is not None:
                        # 1. 获取LSTM预测的、归一化的、相对于逃逸者当前位置的位移
                        norm_rel_to_evader_prediction = self.pursuer_predictions[evader_id].reshape(self._config.lstm_future_len, 3)
                        
                        # 2. 将其反归一化为真实的物理位移（米）
                        real_rel_to_evader_prediction = norm_rel_to_evader_prediction.cpu().numpy() * self.position_scale
                        
                        # 3. 加上逃逸者当前绝对位置，得到预测的未来绝对位置
                        abs_prediction = real_rel_to_evader_prediction + evader_current_pos
                        
                        # 4. 减去追击者当前绝对位置，得到追击者视角的相对位置
                        rel_to_pursuer_prediction = abs_prediction - current_pos
                        
                        # 5. 对这个最终用于观测的相对位置再次归一化，以保持所有观测特征尺度一致
                        obs_components.append((rel_to_pursuer_prediction / self.position_scale).flatten())
                    else:
                        obs_components.append(np.zeros(self._config.lstm_future_len * 3))
                
                other_pursuer_rel_positions = []
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id and pursuer_id in all_positions:
                        rel_pos = all_positions[pursuer_id] - current_pos
                        # 其他追击者的相对位置也进行归一化
                        other_pursuer_rel_positions.append(rel_pos / self.position_scale)
                
                if other_pursuer_rel_positions:
                    obs_components.append(np.concatenate(other_pursuer_rel_positions))
                elif self._config.num_p > 1:
                    obs_components.append(np.zeros(3 * (self._config.num_p - 1)))

                observations[agent_id] = np.concatenate(obs_components)
            else:
                # 逃逸者自身的观测（如果需要）也归一化
                observations[agent_id] = self.normalize_state(self.states[agent_id])
        
        return observations

    def _update_predictions_and_prepare_sl_data(self):
        if not self.lstm_model: return

        self.lstm_model.eval()

        for evader_id in [f'e_{i}' for i in range(self._config.num_e)]:
            if evader_id not in self.states: continue

            self.obs_counters[evader_id] += 1
            history_buffer = self.evader_history_buffers[evader_id]
            
            # 无论何种情况，存入buffer的都应该是归一化之后的状态
            if self.obs_counters[evader_id] % self._config.obs_interval == 0:
                true_state_normalized = self.normalize_state(self.states[evader_id])
                history_buffer.append(true_state_normalized)
                if self._config.lstm_scheme == 2:
                    vs_state_normalized = self.normalize_state(self.virtual_star_states[evader_id])
                    self.virtual_star_history_buffers[evader_id].append(vs_state_normalized)
            else:
                # 伪观测的逻辑也基于归一化数据
                if len(history_buffer) > 0:
                    norm_rel_disp = self.pursuer_predictions[evader_id].reshape(self._config.lstm_future_len, 3)[0].cpu().numpy()
                    prev_norm_state = history_buffer[-1]
                    next_norm_pos = prev_norm_state[:3] + norm_rel_disp
                    estimated_norm_vel = (next_norm_pos - prev_norm_state[:3]) / (self._config.dt / self.velocity_scale * self.position_scale)
                    pseudo_state_normalized = np.concatenate([next_norm_pos, estimated_norm_vel])
                    history_buffer.append(pseudo_state_normalized)
                    if self._config.lstm_scheme == 2:
                        if len(self.virtual_star_history_buffers[evader_id]) > 0:
                            self.virtual_star_history_buffers[evader_id].append(self.virtual_star_history_buffers[evader_id][-1])
                        else:
                            vs_state_normalized = self.normalize_state(self.virtual_star_states[evader_id])
                            self.virtual_star_history_buffers[evader_id].append(vs_state_normalized)
                else: 
                     history_buffer.append(self.normalize_state(self.states[evader_id]))

            if len(history_buffer) > 0:
                history_abs_normalized = np.array(list(history_buffer))
                input_data_for_lstm = history_abs_normalized

                if self._config.lstm_scheme == 2:
                    vs_history_normalized = np.array(list(self.virtual_star_history_buffers[evader_id]))
                    if len(vs_history_normalized) == len(history_abs_normalized):
                        input_data_for_lstm = history_abs_normalized - vs_history_normalized
                
                input_seq = torch.tensor(input_data_for_lstm, dtype=torch.float32).unsqueeze(0).to(self.device)
                mask = torch.ones(1, len(history_buffer)).to(self.device)
                
                with torch.no_grad():
                    new_prediction = self.lstm_model(input_seq, mask=mask).squeeze(0)
                self.pursuer_predictions[evader_id] = new_prediction

            # 为监督学习准备归一化的相对位移真值
            current_e_pos = self.states[evader_id][:3]
            temp_e_state = np.copy(self.states[evader_id])
            future_gt_traj_normalized = []
            for step in range(self._config.lstm_future_len):
                current_sim_time = self._time + datetime.timedelta(seconds=(step + 1) * self._config.dt)
                _, temp_e_state = self._orbit_lib.orbit_hpop(current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in)
                relative_displacement_normalized = (temp_e_state[:3] - current_e_pos) / self.position_scale
                future_gt_traj_normalized.append(relative_displacement_normalized)
            
            # 准备用于存储在buffer中的监督学习数据
            sl_history_input = np.array(list(history_buffer))
            if self._config.lstm_scheme == 2:
                # 对于方案2，回放缓冲区也需要存储相对历史
                if len(vs_history_normalized) == len(history_abs_normalized):
                    sl_history_input = history_abs_normalized - vs_history_normalized

            sl_data = {
                f'sl_history_input_{evader_id}': sl_history_input,
                f'sl_future_ground_truth_{evader_id}': np.array(future_gt_traj_normalized)
            }
            for p_agent_id in [f'p_{i}' for i in range(self._config.num_p)]:
                 if p_agent_id in self.infos:
                    self.infos[p_agent_id].update(sl_data)