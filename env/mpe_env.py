# MPE-Env: 使用继承实现的多智能体追逃环境
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import datetime
import numpy as np
from gymnasium import spaces

# 导入基础环境及其配置
from .pe_env import PEEnv, PEEnvCfg

@dataclass
class MPEEnvCfg(PEEnvCfg):
    """多智能体追逃环境的配置,后续需要调试"""
    # --- 4v1场景 ---
    num_p: int = 4
    num_e: int = 1
    
    # --- 覆盖多智能体逻辑的奖励权重 ---
    reward_dist_weight: float = 0.3  # 距离权重
    reward_time_weight: float = 0.05  # 时间惩罚权重
    reward_formation_weight: float = 0.04  # 群体形成奖励权重
    reward_fuel_weight: float = 0.2  # 燃料消耗惩罚权重
    # 稀疏奖励
    reward_capture: float = 20.0  # 成功抓捕的奖励
    reward_timeout_penalty: float = -2.0  # 超时失败的惩罚
    reward_fuelout_penalty: float = -2.0  # 燃料耗尽的惩罚

    # 过程优势奖励参数
    reward_advantage_weight: float = 0.4  # 过程优势奖励的权重
    advantage_reward_horizon: float = 3600*5  # 优势奖励的预测时间窗口（单位秒）
    arena_radius: float = 100e+3  # 参考距离（单位：米），用于距离缩小

class MPEEnv(PEEnv):
    """
    继承自PEEnv的多智能体追逃环境。
    它为多追击者场景修改了观测空间、奖励函数和终止条件。
    支持课程学习
    """
    def __init__(self, config: MPEEnvCfg = MPEEnvCfg()):
        # 初始化基类。基类的__init__会完成所有设置
        # 除了需要在此处覆盖的观测空间
        super().__init__(config)
        
        self.metadata["is_parallelizable"] = True
        
        # --- 覆盖多智能体场景下的观测空间 ---
        """注意这里观测空间后续不完全观测的情况需要更改"""
        self.observation_spaces = {}
        for a in self.possible_agents:
            if a.startswith('p_'):
                # 维度6 (自身) + 3*num_e (逃逸方) + 3*(num_p-1) (其他追击方)
                obs_shape = (6 + 3 * self._config.num_e + 3 * (self._config.num_p - 1),)
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=obs_shape)
            else:
                # 维度6 (自身) ，逃逸方的观测空间可根据需要进一步扩展
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))
        
        self.infos = {a: {} for a in self.possible_agents}

        # 用于课程学习的统计信息
        self.episode_statistics = {
            'success_count': 0,
            'timeout_count': 0,
            'fuelout_count': 0,
            'total_episodes': 0,
            'success_rate': 0.0
        }

    def observe(self, agent):
        return self._get_observations().get(agent)

    def _get_observations(self):
        observations = {}
        
        # 收集所有智能体位置信息
        all_positions = {}
        for agent_id in self.possible_agents:
            if agent_id in self.states:
                all_positions[agent_id] = self.states[agent_id][:3]
        
        # 为每个智能体构建观测
        for agent_id in self.possible_agents:
            if agent_id not in self.states:
                continue
                
            if agent_id.startswith('p_'):  # 追击方
                obs_components = []
                
                # 1. 自身状态（6维）
                obs_components.append(self.states[agent_id])
                
                # 2. 所有逃逸方绝对位置（3*num_e维）
                evader_positions = []
                for i in range(self._config.num_e):
                    evader_id = f'e_{i}'
                    if evader_id in all_positions:
                        evader_positions.append(all_positions[evader_id])
                
                if evader_positions:
                    obs_components.append(np.concatenate(evader_positions))
                else:
                    # 如果没有逃方，添加零向量以保持维度一致
                    obs_components.append(np.zeros(3 * self._config.num_e))
                
                # 3. 其他追击方相对位置（3*(num_p-1)维）
                other_pursuer_rel_positions = []
                current_pos = all_positions[agent_id]
                
                for i in range(self._config.num_p):
                    pursuer_id = f'p_{i}'
                    if pursuer_id != agent_id and pursuer_id in all_positions:
                        rel_pos = all_positions[pursuer_id] - current_pos
                        other_pursuer_rel_positions.append(rel_pos)
                
                # 处理没有其他追击方的情况
                if other_pursuer_rel_positions:
                    obs_components.append(np.concatenate(other_pursuer_rel_positions))
                else:
                    # 如果没有其他追击者，添加一个零向量以保持维度一致
                    obs_components.append(np.zeros(3 * (self._config.num_p - 1)))

                observations[agent_id] = np.concatenate(obs_components)
            else:  # 逃方
                observations[agent_id] = self.states[agent_id]
        
        return observations

    def _get_rewards(self, actions: Dict[str, np.ndarray]):
        rewards = {a: 0.0 for a in self.agents}
        
        # 收集位置信息
        pursuer_positions = []
        pursuer_ids = [f'p_{i}' for i in range(self._config.num_p)]
        for pid in pursuer_ids:
            if pid in self.states:
                pursuer_positions.append(self.states[pid][:3])
        
        evader_positions = []
        evader_ids = [f'e_{i}' for i in range(self._config.num_e)]
        for eid in evader_ids:
            if eid in self.states:
                evader_positions.append(self.states[eid][:3])
        
        if not evader_positions or not pursuer_positions:
            return rewards
        
        # 计算阵型得分
        formation_score = self._calculate_formation_score(pursuer_positions, evader_positions[0])
        
        # 计算群体形成奖励
        formation_reward = -self._config.reward_formation_weight * formation_score
        
        # 检查抓捕状态
        dists_to_evader = [np.linalg.norm(p_pos - evader_positions[0]) for p_pos in pursuer_positions]
        capture_occurred = min(dists_to_evader) < self._config.dist_cap

        # 过程优势奖励
        num_future_steps = int(self._config.advantage_reward_horizon / self._config.dt)
        future_rewards = {a: 0.0 for a in pursuer_ids}  # 为每个追击方存储过程优势奖励
        
        if num_future_steps > 0 and evader_ids[0] in self.states:
            # 预测逃逸方轨迹
            temp_e_state = np.copy(self.states[evader_ids[0]])
            future_e_traj = []
            for step in range(num_future_steps):
                current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                _, temp_e_state = self._orbit_lib.orbit_hpop(
                    current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in
                )
                future_e_traj.append(temp_e_state[:3])
            
            # 为每个追击方计算过程优势奖励
            for i, agent_id in enumerate(pursuer_ids):
                if agent_id in self.agents:
                    temp_p_state = np.copy(self.states[agent_id])
                    min_future_dist = float('inf')
                    min_step = num_future_steps  # 记录达到最小距离的步数
                    
                    for step in range(num_future_steps):
                        current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                        _, temp_p_state = self._orbit_lib.orbit_hpop(
                            current_sim_time, temp_p_state, self._config.dt, self._config.hpop_in
                        )
                        dist = np.linalg.norm(temp_p_state[:3] - future_e_traj[step])
                        if dist < min_future_dist:
                            min_future_dist = dist
                            min_step = step
                    
                    # 计算过程优势奖励
                    if min_future_dist < self._config.dist_cap:
                        # 奖励: 预测能抓捕。越早、越近，奖励越高
                        time_factor = (num_future_steps - min_step) / num_future_steps
                        dist_factor = (self._config.dist_cap - min_future_dist) / self._config.dist_cap
                        radv = self._config.reward_advantage_weight * time_factor * dist_factor
                    else:
                        # 惩罚: 预测不能抓捕。越早、越远，惩罚越大
                        time_factor = (num_future_steps - min_step) / num_future_steps
                        miss_dist = min_future_dist - self._config.dist_cap
                        dist_factor = 1 - np.exp(-2.3e-5 * miss_dist)  # 归一化距离惩罚因子
                        radv = -self._config.reward_advantage_weight * time_factor * dist_factor
                    
                    future_rewards[agent_id] = radv

        # 分配奖励
        for i, agent_id in enumerate(pursuer_ids):
            if agent_id in self.agents:
                # 缩小距离
                normalized_dist = dists_to_evader[i] / self._config.arena_radius
                # 个体距离惩罚（使用缩小距离和对应权重）
                dist_reward = -self._config.reward_dist_weight * normalized_dist
                
                # 时间惩罚
                time_penalty = -self._config.reward_time_weight
                
                # 燃料消耗
                fuel_penalty = -self._config.reward_fuel_weight * np.linalg.norm(actions.get(agent_id, np.zeros(3)))
                
                # 过程优势奖励
                radv = future_rewards.get(agent_id, 0.0)
                
                # 总奖励 = 距离惩罚 + 队形奖励 + 时间惩罚 + 燃料惩罚 + 过程优势奖励
                rewards[agent_id] = dist_reward + formation_reward + time_penalty + fuel_penalty + radv

        # 抓捕成功/失败的终端奖励
        if capture_occurred:
            for i, agent_id in enumerate(pursuer_ids):
                if agent_id in self.agents:
                    if dists_to_evader[i] < self._config.dist_cap:
                        rewards[agent_id] += self._config.reward_capture  # 捕获者奖励
                    else:
                        rewards[agent_id] += self._config.reward_capture * 0.5  # 协助者奖励
        
        # 为逃逸方分配奖励（当前版本不计算）
        # if evader_ids[0] in self.agents:
        #     pursuer_total_reward = sum(rewards.get(pid, 0) for pid in pursuer_ids)
        #     rewards[evader_ids[0]] = -pursuer_total_reward

        return rewards

    def _calculate_formation_score(self, pursuer_positions, evader_pos):
        """计算形成得分，鼓励良好的包围阵型。返回值越小越好。"""
        if len(pursuer_positions) < 2:
            return 0.0
        
        # 计算每个追击方到逃逸方的单位向量
        unit_vectors = []
        for pursuer_pos in pursuer_positions:
            vec = pursuer_pos - evader_pos
            dist = np.linalg.norm(vec)
            if dist < 1e-6:
                unit_vectors.append(np.zeros(3))
            else:
                unit_vectors.append(vec / dist)
        
        # 计算所有单位向量之和的模长。理想包围是各向量互相抵消，模长为0
        sum_of_vectors = np.sum(unit_vectors, axis=0)
        formation_score = np.linalg.norm(sum_of_vectors)
        
        return formation_score

    def _get_terminations(self):
        """判断终止条件，分为成功和失败两种情况"""
        terminations = {a: False for a in self.agents}
        termination_reasons = {a: None for a in self.agents}  # 记录终止原因
        
        # 1. 检查抓捕成功
        evader_pos = None
        if 'e_0' in self.states:  # 目前只有一个逃方
            evader_pos = self.states['e_0'][:3]
        
        if evader_pos is not None:
            capture_occurred = False
            for i in range(self._config.num_p):
                pursuer_id = f'p_{i}'
                if pursuer_id in self.states:
                    dist = np.linalg.norm(self.states[pursuer_id][:3] - evader_pos)
                    if dist < self._config.dist_cap:
                        capture_occurred = True
                        break
            if capture_occurred:
                terminations = {a: True for a in self.agents}
                termination_reasons = {a: 'capture_success' for a in self.agents}
                return terminations, termination_reasons

        # 2. 检查燃料耗尽（失败）
        if any(dv <= 0 for dv in self.remain_Dvs.values()):
            terminations = {a: True for a in self.agents}
            termination_reasons = {a: 'fuel_out' for a in self.agents}
            return terminations, termination_reasons

        # 3. 检查超时（失败）
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            terminations = {a: True for a in self.agents}
            termination_reasons = {a: 'timeout' for a in self.agents}
            return terminations, termination_reasons

        return terminations, termination_reasons

    def _get_truncations(self):
        """截断条件（当前版本为空，所有终止都通过终止条件处理）"""
        return {a: False for a in self.agents}

    def step(self, actions: Dict[str, np.ndarray]):
        """重写step方法,支持终止条件分类和课程学习统计"""
        for a in self.agents:
            if a.startswith('p_'):
                dv_step = self._config.p_dv_step
            else:
                dv_step = self._config.e_dv_step

            if np.linalg.norm(actions[a]) > dv_step:
                actions[a] = actions[a] / np.linalg.norm(actions[a]) * dv_step

            if np.linalg.norm(actions[a]) > self.remain_Dvs[a]:
                actions[a] = actions[a] / np.linalg.norm(actions[a]) * self.remain_Dvs[a]

            self.states[a][3:] += actions[a]
            self.remain_Dvs[a] -= np.linalg.norm(actions[a])

        for a in self.agents:
            _, new_state = self._orbit_lib.orbit_hpop(
                self._time,
                self.states[a],
                self._config.dt,
                self._config.hpop_in
            )
            self.states[a] = new_state
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)

        observations = self._get_observations()
        rewards = self._get_rewards(actions)
        
        # 获取终止原因
        terminations, termination_reasons = self._get_terminations()
        truncations = self._get_truncations()

        self.terminations = terminations
        self.truncations = truncations

        # 更新课程学习统计信息
        if any(terminations.values()):
            self.episode_statistics['total_episodes'] += 1
            reason = list(termination_reasons.values())[0] if termination_reasons else 'unknown'
            
            if reason == 'capture_success':
                self.episode_statistics['success_count'] += 1
                # 为成功添加额外奖励
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_capture
            elif reason == 'timeout':
                self.episode_statistics['timeout_count'] += 1
                # 为超时给追击方添加惩罚
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_timeout_penalty
            elif reason == 'fuel_out':
                self.episode_statistics['fuelout_count'] += 1
                # 为燃料耗尽给追击方添加惩罚
                for a in self.agents:
                    if a.startswith('p_'):
                        rewards[a] += self._config.reward_fuelout_penalty
            
            # 更新成功率
            self.episode_statistics['success_rate'] = (
                self.episode_statistics['success_count'] / self.episode_statistics['total_episodes']
            )

        self.infos = {a: {} for a in self.agents}
        # 将终止原因添加到info中
        for agent in self.agents:
            self.infos[agent]['termination_reason'] = termination_reasons.get(agent, None)
            self.infos[agent]['episode_statistics'] = self.episode_statistics.copy()

        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                self.infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed=None, options=None):
        """重写reset方法,保留课程学习统计"""
        # 调用父类的reset
        observations, self.infos = super().reset(seed, options)

        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        
        # 在infos中添加统计信息
        for agent in self.agents:
            self.infos[agent]['episode_statistics'] = self.episode_statistics.copy()
        
        return observations, self.infos

    def get_success_rate(self):
        """获取当前成功率，用于课程学习"""
        return self.episode_statistics['success_rate']

    def get_episode_statistics(self):
        """获取完整的课程学习统计信息"""
        return self.episode_statistics.copy()

    def set_difficulty_parameters(self, 
                                 episode_length=None, 
                                 dist_cap=None, 
                                 reward_weights=None,
                                 p_init_dv=None):
        """动态调整难度参数，支持课程学习"""
        if episode_length is not None:
            self._config.episode_length = episode_length
        if dist_cap is not None:
            self._config.dist_cap = dist_cap
        if p_init_dv is not None:
            self._config.p_init_dv = p_init_dv
        if reward_weights is not None:
            # 可以动态调整奖励权重
            if 'reward_capture' in reward_weights:
                self._config.reward_capture = reward_weights['reward_capture']
            if 'reward_timeout_penalty' in reward_weights:
                self._config.reward_timeout_penalty = reward_weights['reward_timeout_penalty']