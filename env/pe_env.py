
# 1v1 pursuit evasion game
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .OrbitLib import OrbitLib, HPOP_In
from .viewer import Viewer
from copy import copy

@dataclass
class PEEnvCfg:
    evader_policy_type: str = "None"  # 可选项 "random", "RL"
    ###
    # 追逃智能体数量
    ###
    num_p: int = 1
    num_e: int = 1
    ###
    # 初始条件
    ###
    init_utc = datetime.datetime(2030, 1, 1, 0, 0, 0)
    p_init_dv: float = 500.0  # 追击方初始剩余delta V, m/s
    e_init_dv: float = 100.0  # 逃逸方初始剩余delta V, m/s
    ###
    # 终止条件
    ###
    dist_cap: float = 30.0e3  # 距离 < dist_cap (30km), 抓捕成功
    episode_length: float = 3600.0 * 24  # 每个episode的时间长度

    ###
    # 奖励函数设计 (New)
    ###
    reward_dist_weight: float = 0.00001 # 距离奖励的经验权重 
    reward_time_weight: float = 0.1 # 时间奖励的固定系数
    reward_advantage_weight: float = 5.0 # 过程优势奖励的经验权重 
    reward_fuel_weight: float = 1.0 # 燃料消耗的经验权重 
    reward_capture: float = 1000.0 # 成功抓捕的奖励
    advantage_reward_horizon: float = 3600.0 # 优势奖励的预测时间窗口 (秒, 60分钟)

    ###
    # 仿真参数设置
    ###
    dt: float = 60.0  # 每次机动的间隔时间
    p_dv_step: float = 1.5 # 追击方每次机动的最大速度增量, m/s
    e_dv_step: float = 1.0 # 逃逸方每次机动的最大速度增量, m/s
    hpop_in = HPOP_In(  # HPOP 初始化参数，全局变量
        inial=True,
        mass=50,
        fuel=20,
        thrust=0.0,
        Isp=20.0,
        Sd=1.0,
        Sr=1.0,
        Cd=2.2,
        eta=1.0,
        Propagator_Type=10,  # 二体动力学
        Dyn_Type=0  # 无效，J2摄动
    )

    ###
    # 渲染
    ###
    debug_vis = False
    width: int = 800
    height: int = 600
    max_history = 60

    def check_params(self):
        # 修改：允许多个追击方和逃跑方
        assert self.num_p >= 1
        assert self.num_e >= 1
        assert self.p_dv_step > 0.0
        assert self.e_dv_step > 0.0
        assert self.p_init_dv > 0.0
        assert self.e_init_dv > 0.0


class PEEnv(ParallelEnv):
    metadata = {
        "name": "Orbital-Pursuit-Evasion-v0",
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 15
    }

    def __init__(self, config: PEEnvCfg = PEEnvCfg()):
        super().__init__()
        self._config = config
        self._config.check_params()  # 检查参数是否合法

        self._orbit_lib = OrbitLib()

        self._time = self._config.init_utc
        self.render_mode = None

        possible_p = [f'p_{i}' for i in range(self._config.num_p)]
        possible_e = [f'e_{i}' for i in range(self._config.num_e)]
        self.possible_agents = possible_p + possible_e
        
        self.agents = []

        self.action_spaces = {}
        for a in self.possible_agents:
            if a.startswith('p_'):
                dv_step = self._config.p_dv_step
            else:
                dv_step = self._config.e_dv_step
            self.action_spaces[a] = spaces.Box(-dv_step, dv_step, shape=(3,))
        
        self.observation_spaces = {}
        for a in self.possible_agents:
            if a.startswith('p_'):
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6 + 3 * self._config.num_e,))
            else:
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

        self.states = {a: np.zeros(6, ) for a in self.agents}
        self.remain_Dvs = {a: 0.0 for a in self.agents}
        self.last_dist = 0.0 # 用于计算奖励塑形

        # 渲染器将在第一次调用render()时被初始化
        self.viewer = None

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        sma = 42166300.0  # 轨道半长轴, m
        ecc = 0.0
        inc = 0.0
        argp = 0.0
        raan = 0.0
        ta_ref = np.random.uniform(0.0, 2 * np.pi)

        self.states = {}
        # 均匀分布n个追击方
        angle_step = 2 * np.pi / self._config.num_p
        for i in range(self._config.num_p):
            agent_id = f'p_{i}'
            ta_pur = (ta_ref + i * angle_step) % (2 * np.pi)
            self.states[agent_id] = self._orbit_lib.coe2rv(np.array([
                sma, ecc, inc, raan, argp, ta_pur
            ]))
        
        for i in range(self._config.num_e):
            agent_id = f'e_{i}'
            # 确保逃跑方初始位置与最近追击方的距离在 (dist_cap + 20km, dist_cap + 120km) 的动态范围内
            while True:
                ta_eva = (ta_ref + np.random.uniform(low=-0.5, high=0.5)) % (2 * np.pi)
                eva_state = self._orbit_lib.coe2rv(np.array([
                    sma, ecc, inc, raan, argp, ta_eva
                ]))

                min_dist = float('inf')
                for j in range(self._config.num_p):
                    dist = np.linalg.norm(eva_state[:3] - self.states[f'p_{j}'][:3])
                    if dist < min_dist:
                        min_dist = dist
                
                # 检查与最近的追击方的距离是否在 (dist_cap + 20km, dist_cap + 120km) 范围内
                if min_dist > self._config.dist_cap + 20.0e3 and min_dist < self._config.dist_cap + 120.0e3:
                    self.states[agent_id] = eva_state
                    break

        self._time = self._config.init_utc
        
        self.remain_Dvs = {}
        for a in self.agents:
            if a.startswith('p_'):
                self.remain_Dvs[a] = self._config.p_init_dv
            else:
                self.remain_Dvs[a] = self._config.e_init_dv

        # 初始化上一步距离
        if 'p_0' in self.states and 'e_0' in self.states:
            self.last_dist = np.linalg.norm(self.states['p_0'][:3] - self.states['e_0'][:3])
        else:
            self.last_dist = 0.0

        if self.viewer is not None:
            self.viewer.reset()

        observations = self._get_observations()
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
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
        truncations = self._get_truncations()
        terminations = self._get_terminations()

        infos = {a: {} for a in self.agents}
        for agent in list(self.agents):
            if terminations.get(agent, False) or truncations.get(agent, False):
                # For SB3, it's important to have the final observation in the info dict
                infos[agent]['final_observation'] = observations[agent]
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(
                width=self._config.width,
                height=self._config.height,
                agents=self.possible_agents,
                max_history=self._config.max_history
            )
        self.viewer.update(self.states)
        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_observations(self):
        observations = {}
        evader_positions = []
        for i in range(self._config.num_e):
            evader_id = f'e_{i}'
            if evader_id in self.states:
                evader_positions.append(self.states[evader_id][:3])
        
        for agent_id in self.possible_agents: # Observe for all possible agents
            if agent_id not in self.states:
                continue # Skip if state is not available

            if agent_id.startswith('p_'):
                obs = np.concatenate([
                    self.states[agent_id],
                    np.concatenate(evader_positions) if evader_positions else np.array([])
                ])
            else:
                obs = self.states[agent_id]
            observations[agent_id] = obs
        
        return observations

    def _get_rewards(self, actions: Dict[str, np.ndarray]):
        rewards = {a: 0.0 for a in self.agents}
        

        p_agent = 'p_0'
        e_agent = 'e_0'

        if p_agent not in self.states or e_agent not in self.states:
            return rewards

        p_state = self.states[p_agent]
        e_state = self.states[e_agent]
        
        # --- 1. 距离奖励 (Distance Reward) ---
        current_dist = np.linalg.norm(p_state[:3] - e_state[:3])
        dist_reward = self._config.reward_dist_weight * current_dist
        rewards[p_agent] -= dist_reward
        rewards[e_agent] += dist_reward

        # --- 2. 时间奖励 (Time Reward) ---
        is_capture_condition_met = current_dist < self._config.dist_cap
        if not is_capture_condition_met:
            rewards[p_agent] -= self._config.reward_time_weight
            rewards[e_agent] += self._config.reward_time_weight
        
        # --- 4. 燃料消耗奖励 (Fuel Consumption Reward) ---
        for agent_id, action in actions.items():
            if agent_id in rewards:
                fuel_consumption = np.linalg.norm(action)
                fuel_penalty = self._config.reward_fuel_weight * fuel_consumption
                rewards[agent_id] -= fuel_penalty

        # --- 3. 过程优势诱导奖励 (radv) ---
        # (新版设计：固定预测60步，结合时间和距离，区分奖励与惩罚)
        num_future_steps = int(self._config.advantage_reward_horizon / self._config.dt)
        
        if num_future_steps > 0:
            temp_p_state = np.copy(p_state)
            temp_e_state = np.copy(e_state)
            
            future_dists = []
            for step in range(num_future_steps):
                current_sim_time = self._time + datetime.timedelta(seconds=step * self._config.dt)
                
                _, temp_p_state = self._orbit_lib.orbit_hpop(
                    current_sim_time, temp_p_state, self._config.dt, self._config.hpop_in
                )
                _, temp_e_state = self._orbit_lib.orbit_hpop(
                    current_sim_time, temp_e_state, self._config.dt, self._config.hpop_in
                )
                
                future_dists.append(np.linalg.norm(temp_p_state[:3] - temp_e_state[:3]))

            if future_dists:
                min_future_dist = min(future_dists)
                delta_h_steps = future_dists.index(min_future_dist) + 1

                if min_future_dist < self._config.dist_cap:
                    # 奖励: 预测能抓捕。越早、越近，奖励越高
                    time_factor = (num_future_steps - delta_h_steps) / num_future_steps
                    dist_factor = (self._config.dist_cap - min_future_dist) / self._config.dist_cap
                    radv = self._config.reward_advantage_weight * time_factor * dist_factor
                else:
                    # 惩罚: 预测不能抓捕。越早、越远，惩罚越大
                    time_factor = (num_future_steps - delta_h_steps) / num_future_steps
                    miss_dist = min_future_dist - self._config.dist_cap
                    dist_factor = 1 - np.exp(-2.3e-5 * miss_dist) # 归一化距离惩罚因子
                    radv = -self._config.reward_advantage_weight * time_factor * dist_factor
                
                rewards[p_agent] += radv
                rewards[e_agent] -= radv

        # --- 5. 终端奖励 (Terminal Reward) ---
        if is_capture_condition_met:
            rewards[p_agent] += self._config.reward_capture
            rewards[e_agent] -= self._config.reward_capture

        return rewards

    def _get_terminations(self):
        """ 判断是否成功抓捕 """
        terminations = {a: False for a in self.agents}
        
        if 'p_0' in self.states and 'e_0' in self.states:
            dist = np.linalg.norm(self.states['p_0'][:3] - self.states['e_0'][:3])
            if dist < self._config.dist_cap:
                terminations = {a: True for a in self.agents}
        
        if any(dv <= 0 for dv in self.remain_Dvs.values()):
            terminations = {a: True for a in self.agents}

        return terminations

    def _get_truncations(self):
        """ 判断是否超过一个 episode 的时间长度 """
        truncations = {a: False for a in self.agents}
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            truncations = {a: True for a in self.agents}
        return truncations
