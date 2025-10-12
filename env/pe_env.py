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
    evader_policy_type: str = "None"  # 可选项 "random"， "RL"
    ###
    # 追逃智能体数量
    ###
    num_p: int = 1
    num_e: int = 1
    ###
    # 初始条件
    ###
    init_utc = datetime.datetime(2030, 1, 1, 0, 0, 0)
    init_dv: float = 100.0  # 初始剩余delta V, m/s
    ###
    # 终止条件
    ###
    dist_cap: float = 30.0e3  # 距离 < dist_cap, 抓捕成功
    episode_length: float = 3600.0 * 24  # 每个episode的时间长度

    ###
    # TODO: 奖励函数
    ###
    coeff_dist: float = 1.0
    coeff_cap: float = 30.0

    ###
    # 仿真参数设置
    ###
    dt: float = 60.0  # 每次机动的间隔时间
    dv_step: float = 1 #
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
        assert self.dv_step > 0.0
        assert self.init_dv > 0.0


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

        # --- 核心修改部分：添加 self.possible_agents ---
        # possible_agents 是环境中所有可能的智能体ID（不变的列表）
        possible_p = [f'p_{i}' for i in range(self._config.num_p)]
        possible_e = [f'e_{i}' for i in range(self._config.num_e)]
        self.possible_agents = possible_p + possible_e
        
        # agents 是当前回合活跃的智能体ID（在 reset 中可能会被重置）
        self.agents = []

        # 动作空间、观测空间
        self.action_spaces = {
            a: spaces.Box(-self._config.dv_step, self._config.dv_step, shape=(3,))
            for a in self.possible_agents
        }
        
        # 为不同角色定义不同的观测空间
        self.observation_spaces = {}
        for a in self.possible_agents:
            if a.startswith('p_'):  # 追击方
                # 自身状态(6维) + 所有逃跑方位置(3 * num_e维)
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6 + 3 * self._config.num_e,))
            else:  # 逃跑方
                # 自身状态(6维) - 逃跑方看不到追击方
                self.observation_spaces[a] = spaces.Box(-np.inf, np.inf, shape=(6,))

        self.states = {a: np.zeros(6, ) for a in self.agents}
        self.remain_Dvs = {a: 0.0 for a in self.agents}

        # 渲染
        self.viewer = Viewer(
            width=self._config.width,
            height=self._config.height,
            agents=self.agents,
            max_history=self._config.max_history
        )

    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        sma = 42166300.0  # 轨道半长轴, m
        ecc = 0.0  # 偏心率，无量纲
        inc = 0.0  # 轨道倾角，rad
        argp = 0.0  # 近地点幅角，rad
        raan = 0.0  # 升交点赤经，rad
        ta_ref = np.random.uniform(0.0, 2 * np.pi)  # 虚拟参考星真近点角，rad

        # 为所有追击方和逃跑方初始化状态
        self.states = {}
        for i in range(self._config.num_p):
            agent_id = f'p_{i}'
            ta_pur = (ta_ref + np.random.uniform(low=-0.5, high=0.5)) % (2 * np.pi)
            self.states[agent_id] = self._orbit_lib.coe2rv(np.array([
                sma, ecc, inc, raan, argp, ta_pur
            ]))
        
        for i in range(self._config.num_e):
            agent_id = f'e_{i}'
            ta_eva = (ta_ref + np.random.uniform(low=-0.5, high=0.5)) % (2 * np.pi)
            self.states[agent_id] = self._orbit_lib.coe2rv(np.array([
                sma, ecc, inc, raan, argp, ta_eva
            ]))

        self._time = self._config.init_utc
        self.remain_Dvs = {a: self._config.init_dv for a in self.agents}

        self.viewer.reset()

        observations = self._get_observations()

        infos = {a: {} for a in self.agents}  # dummy infos

        return observations, infos

    def step(self, actions: Dict[str, np.ndarray]):
        # 施加脉冲
        for a in self.agents:
            # 动作模约束
            if np.linalg.norm(actions[a]) > self._config.dv_step:
                actions[a] = actions[a] / np.linalg.norm(actions[a]) * self._config.dv_step

            if np.linalg.norm(actions[a]) > self.remain_Dvs[a]:
                actions[a] = actions[a] / np.linalg.norm(actions[a]) * self.remain_Dvs[a]

            self.states[a][3:] += actions[a]  # 施加速度增量
            self.remain_Dvs[a] -= np.linalg.norm(actions[a])

        # 在J2000下递推
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
        rewards = self._get_rewards()
        truncations = self._get_truncations()
        terminations = self._get_terminations()

        infos = {a: {} for a in self.agents}  # dummy infos
        #修复：移除已终止的agents,移除的agent的infos也要打印
        for agent in list(self.agents):  # 使用list创建副本，避免在迭代时修改
            if terminations[agent] or truncations[agent]:
                self.agents.remove(agent)

        return observations, rewards, terminations, truncations, infos

    def render(self):
        self.viewer.update(self.states)
        return None
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_observations(self):
        # 修改：为不同角色提供不同的观测
        observations = {}
        
        # 获取所有逃跑方的位置（用于构建追击方的观测）
        evader_positions = []
        for i in range(self._config.num_e):
            evader_id = f'e_{i}'
            if evader_id in self.states:
                evader_positions.append(self.states[evader_id][:3])
        
        # 为每个智能体构建观测
        for agent_id in self.agents:
            if agent_id.startswith('p_'):  # 追击方
                # 自身状态 + 所有逃跑方位置
                obs = np.concatenate([
                    self.states[agent_id],  # 自身状态 (6维)
                    np.concatenate(evader_positions)  # 所有逃跑方位置 (3 * num_e维)
                ])
            else:  # 逃跑方
                # 逃跑方只能看到自身状态
                obs = self.states[agent_id]  # 自身状态 (6维)
            
            observations[agent_id] = obs
        
        return observations

    def _get_rewards(self):
        # TODO: 构造rewards
        # 注意：这里只考虑第一个追击方和第一个逃跑方
        # 后续可以根据需要扩展
        rewards = {a: 0.0 for a in self.agents}
        
        # 计算第一个追击方和第一个逃跑方的距离
        if 'p_0' in self.states and 'e_0' in self.states:
            dist = np.linalg.norm(self.states['p_0'][:3] - self.states['e_0'][:3])
        else:
            dist = 0.0  # 如果不存在，设为0
        
        for a in self.agents:
            if a.startswith('p_'):  # 追击方
                rewards[a] += -dist * self._config.coeff_dist
            else:  # 逃跑方
                rewards[a] += dist * self._config.coeff_dist

        if dist <= self._config.dist_cap:
            for a in self.agents:
                if a.startswith('p_'):  # 追击方
                    rewards[a] += self._config.coeff_cap
                else:  # 逃跑方
                    rewards[a] += -self._config.coeff_cap

        return rewards

    def _get_terminations(self):
        """ 判断是否成功抓捕 """
        terminations = {a: False for a in self.agents}
        
        # 距离判断（只考虑第一个追击方和第一个逃跑方）
        if 'p_0' in self.states and 'e_0' in self.states:
            dist = np.linalg.norm(self.states['p_0'][:3] - self.states['e_0'][:3])
            if dist < self._config.dist_cap:
                terminations = {a: True for a in self.agents}
        
        # 剩余速度增量判断
        if any(dv <= 0 for dv in self.remain_Dvs.values()):
            terminations = {a: True for a in self.agents}

        return terminations

    def _get_truncations(self):
        """ 判断是否超过一个 episode 的时间长度 """
        truncations = {a: False for a in self.agents}
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            truncations = {a: True for a in self.agents}
        return truncations
