# 1v1 pursuit evasion game
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

from OrbitLib import OrbitLib


@dataclass
class PEEvnCfg:
    init_dv: float = 100.0  # 初始剩余delta V, m/s
    evader_policy_type: str = "None"  # 可选项 "random"， "RL"
    episode_length: float = 3600.0  # 每个episode的时间长度
    dt: float = 60.0  # 每次机动的间隔时间


class PEEnv(ParallelEnv):
    metadata = {
        "name": "Orbital-Pursuit-Evasion-v0",
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 15
    }

    def __init__(self, config: PEEvnCfg):
        super().__init__()
        self._config = config
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,))

    def reset(self, seed=None, options=None):
        sma = 42166300.0  # 轨道半长轴, m
        ecc = 0.0  # 偏心率，无量纲
        inc = 0.0  # 轨道倾角，度
        argp = 0.0  # 近地点幅角，度
        raan = 0.0  # 升交点赤经，度
        ta = np.random.uniform(0.0, 360.0)  # 真近点角，度

        ta_eva = ta + np.random.uniform(low=-0.5, high=0.5)
        ta_pur = ta_eva + np.random.uniform(low=-0.5, high=0.5)
        # TODO: 此处规范化 ta到0.0-360.0

        coe_eva = np.array([sma, ecc, inc, argp, raan, ta_eva])
        coe_pur = np.array([sma, ecc, inc, argp, raan, ta_pur])

        remain_Dv = self._config.init_dv

        # TODO: 返回观测
        observations = {
            agent: (
                
            )
            for agent in self.agents
        }

        return

    def step(self, actions):
        # TODO: 施加脉冲

        # TODO: 在J2000下递推

        # TODO: 返回HILL坐标系下的RV

        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_rewards(self):
        # TODO: 构造rewards
        pass
