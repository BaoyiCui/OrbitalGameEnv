# 1v1 pursuit evasion game
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces

import numpy as np

from pettingzoo import ParallelEnv
from pettingzoo.utils.env import AgentID

from .OrbitLib import OrbitLib, HPOP_In


@dataclass
class PEEnvCfg:
    evader_policy_type: str = "None"  # 可选项 "random"， "RL"
    num_p: int = 1
    num_e: int = 1
    ###
    # 初始条件
    ###
    init_utc = datetime.datetime(2030, 1, 1, 0, 0, 0)
    dt: float = 60.0  # 每次机动的间隔时间
    init_dv: float = 100.0  # 初始剩余delta V, m/s
    ###
    # 终止条件
    ###
    dist_cap = 30.0e3  # 距离 < dist_cap, 抓捕成功
    episode_length: float = 3600.0  # 每个episode的时间长度

    ###
    # TODO: 奖励函数
    ###

    ###
    # 仿真参数设置
    ###
    dt: float = 60.0  # 每次机动的间隔时间
    orbit_lib_path: str = "/home/baoyicui/Workspaces/OrbitalGameEnv/env/OrbitLib/so/X86/libOrbit.so"
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


class PEEnv(ParallelEnv):
    metadata = {
        "name": "Orbital-Pursuit-Evasion-v0",
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 15
    }

    def __init__(self, config: PEEnvCfg):
        super().__init__()
        self._config = config
        self._orbit_lib = OrbitLib(self._config.orbit_lib_path)

        self._time = self._config.init_utc

        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,))
        self.agents = list()
        self.agents.append(f'p_{i}' for i in range(self._config.num_p))
        self.agents.append(f'e_{i}' for i in range(self._config.num_e))

        self.states = {agent: np.zeros(6, ) for agent in self.agents}
        self.remain_Dvs = {agent: 0.0 for agent in self.agents}

    def reset(self, seed=None, options=None):
        sma = 42166300.0  # 轨道半长轴, m
        ecc = 0.0  # 偏心率，无量纲
        inc = 0.0  # 轨道倾角，rad
        argp = 0.0  # 近地点幅角，rad
        raan = 0.0  # 升交点赤经，rad
        ta_ref = np.random.uniform(0.0, 2 * np.pi)  # 虚拟参考星真近点角，rad

        ta_eva = (ta_ref + np.random.uniform(low=-0.5, high=0.5)) % (2 * np.pi)  # 取模限定到 [0.0, 2 * pi)
        ta_pur = (ta_eva + np.random.uniform(low=-0.5, high=0.5)) % (2 * np.pi)

        self.states['p_0'] = self._orbit_lib.coe2rv(np.array([
            sma, ecc, inc, raan, argp, raan, ta_pur
        ]))
        self.states['e_0'] = self._orbit_lib.coe2rv(np.array([
            sma, ecc, inc, raan, argp, raan, ta_eva
        ]))

        # self.remain_Dv = self._config.init_dv
        self.remain_Dvs = {a: self._config.init_dv for a in self.agents}

        observations = {
            agent: self.states[agent]
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}  # dummy infos

        return observations, infos

    def step(self, actions):
        # 施加脉冲
        for agent in self.agents:
            self.states[agent][3:] += actions[agent]  # 施加速度增量
        # 在J2000下递推
        for agent in self.agents:
            self.states[agent] = self._orbit_lib.orbit_hpop(
                self._config.init_utc,
                self.states[agent],
                self._config.dt,
                self._config.hpop_in
            )
        self._time = self._time + datetime.timedelta(seconds=self._config.dt)
        observations = {
            agent: self.states[agent]
            for agent in self.agents
        }

        rewards = self._get_rewards()
        truncations = self._get_truncations()
        terminations = self._get_terminations()

        infos = {agent: {} for agent in self.agents}  # dummy infos

        return observations, rewards, terminations, truncations, infos

    def render(self):
        # TODO: 补全render
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _get_rewards(self):
        # TODO: 构造rewards
        rewards = {a: 0.0 for a in self.agents}
        return rewards

    def _get_terminations(self):
        """ 判断是否成功抓捕 """
        # 距离判断
        dist = np.linalg.norm(self.states['p_0'][:3] - self.states['e_0'][:3])
        terminations = {a: False for a in self.agents}
        if dist < self._config.dist_cap:
            terminations = {a: True for a in self.agents}
        # TODO: 剩余速度增量判断

        return terminations

    def _get_truncations(self):

        """ 判断是否超过一个 episode 的时间长度 """
        truncations = {a: False for a in self.agents}
        if self._time >= self._config.init_utc + datetime.timedelta(seconds=self._config.episode_length):
            truncations = {a: True for a in self.agents}
        return truncations
