from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from typing import TypedDict, Any, Literal

import oge_py
import gymnasium as gym
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from scipy.signal import step


class OGEEnvStepMetadata(TypedDict):
    """Step info options."""
    current_time: float


@dataclass
class OGEEnvCfg:
    num_pursuers: int = 4
    num_evaders: int = 1
    pass


class OGEEnv(gym.Env, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
            self,
            cfg: OGEEnvCfg
    ):
        # TODO: check cfg

        utils.EzPickle.__init__(
            self,
            cfg
        )

        # Initialize OGE
        self.oge = oge_py.OGEInterface()
        self.cfg = cfg
        self.set_params()
        self.oge.init()

        self.action_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.oge.get_obs_size(),),
            dtype=np.float64,
        )

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, OGEEnvStepMetadata]:
        super().reset(seed=seed, options=options)
        self.oge.reset()
        observations = self.oge.get_observations()
        return observations, self._get_info()

    def step(
            self,
            actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, OGEEnvStepMetadata]:
        """Perform one step

        :param actions:
        :return:
        """
        self.oge.act(actions)
        observations = self.oge.get_observations()
        rewards = self.oge.get_rewards(actions)
        termination = self.oge.is_terminal()
        truncation = self.oge.is_truncated()

        return observations, rewards, termination, truncation, self._get_info()

    def render(self):
        # TODO: need to render here or using c++ backend?
        pass

    def set_params(self) -> None:
        cfg_dict = asdict(self.cfg)
        for key, value in cfg_dict.items():
            if isinstance(value, bool):
                self.oge.setBool(key, value)
            elif isinstance(value, int):
                self.oge.setInt(key, value)
            elif isinstance(value, float):
                self.oge.setFloat(key, value)
            elif isinstance(value, str):
                self.oge.setString(key, value)
            else:
                raise TypeError(f"Unsupported type of key={key}: {type(value)}")

    def _get_info(self) -> OGEEnvStepMetadata:
        return {
            "current_time": self.oge.get_current_time()
        }
