from __future__ import annotations

from typing import Any
from dataclasses import dataclass, asdict

import oge_py
import gymnasium.vector.utils
import numpy as np

from oge_py.env import OGEEnv, OGEEnvCfg
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv


@dataclass
class OGEVectorEnvCfg(OGEEnvCfg):
    num_envs: int = 4
    batch_size: int = 0
    num_threads: int = 0
    thread_affinity_offset: int = -1
    autoreset_mode: str = "NextStep"


class OGEVectorEnv(VectorEnv):
    def __init__(
            self,
            cfg: OGEVectorEnvCfg
    ):
        self.cfg = cfg

        settings = oge_py.OGESettings()
        cfg_dict = asdict(self.cfg)
        _VECTOR_ONLY_KEYS = {"num_envs", "batch_size", "num_threads", "thread_affinity_offset", "autoreset_mode"}
        for key, value in cfg_dict.items():
            if key in _VECTOR_ONLY_KEYS:
                continue
            if isinstance(value, bool):
                settings.set_bool(key, value)
            elif isinstance(value, int):
                settings.set_int(key, value)
            elif isinstance(value, float):
                settings.set_float(key, value)
            elif isinstance(value, str):
                settings.set_string(key, value)

        self.oge = oge_py.OGEVectorInterface(
            self.cfg.num_envs,
            self.cfg.batch_size,
            self.cfg.num_threads,
            self.cfg.thread_affinity_offset,
            self.cfg.autoreset_mode,
            settings
        )

        self.metadata["autoreset_mode"] = (
            self.cfg.autoreset_mode
            if isinstance(self.cfg.autoreset_mode, AutoresetMode)
            else AutoresetMode(self.cfg.autoreset_mode)
        )

        _, single_obs_size = self.oge.get_single_observation_size()
        self.single_observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(single_obs_size,),
            dtype=np.float64
        )
        self.single_action_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float64
        )
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space,
            self.cfg.batch_size,
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space,
            self.cfg.batch_size,
        )

    def reset(
            self,
            *,
            seed: int | np.ndarray | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset all sub-environments. Partial reset via reset_mask is not supported."""
        if options is not None and "reset_mask" in options:
            raise ValueError(
                "Partial reset via 'reset_mask' is not supported. "
                "Use autoreset_mode to handle per-environment resets automatically."
            )

        reset_indices = list(range(self.cfg.num_envs))

        if seed is None:
            reset_seeds = [-1] * self.cfg.num_envs
        elif isinstance(seed, int):
            reset_seeds = list(range(seed, seed + self.cfg.num_envs))
        elif isinstance(seed, np.ndarray):
            reset_seeds = seed.tolist()
        else:
            raise TypeError("Unsupported seed type")

        return self.oge.reset(reset_indices, reset_seeds)

    def step(self, actions: np.ndarray) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]
    ]:
        self.send(actions)
        return self.recv()

    def send(self, actions: np.ndarray):
        self.oge.send(actions)

    def recv(self):
        return self.oge.recv()
