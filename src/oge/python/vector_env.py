from __future__ import annotations

from typing import Any
from dataclasses import dataclass

import oge_py
import gymnasium.vector.utils
import numpy as np

from oge_py.env import OGEEnv, OGEEnvCfg
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv


@dataclass
class OGEVectorEnvCfg(OGEEnvCfg):
    num_envs: int = 4,
    batch_size: int = 0,
    num_threads: int = 0,
    thread_affinity_offset: int = -1,
    autoreset_mode: str = "NextStep"


class OGEVectorEnv(VectorEnv):
    def __init__(
            self,
            cfg: OGEVectorEnvCfg
    ):
        self.cfg = cfg
        self.oge = oge_py.OGEVectorInterface(
            self.cfg.num_envs,
            self.cfg.batch_size,
            self.cfg.num_threads,
            self.cfg.thread_affinity_offset,
            self.cfg.autoreset_mode
        )

        self.metadata["autoreset_mode"] = (
            self.cfg.autoreset_mode
            if isinstance(self.cfg.autoreset_mode, AutoresetMode)
            else AutoresetMode(self.cfg.autoreset_mode)
        )

        self.single_observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.oge.get_single_observation_size(),),
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
        """Reset the sub-environment"""
        if options is None or "reset_mask" not in options:
            reset_indices = np.arange(self.cfg.num_envs)
        else:
            reset_mask = options["reset_mask"]
            assert isinstance(reset_mask, np.ndarray) and reset_mask.dtype == np.bool_
            (reset_indices,) = np.where(reset_mask)

        if seed is None:
            reset_seeds = np.full(len(reset_indices), -1)
        elif isinstance(seed, int):
            reset_seeds = np.arange(seed, seed + len(reset_indices))
        elif isinstance(seed, np.ndarray):
            reset_seeds = seed
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
