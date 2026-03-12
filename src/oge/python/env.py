from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TypedDict

import oge_py
import gymnasium as gym
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding


class OGEEnvStepMetadata(TypedDict):
    """Step info options."""
    # TODO
    pass


@dataclass
class OGEEnvCfg:
    # TODO
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
        self.set_params()  # TODO: set settings
        self.oge.init()

        pass

    def reset(self):
        super().reset()
        pass

    def step(self):
        pass

    def render(self):
        pass

    def set_params(self):
        # TODO: 将 self.cfg 中的每个参数通过 self.oge 进行设置
        pass
