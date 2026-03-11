from __future__ import annotations

from typing import Any

import oge_py
import gymnasium.vector.utils
import numpy as np

from oge.python.env import OGEEnvCfg
from oge_py.env import OGEEnv, OGEEnvCfg
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import AutoresetMode, VectorEnv


class OGEVectorEnv(VectorEnv):
    def __init__(
            self
    ):
        pass

    def reset(self):
        pass  # todo

    def step(self):
        pass  # TODO

    def send(self):
        pass  # TODO

    def recv(self):
        pass  # TODO
