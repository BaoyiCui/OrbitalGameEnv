import sys

from functools import lru_cache
from importlib import import_module
from typing import Any, Literal
from dataclasses import dataclass

import oge_py
import gymnasium as gym
import numpy as np

from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict


@dataclass
class OGEEnvCfg:
    # 仿真参数设置
    dt: float = 60.0  # 每次机动的间隔时间
    dv_max_per_step_p: float = 0.01  # pursuer's max dv per step, km/s
    dv_max_per_step_e: float = 0.005  # evader's max dv per step, km/s

    # 初始条件

    # 终止条件

    # 奖励函数设计


class OGEEnvStepMetadata(TypedDict):
    """Step info options."""
    seeds: NotRequired[tuple[int, int]]


class OGEEnv(gym.Env, utils.EzPickle):
    """Gymnasium wrapper around the Orbital Game Environment (OGE)."""
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(
            self,
            config: OGEEnvCfg
    ):
        super().__init__()

