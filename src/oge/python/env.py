import sys
from functools import lru_cache
from importlib import import_module
from typing import Any, Literal

import oge_py
import gymnasium as gym
import numpy as np

from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

class OGEEnvStepMetadata(TypedDict):
    """Step info options."""
    lives: int
    episode_frame_number: int
    frame_number: int
    seeds: NotRequired[tuple[int, int]]

class OGEEnv(gym.Env, utils.EzPickle):
    """Gymnasium wrapper around the Orbital Game Environment (OGE)."""
