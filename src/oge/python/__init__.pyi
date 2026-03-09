from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, TypeAlias, overload

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from oge_py import _oge_py


class SatState:
    def __repr__(self) -> str: ...


class OGESettings:
    def __init__(self) -> None: ...


class OGEInterface:
    def __init__(self, settings: OGESettings) -> None: ...

    def get_rewards(self, actions) -> np.ndarray: ...

    def get_observations(self) -> np.ndarray: ...

    def get_terminal(self) -> bool: ...

    def get_truncated(self) -> bool: ...

    def act(self, actions: np.ndarray) -> None: ...
