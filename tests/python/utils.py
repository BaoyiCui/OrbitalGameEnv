import os
from unittest.mock import patch

import oge_py
import gymnasium as gym
import pytest

@pytest.fixture
def oge():
    """Gets an OGE interface."""
    yield oge_py.OGEInterface()

# @pytest.fixture
# def
