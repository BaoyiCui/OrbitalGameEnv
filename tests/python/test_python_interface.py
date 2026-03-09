import os
import pickle
import tempfile

import oge_py
import numpy as np
import pytest

def test_oge_version():
    assert hasattr(oge_py, "__version__")

def test_oge_construction(oge):
    assert isinstance(oge, oge_py.OGEInterface)

def test_bool_config(oge):
    assert