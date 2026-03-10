import pytest
import oge_py

@pytest.fixture
def oge_module():
    return oge_py


@pytest.fixture
def oge():
    return oge_py.OGEInterface()