import pytest


@pytest.fixture(scope="session")
def oge_module():
    return pytest.importorskip("oge_py", reason="oge_py is not installed")


@pytest.fixture
def oge(oge_module):
    """Create an OGE interface instance for tests."""
    return oge_module.OGEInterface()
