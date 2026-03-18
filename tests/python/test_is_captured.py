import numpy as np
import pytest

import oge_py
from oge_py.env import OGEEnv, OGEEnvCfg


@pytest.fixture
def cfg() -> OGEEnvCfg:
    return OGEEnvCfg(num_pursuers=1, num_evaders=1)


@pytest.fixture
def env(cfg) -> OGEEnv:
    e = OGEEnv(cfg)
    yield e
    e.close()


def _zero_actions(cfg: OGEEnvCfg) -> np.ndarray:
    n = cfg.num_pursuers + cfg.num_evaders
    return np.zeros((n, 3), dtype=np.float64)


class TestIsCaptured:
    def test_not_captured_after_reset(self, env):
        """Immediately after reset, no capture should have occurred."""
        env.reset()
        assert env.is_captured() is False

    def test_not_captured_after_zero_step(self, env, cfg):
        """A zero-action step from default init should not cause capture."""
        env.reset()
        env.step(_zero_actions(cfg))
        assert env.is_captured() is False

    def test_captured_when_pursuer_on_evader(self, env, cfg):
        """Place the pursuer exactly on the evader — should be captured after one step."""
        env.reset()
        states = env.oge.get_sat_states()

        # Find evader and pursuer keys
        evader_key = [k for k in states if k.startswith("e")][0]
        pursuer_key = [k for k in states if k.startswith("p")][0]

        # Move pursuer to evader's position and velocity
        evader_state = states[evader_key]
        states[pursuer_key].r_j2000 = evader_state.r_j2000.copy()
        states[pursuer_key].v_j2000 = evader_state.v_j2000.copy()

        env.reset(options={"states": states})
        # After reset_with_states the positions overlap, but capture is checked during act
        env.step(_zero_actions(cfg))
        assert env.is_captured() is True

    def test_returns_bool(self, env):
        """is_captured() should always return a bool."""
        env.reset()
        assert isinstance(env.is_captured(), bool)