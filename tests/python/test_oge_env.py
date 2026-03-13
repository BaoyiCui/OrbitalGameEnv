import numpy as np
import pytest

from oge_py.env import OGEEnv, OGEEnvCfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cfg() -> OGEEnvCfg:
    return OGEEnvCfg(num_pursuers=4, num_evaders=1)


@pytest.fixture
def env(default_cfg) -> OGEEnv:
    e = OGEEnv(default_cfg)
    yield e
    e.close()


def _zero_actions(cfg: OGEEnvCfg) -> np.ndarray:
    """Valid zero-filled action array shaped (num_agents, 3)."""
    n = cfg.num_pursuers + cfg.num_evaders
    return np.zeros((n, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_action_space_shape(self, env, default_cfg):
        """action_space covers a single agent's 3-D action."""
        assert env.action_space.shape == (3,)

    def test_observation_space_is_1d(self, env):
        """observation_space is a 1-D Box with positive size."""
        assert env.observation_space.shape[0] > 0

    def test_custom_cfg_is_applied(self):
        cfg = OGEEnvCfg(num_pursuers=4, num_evaders=1)
        env = OGEEnv(cfg)
        assert env.cfg.num_pursuers == 4
        assert env.cfg.num_evaders == 1
        env.close()


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_obs_matches_observation_space(self, env):
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[-1] == env.observation_space.shape[-1]
        assert obs.dtype == np.float64

    def test_reset_info_contains_current_time(self, env):
        _, info = env.reset()
        assert "current_time" in info
        assert isinstance(info["current_time"], float)

    def test_reset_obs_shape_stable_across_calls(self, env):
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        assert obs1.shape == obs2.shape

    def test_reset_with_seed_does_not_raise(self, env):
        env.reset(seed=42)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_five_tuple(self, env, default_cfg):
        env.reset()
        result = env.step(_zero_actions(default_cfg))
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_obs_matches_observation_space(self, env, default_cfg):
        env.reset()
        obs, _, _, _, _ = env.step(_zero_actions(default_cfg))
        assert isinstance(obs, np.ndarray)
        assert obs.shape[-1] == env.observation_space.shape[-1]

    def test_step_rewards_shape_matches_num_agents(self, env, default_cfg):
        env.reset()
        _, rewards, _, _, _ = env.step(_zero_actions(default_cfg))
        n_agents = default_cfg.num_pursuers + default_cfg.num_evaders
        assert isinstance(rewards, np.ndarray)
        assert rewards.shape == (n_agents,)

    def test_step_terminated_and_truncated_are_bool(self, env, default_cfg):
        env.reset()
        _, _, terminated, truncated, _ = env.step(_zero_actions(default_cfg))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_info_contains_current_time(self, env, default_cfg):
        env.reset()
        _, _, _, _, info = env.step(_zero_actions(default_cfg))
        assert "current_time" in info
        assert isinstance(info["current_time"], float)

    def test_step_current_time_advances(self, env, default_cfg):
        _, info0 = env.reset()
        _, _, _, _, info1 = env.step(_zero_actions(default_cfg))
        assert info1["current_time"] >= info0["current_time"]

    def test_multiple_steps_do_not_raise(self, env, default_cfg):
        env.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(_zero_actions(default_cfg))
            if terminated or truncated:
                env.reset()


# ---------------------------------------------------------------------------
# gymnasium API compliance
# ---------------------------------------------------------------------------

class TestGymnasiumCompliance:
    def test_spaces_contain_sample(self, env):
        obs, _ = env.reset()
        for i in range(obs.shape[0]):
            assert env.observation_space.contains(obs[i])

    def test_action_space_sample_is_valid_shape(self, env):
        sample = env.action_space.sample()
        assert sample.shape == (3,)
        assert sample.dtype == np.float64
