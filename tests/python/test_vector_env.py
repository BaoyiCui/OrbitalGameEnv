"""Integration tests for OGEVectorInterface (C++ binding) and OGEVectorEnv (Python wrapper)."""
from __future__ import annotations

import numpy as np
import pytest

import oge_py
from oge_py.vector_env import OGEVectorEnv, OGEVectorEnvCfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_PURSUERS = 4
NUM_EVADERS = 1
NUM_AGENTS = NUM_PURSUERS + NUM_EVADERS
NUM_ENVS = 4
BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def iface() -> oge_py.OGEVectorInterface:
    """A real OGEVectorInterface with 4 envs."""
    settings = oge_py.OGESettings()
    settings.set_int("num_pursuers", NUM_PURSUERS)
    return oge_py.OGEVectorInterface(
        NUM_ENVS,
        BATCH_SIZE,
        0,   # num_threads: auto
        -1,  # thread_affinity_offset: disabled
        "NextStep",
        settings,
    )


@pytest.fixture(scope="module")
def obs_size(iface: oge_py.OGEVectorInterface) -> int:
    """Per-agent observation size, resolved from the C++ binding."""
    _, size = iface.get_single_observation_size()
    return size


@pytest.fixture
def cfg() -> OGEVectorEnvCfg:
    return OGEVectorEnvCfg(
        num_pursuers=NUM_PURSUERS,
        num_evaders=NUM_EVADERS,
        num_envs=NUM_ENVS,
        batch_size=BATCH_SIZE,
        num_threads=0,
        thread_affinity_offset=-1,
        autoreset_mode="NextStep",
    )


@pytest.fixture
def env(cfg: OGEVectorEnvCfg) -> OGEVectorEnv:
    return OGEVectorEnv(cfg)


def _zero_actions() -> np.ndarray:
    """Zero actions with shape (BATCH_SIZE, NUM_AGENTS, 3)."""
    return np.zeros((BATCH_SIZE, NUM_AGENTS, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# OGEVectorInterface — construction
# ---------------------------------------------------------------------------

class TestVectorInterfaceConstruction:
    def test_get_num_envs(self, iface):
        assert iface.get_num_envs() == NUM_ENVS

    def test_get_observation_shape_is_two_tuple(self, iface):
        shape = iface.get_single_observation_size()
        assert isinstance(shape, tuple)
        assert len(shape) == 2

    def test_observation_shape_num_agents(self, iface):
        num_agents, _ = iface.get_single_observation_size()
        assert num_agents == NUM_AGENTS

    def test_observation_shape_obs_size_positive(self, iface):
        _, obs_size = iface.get_single_observation_size()
        assert obs_size > 0


# ---------------------------------------------------------------------------
# OGEVectorInterface — reset()
# ---------------------------------------------------------------------------

class TestVectorInterfaceReset:
    @pytest.fixture(autouse=True)
    def do_reset(self, iface):
        indices = list(range(NUM_ENVS))
        seeds = [-1] * NUM_ENVS
        self.result = iface.reset(indices, seeds)

    def test_reset_returns_two_tuple(self):
        assert isinstance(self.result, tuple) and len(self.result) == 2

    def test_reset_obs_is_ndarray(self):
        obs, _ = self.result
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_ndim(self):
        obs, _ = self.result
        assert obs.ndim == 3

    def test_reset_obs_shape_batch(self):
        obs, _ = self.result
        assert obs.shape[0] == BATCH_SIZE

    def test_reset_obs_shape_agents(self):
        obs, _ = self.result
        assert obs.shape[1] == NUM_AGENTS

    def test_reset_obs_dtype(self):
        obs, _ = self.result
        assert obs.dtype == np.float64

    def test_reset_info_contains_env_id(self):
        _, info = self.result
        assert "env_id" in info

    def test_reset_info_env_id_shape(self):
        _, info = self.result
        assert info["env_id"].shape == (BATCH_SIZE,)

    def test_reset_info_env_ids_cover_all_envs(self):
        _, info = self.result
        assert set(info["env_id"].tolist()) == set(range(NUM_ENVS))


# ---------------------------------------------------------------------------
# OGEVectorInterface — send() / recv()
# ---------------------------------------------------------------------------

class TestVectorInterfaceSendRecv:
    def test_recv_returns_five_tuple(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        result = iface.recv()
        assert isinstance(result, tuple) and len(result) == 5

    def test_recv_obs_shape(self, iface, obs_size):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        obs, _, _, _, _ = iface.recv()
        assert obs.shape == (BATCH_SIZE, NUM_AGENTS, obs_size)

    def test_recv_rewards_shape(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, rewards, _, _, _ = iface.recv()
        assert rewards.shape == (BATCH_SIZE, NUM_AGENTS)

    def test_recv_terminated_shape(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, terminated, _, _ = iface.recv()
        assert terminated.shape == (BATCH_SIZE,)

    def test_recv_truncated_shape(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, _, truncated, _ = iface.recv()
        assert truncated.shape == (BATCH_SIZE,)

    def test_recv_info_contains_env_id(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, _, _, info = iface.recv()
        assert "env_id" in info

    def test_recv_info_contains_current_times(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, _, _, info = iface.recv()
        assert "current_times" in info

    def test_recv_current_times_shape(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, _, _, info = iface.recv()
        assert info["current_times"].shape == (BATCH_SIZE,)

    def test_current_time_advances_after_step(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        iface.send(_zero_actions())
        _, _, _, _, info0 = iface.recv()
        iface.send(_zero_actions())
        _, _, _, _, info1 = iface.recv()
        assert np.all(info1["current_times"] >= info0["current_times"])

    def test_multiple_steps_do_not_raise(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        for _ in range(5):
            iface.send(_zero_actions())
            iface.recv()

    def test_send_wrong_batch_size_raises(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        bad_actions = np.zeros((BATCH_SIZE + 1, NUM_AGENTS, 3), dtype=np.float64)
        with pytest.raises(Exception):
            iface.send(bad_actions)

    def test_send_wrong_action_dim_raises(self, iface):
        iface.reset(list(range(NUM_ENVS)), [-1] * NUM_ENVS)
        bad_actions = np.zeros((BATCH_SIZE, NUM_AGENTS, 2), dtype=np.float64)
        with pytest.raises(Exception):
            iface.send(bad_actions)


# ---------------------------------------------------------------------------
# OGEVectorEnv — construction
# ---------------------------------------------------------------------------

class TestVectorEnvConstruction:
    def test_single_observation_space_shape_positive(self, env):
        assert env.single_observation_space.shape[0] > 0

    def test_single_observation_space_dtype(self, env):
        assert env.single_observation_space.dtype == np.float64

    def test_single_action_space_shape(self, env):
        assert env.single_action_space.shape == (3,)

    def test_single_action_space_dtype(self, env):
        assert env.single_action_space.dtype == np.float64

    def test_observation_space_batch_dim(self, env):
        assert env.observation_space.shape[0] == BATCH_SIZE

    def test_action_space_batch_dim(self, env):
        assert env.action_space.shape[0] == BATCH_SIZE


# ---------------------------------------------------------------------------
# OGEVectorEnv — reset()
# ---------------------------------------------------------------------------

class TestVectorEnvReset:
    def test_reset_returns_two_tuple(self, env):
        result = env.reset()
        assert isinstance(result, tuple) and len(result) == 2

    def test_reset_obs_is_ndarray(self, env):
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_obs_batch_dim(self, env):
        obs, _ = env.reset()
        assert obs.shape[0] == BATCH_SIZE

    def test_reset_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float64

    def test_reset_with_int_seed_does_not_raise(self, env):
        env.reset(seed=42)

    def test_reset_with_ndarray_seed_does_not_raise(self, env):
        env.reset(seed=np.array([10, 20, 30, 40]))

    def test_reset_with_invalid_seed_raises(self, env):
        with pytest.raises(TypeError):
            env.reset(seed="bad")

    def test_reset_mask_raises(self, env):
        with pytest.raises(ValueError):
            env.reset(options={"reset_mask": np.array([True, False, True, False], dtype=np.bool_)})

    def test_reset_obs_shape_stable_across_calls(self, env):
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        assert obs1.shape == obs2.shape


# ---------------------------------------------------------------------------
# OGEVectorEnv — step()
# ---------------------------------------------------------------------------

class TestVectorEnvStep:
    def test_step_returns_five_tuple(self, env):
        env.reset()
        result = env.step(_zero_actions())
        assert isinstance(result, tuple) and len(result) == 5

    def test_step_obs_batch_dim(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(_zero_actions())
        assert obs.shape[0] == BATCH_SIZE

    def test_step_rewards_batch_dim(self, env):
        env.reset()
        _, rewards, _, _, _ = env.step(_zero_actions())
        assert rewards.shape[0] == BATCH_SIZE

    def test_step_terminated_shape(self, env):
        env.reset()
        _, _, terminated, _, _ = env.step(_zero_actions())
        assert terminated.shape == (BATCH_SIZE,)

    def test_step_truncated_shape(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(_zero_actions())
        assert truncated.shape == (BATCH_SIZE,)

    def test_step_info_contains_current_times(self, env):
        env.reset()
        _, _, _, _, info = env.step(_zero_actions())
        assert "current_times" in info

    def test_multiple_steps_do_not_raise(self, env):
        env.reset()
        for _ in range(10):
            env.step(_zero_actions())


# ---------------------------------------------------------------------------
# OGEVectorEnv — send() / recv()
# ---------------------------------------------------------------------------

class TestVectorEnvSendRecv:
    def test_send_then_recv_returns_five_tuple(self, env):
        env.reset()
        env.send(_zero_actions())
        result = env.recv()
        assert isinstance(result, tuple) and len(result) == 5

    def test_send_returns_none(self, env):
        env.reset()
        result = env.send(_zero_actions())
        assert result is None