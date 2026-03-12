import numpy as np

from .utils import make_zero_actions


def test_init_then_get_observations_returns_1d_array(oge):
    oge.init()
    obs = oge.get_observations()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 2
    assert obs.size > 0


def test_get_rewards_output_matches_num_agents(oge):
    num_pursuers = 2
    num_evaders = 1
    oge.setInt("num_pursuers", num_pursuers)
    oge.setInt("num_evaders", num_evaders)
    oge.init()

    n_agents = num_pursuers + num_evaders
    rewards = oge.get_rewards(make_zero_actions(n_agents))

    assert isinstance(rewards, np.ndarray)
    assert rewards.ndim == 1
    assert rewards.shape[0] == n_agents


def test_act_and_reset_do_not_raise(oge):
    oge.init()
    n_agents = oge.getInt("num_pursuers") + oge.getInt("num_evaders")

    oge.act(make_zero_actions(n_agents))
    oge.reset()


def test_observation_shape_stable_across_reset(oge):
    oge.init()
    before = oge.get_observations()
    oge.reset()
    after = oge.get_observations()

    assert before.ndim == 2
    assert after.ndim == 2
    assert before.shape == after.shape
