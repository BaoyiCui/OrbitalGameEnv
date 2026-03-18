import numpy as np
import pytest
import oge_py


def test_oge_version(oge_module):
    assert hasattr(oge_module, "__version__")
    assert isinstance(oge_module.__version__, str)
    assert oge_module.__version__


def test_oge_construction(oge, oge_module):
    assert isinstance(oge, oge_module.OGEInterface)


def test_settings_property_available(oge, oge_module):
    assert isinstance(oge.get_settings(), oge_module.OGESettings)


def test_set_get_int_roundtrip(oge):
    oge.setInt("num_pursuers", 3)
    assert oge.getInt("num_pursuers") == 3


def test_set_get_float_roundtrip(oge):
    oge.setFloat("terminal_time", 600.0)
    assert oge.getFloat("terminal_time") == pytest.approx(600.0)


def test_get_unknown_key_raises(oge):
    with pytest.raises(IndexError):
        oge.getInt("__missing_int_key__")
    with pytest.raises(IndexError):
        oge.getFloat("__missing_float_key__")
    with pytest.raises(IndexError):
        oge.getBool("__missing_bool_key__")
    with pytest.raises(IndexError):
        oge.getString("__missing_str_key__")


def test_act_rejects_invalid_action_shape(oge):
    bad_actions = np.zeros((3,), dtype=np.float64)
    with pytest.raises(RuntimeError, match=r"shape \(num_agents, 3\)"):
        oge.act(bad_actions)


def test_get_rewards_rejects_invalid_ndim(oge):
    bad_actions = np.zeros((1, 3, 1), dtype=np.float64)
    with pytest.raises(RuntimeError, match=r"two dimensions"):
        oge.get_rewards(bad_actions)


def test_get_rewards_rejects_invalid_action_width(oge):
    bad_actions = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(RuntimeError, match=r"shape \(num_agents, 3\)"):
        oge.get_rewards(bad_actions)


def test_settings_validate_rejects_bad_num_pursuers(oge_module):
    settings = oge_module.OGESettings()
    settings.set_int("num_pursuers", 0)
    with pytest.raises(ValueError, match=r"num_pursuers"):
        settings.validate()


def test_interface_init_rejects_invalid_configuration(oge):
    oge.setInt("num_evaders", 2)
    with pytest.raises(ValueError, match=r"num_evaders"):
        oge.init()


def test_sat_state_repr(oge_module):
    sat_state = oge_module.SatState()
    text = repr(sat_state)
    assert isinstance(text, str)
    assert text


@pytest.fixture
def oge_ready():
    """An initialized OGEInterface with default settings (1 evader, 4 pursuers)."""
    iface = oge_py.OGEInterface()
    iface.init()
    iface.reset()
    return iface


def test_get_sat_states_returns_dict(oge_ready):
    states = oge_ready.get_sat_states()
    assert isinstance(states, dict)


def test_get_sat_states_key_count(oge_ready):
    states = oge_ready.get_sat_states()
    num_agents = oge_ready.getInt("num_evaders") + oge_ready.getInt("num_pursuers")
    assert len(states) == num_agents


def test_get_sat_states_values_are_sat_state(oge_ready, oge_module):
    states = oge_ready.get_sat_states()
    for state in states.values():
        assert isinstance(state, oge_module.SatState)


def test_get_sat_states_fields(oge_ready):
    states = oge_ready.get_sat_states()
    for state in states.values():
        assert state.r_j2000.shape == (3,)
        assert state.v_j2000.shape == (3,)
        assert isinstance(state.dv_remain, float)
        assert isinstance(state.is_alive, bool)


def test_reset_with_states_restores_state(oge_ready):
    # capture state after first reset
    original = oge_ready.get_sat_states()

    # advance one step to change state
    num_agents = oge_ready.getInt("num_evaders") + oge_ready.getInt("num_pursuers")
    actions = np.zeros((num_agents, 3), dtype=np.float64)
    oge_ready.act(actions)

    # reset back to captured state
    oge_ready.reset_with_states(original)
    restored = oge_ready.get_sat_states()

    for name, state in original.items():
        assert np.allclose(restored[name].r_j2000, state.r_j2000)
        assert np.allclose(restored[name].v_j2000, state.v_j2000)
        assert restored[name].dv_remain == pytest.approx(state.dv_remain)
        assert restored[name].is_alive == state.is_alive


def test_reset_with_states_resets_time(oge_ready):
    original = oge_ready.get_sat_states()
    num_agents = oge_ready.getInt("num_evaders") + oge_ready.getInt("num_pursuers")
    actions = np.zeros((num_agents, 3), dtype=np.float64)
    oge_ready.act(actions)
    assert oge_ready.get_current_time() > 0.0

    oge_ready.reset_with_states(original)
    assert oge_ready.get_current_time() == pytest.approx(0.0)
