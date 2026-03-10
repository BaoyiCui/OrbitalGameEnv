import numpy as np
import pytest


def test_oge_version(oge_module):
    assert hasattr(oge_module, "__version__")
    assert isinstance(oge_module.__version__, str)
    assert oge_module.__version__


def test_oge_construction(oge, oge_module):
    assert isinstance(oge, oge_module.OGEInterface)


def test_settings_property_available(oge, oge_module):
    assert isinstance(oge.settings, oge_module.OGESettings)


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
