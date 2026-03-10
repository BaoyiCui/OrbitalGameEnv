import numpy as np

def make_zero_actions(num_agents: int, *, dtype=np.float64) -> np.ndarray:
    """Build a valid action tensor with shape (num_agents, 3)."""
    return np.zeros((num_agents, 3), dtype=dtype)
