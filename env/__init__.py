import gymnasium as gym

gym.register(
    id="Orbital-Pursuit-Evasion-v0",
    entry_point="pe_env:PEEnv",
    disable_env_checker=True,
    kwargs={

    }
)
