import oge_py
from oge_py.env import OGEEnv, OGEEnvCfg

if __name__ == "__main__":
    cfg = OGEEnvCfg()
    env = OGEEnv(cfg)

    print(env.cfg.num_pursuers)
    print(env.cfg.num_evaders)
    obs, _ = env.reset()

    print(obs)
