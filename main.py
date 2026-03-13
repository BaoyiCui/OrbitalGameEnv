from oge_py import OGEVectorEnv, OGEVectorEnvCfg

if __name__ == "__main__":
    vec_env_cfg = OGEVectorEnvCfg(
        num_envs=4,
        num_evaders=1,
        num_pursuers=4,
        batch_size=0,
        num_threads=0,
        thread_affinity_offset=-1,
        autoreset_mode="NextStep"
    )
    vec_env = OGEVectorEnv(vec_env_cfg)

    obs, info = vec_env.reset()
    print("reset obs shape:", obs.shape)
