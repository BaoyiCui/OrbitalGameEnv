from env.pe_env import PEEnv, PEEnvCfg


def train(env_fn, steps=1e4, seed:int | None=0, **env_kwargs ):
    env = env_fn.parallel_env(**env_kwargs)

    env = PEEnv()

    print(f'Starting training on {env.metadata['name']}')

    model = PPO()


if __name__ == '__main__':
    main()
