import numpy as np

from scripts.env.pe_env import PEEnv


def main():
    env = PEEnv()
    env.reset()

    for i in range(10000):
        actions = {
            "p_0": np.zeros(3, ),
            "e_0": np.zeros(3, )
        }
        observations, rewards, terminations, truncations, _ = env.step(actions)

        print(rewards)

        env.render()

        if any(terminations.values()) or any(truncations.values()):
            print("reset")
            env.reset()


if __name__ == "__main__":
    main()
