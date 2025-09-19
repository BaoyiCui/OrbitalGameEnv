import time

import numpy as np

from env.pe_env import PEEnv


def main():
    env = PEEnv()
    env.reset()

    for i in range(100):
        actions = {
            "p_0": np.zeros(3, ),
            "e_0": np.zeros(3, )
        }
        observations, rewards, terminations, truncations, _ = env.step(actions)

        env.render()
        time.sleep(0.1)

        print(i)

        if any(terminations.values()) or any(truncations.values()):
            print("reset")
            env.reset()


if __name__ == "__main__":
    main()
