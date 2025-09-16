import gymnasium as gym
import argparse

from env.OrbitLib import OrbitLib
import numpy as np
from env import orbitx


# def main():
#     parser = argparse.ArgumentParser(description="PE Env")
#     parser.add_argument("--render-mode", default=None)
#     parser.add_argument("--seed", default=42)
#
#     args = parser.parse_args()
#
#     env = gym.make("Orbital-Pursuit-Evasion-v0")

def main():
    sma = 42166.0  # 轨道半长轴, km
    ecc = 0.0  # 偏心率，无量纲
    inc = 0.0  # 轨道倾角，度
    argp = 0.0  # 近地点幅角，度
    raan = 0.0  # 升交点赤经，度
    # ta = np.random.uniform(0.0, 360.0)  # 真近点角，度
    ta = 0

    orbit_lib = OrbitLib("/home/baoyicui/Workspaces/OrbitalGameEnv/env/OrbitLib/so/X86/libOrbit.so")

    rv = orbitx.coe2rv(np.array([sma, ecc, inc, raan, argp, ta]))


    

    pass


if __name__ == "__main__":
    main()
