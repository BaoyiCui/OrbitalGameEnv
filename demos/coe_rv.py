import numpy as np
import datetime

from env import orbitx
from env.OrbitLib import OrbitLib


def main():
    sma = 42166.0  # 轨道半长轴, km
    ecc = 0.0  # 偏心率，无量纲
    inc = 45.0  # 轨道倾角，度
    argp = 0.0  # 近地点幅角，度
    raan = 30  # 升交点赤经，度
    # ta = np.random.uniform(0.0, 360.0)  # 真近点角，度
    ta = 0

    orbit_lib = OrbitLib("/home/baoyicui/Workspaces/OrbitalGameEnv/env/OrbitLib/so/X86/libOrbit.so")

    rv = orbitx.coe2rv(np.array([sma, ecc, inc, raan, argp, ta]))  # orbitx中使用的单位是 km, km/s, s

    coe_ = orbit_lib.rv2coe(rv * 1e3)  # orbit_lib中使用的单位是 m, m/s, s
    rv_ = orbit_lib.coe2rv(coe_)

    print(f'orbitx计算结果: {rv}')
    print(f'orbit_lib计算结果: {rv_}')


if __name__ == '__main__':
    main()
