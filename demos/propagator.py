import numpy as np
import datetime

from env import orbitx
from env.OrbitLib import OrbitLib, HPOP_In


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
    # HPOP初始化参数
    # hpopin = HPOP_In(  # HPOP 初始化参数，全局变量
    #     inial=True,
    #     mass=50,
    #     fuel=20,
    #     thrust=0.0,
    #     Isp=20.0,
    #     Sd=1.0,
    #     Sr=1.0,
    #     Cd=2.2,
    #     eta=1.0,
    #     Propagator_Type=0,  # RK78
    #     Dyn_Type=0  # J2摄动
    # )
    hpopin = HPOP_In(  # HPOP 初始化参数，全局变量
        inial=True,
        mass=50,
        fuel=20,
        thrust=0.0,
        Isp=20.0,
        Sd=1.0,
        Sr=1.0,
        Cd=2.2,
        eta=1.0,
        Propagator_Type=10,  # 二体动力学
        Dyn_Type=0  # 无效，J2摄动
    )

    rv_f = orbitx.rv_from_r0v0(rv, 600.0)  # orbitx中使用的单位是 km, km/s, s
    utc = datetime.datetime(2030, 1, 1, 0, 0, 0)

    utc_f, rv_f_ = orbit_lib.orbit_hpop(
        utc,
        rv_j2000=rv * 1e3,  # orbit_lib中使用的单位是 m, m/s, s
        h=600.0,
        hpop_in=hpopin
    )

    print(f'{rv_f}')
    print(f'{rv_f_}')
    print(f'{utc}, {utc_f}')


if __name__ == '__main__':
    main()
