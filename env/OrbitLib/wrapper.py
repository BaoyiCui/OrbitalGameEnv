from __future__ import annotations

import os.path

import numpy as np
import datetime
from numpy.ctypeslib import ndpointer
from ctypes import (
    CDLL,
    c_bool,
    c_double,
    c_int,
    POINTER,
    Structure
)

# Array3 = ndpointer(dtype=np.float64, ndim=1, shape=(3,), flags=("C_CONTIGUOUS", "ALIGNED"))
# Array6 = ndpointer(dtype=np.float64, ndim=1, shape=(6,), flags=("C_CONTIGUOUS", "ALIGNED"))
# Mat3x3 = ndpointer(dtype=np.float64, ndim=2, shape=(3, 3), flags=("C_CONTIGUOUS", "ALIGNED"))
Array3 = c_double * 3
Array6 = c_double * 6
Mat3x3 = (c_double * 3) * 3


###
# 结构体映射
###
class HPOP_RVdata(Structure):
    _fields_ = [
        ("Year", c_int),
        ("Month", c_int),
        ("Day", c_int),
        ("Hour", c_int),
        ("Minute", c_int),
        ("Second", c_double),
        ("RJ2000", c_double * 3),  # J2000位置(m)
        ("VJ2000", c_double * 3),  # J2000速度(m/s)
    ]


class orbitStepOutput(Structure):
    _fields_ = [
        ('number', c_int),  # 第几个点
        ("Year", c_int),
        ("Month", c_int),
        ("Day", c_int),
        ("Hour", c_int),
        ("Minute", c_int),
        ("Second", c_double),
        ("RJ2000", c_double * 3),  # J2000位置(m)
        ("VJ2000", c_double * 3),  # J2000速度(m/s)
        ('lla', c_double * 3),  # 纬经高，deg,deg,m
    ]


class HPOP_In(Structure):
    _fields_ = [
        ('inial', c_bool),  # 是否初始化
        ('mass', c_double),  # 质量(kg)
        ('fuel', c_double),  # 燃料(kg)，新增
        ('thrust', c_double),  # 推力(N)，新增
        ('Isp', c_double),  # 比冲，新增
        ('Sd', c_double),  # 大气迎风面积(m^2)
        ('Sr', c_double),  # 光压反射面积(m^2)
        ('Cd', c_double),  # 大气阻尼系数
        ('eta', c_double),  # 表面反射率
        ('Propagator_Type', c_int),  # 积分器类型, 0 - RK78, 1 - RK56, 2 - 变步长龙格库塔, 3 - 定步长龙格库塔，10 - 二体动力学不采用积分
        ('Dyn_Type', c_int),  # 动力学类型, 0 - J2摄动, 1 - 所有摄动力（CForceModel中可具体设置引力场类型，是否考虑日月引力、太阳光呀、大气阻力、星历类型等等），3 - J4摄动
    ]


class OrbitLib:
    def __init__(self, so_path: str) -> None:
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Shared library not found: {so_path}")
        self.lib = CDLL(so_path)

        ###
        # 轨道递推
        ###
        self.lib.OrbitHPOP1.argtypes = [Array6, Array6, c_double, HPOP_In, Array6, Array6]
        self.lib.OrbitHPOP1.restype = None

        ###
        # 坐标变换
        ###
        self.lib.WGS842J2000_R.argtypes = [c_double, Array3, Array3]
        self.lib.WGS842J2000_R.restype = None

        self.lib.J20002WGS84_R.argtypes = [c_double, Array3, Array3]
        self.lib.J20002WGS84_R.restype = None

        self.lib.WGS842J2000_RV.argtypes = [c_double, Array3, Array3, Array3, Array3]
        self.lib.WGS842J2000_RV.restype = None

        self.lib.J20002WGS84_RV.argtypes = [c_double, Array3, Array3, Array3, Array3]
        self.lib.J20002WGS84_RV.restype = None

        self.lib.J20002LLA.argtypes = [c_double, Array3, Array3]
        self.lib.J20002LLA.restype = None

        self.lib.LLA2J2000.argtypes = [c_double, Array3, Array3]
        self.lib.LLA2J2000.restype = None

        self.lib.WGS842LLA.argtypes = [Array3, Array3]
        self.lib.WGS842LLA.restype = None

        self.lib.LLA2WGS84.argtypes = [Array3, Array3]
        self.lib.LLA2WGS84.restype = None

        ###
        # 坐标变换矩阵
        ###
        self.lib.DCM_J2000_WGS84.argtypes = [c_double, Mat3x3]
        self.lib.DCM_J2000_WGS84.restype = None

        self.lib.DCM_WGS84_J2000.argtypes = [c_double, Mat3x3]
        self.lib.DCM_WGS84_J2000.restype = None

        self.lib.DCM_LVLH_to_J2000.argtypes = [Array6, Mat3x3]
        self.lib.DCM_LVLH_to_J2000.restype = None

        self.lib.DCM_J2000_to_LVLH.argtypes = [Array6, Mat3x3]
        self.lib.DCM_J2000_to_LVLH.restype = None

        ###
        # 轨道根数 <-> RV
        ###
        self.lib.COE2RV.argtypes = [POINTER(c_double), POINTER(c_double)]  # rv[], coe[]
        self.lib.COE2RV.restype = None
        self.lib.RV2COE.argtypes = [POINTER(c_double), POINTER(c_double)]
        self.lib.RV2COE.restype = None

        ###
        # 时间转换
        ###
        self.lib.JulianDay.argtypes = [Array6]
        self.lib.JulianDay.restype = c_double

        self.lib.JulianDay_Modified.argtypes = [Array6]
        self.lib.JulianDay_Modified.restype = c_double

        self.lib.UTC_TDT.argtypes = [c_double]
        self.lib.UTC_TDT.restype = c_double

        self.lib.TDT_UTC.argtypes = [c_double]
        self.lib.TDT_UTC.restype = c_double

        self.lib.MJD_UTC.argtypes = [c_double, Array6]
        self.lib.MJD_UTC.restype = None

        self.lib.RAD2DEG.argtypes = [c_double]
        self.lib.RAD2DEG.restype = c_double

        self.lib.DEG2RAD.argtypes = [c_double]
        self.lib.DEG2RAD.restype = c_double

    def orbit_hpop(
            self,
            utc: datetime.datetime,
            rv_j2000: np.ndarray,
            h: float,
            hpop_in: HPOP_In
    ):
        utc_in = Array6(*(utc.year, utc.month, utc.day, utc.hour, utc.minute, utc.second))
        rv_j2000_in = Array6(*rv_j2000)
        h_in = c_double(h)
        utc_out = Array6()
        rv_j2000_out = Array6()

        self.lib.OrbitHPOP1(
            utc_in,
            rv_j2000_in,
            h_in,
            hpop_in,
            utc_out,
            rv_j2000_out,
        )

        return utc + datetime.timedelta(seconds=h), np.array(rv_j2000_out)

    def coe2rv(self, coe):
        coe_in = Array6(*coe)
        rvj2000_out = Array6()
        self.lib.COE2RV(rvj2000_out, coe_in)
        return np.array(rvj2000_out)

    def rv2coe(self, rv_j2000):
        rv_j2000_in = Array6(*rv_j2000)
        coe_out = Array6()
        self.lib.RV2COE(rv_j2000_in, coe_out)
        return np.array(coe_out)



