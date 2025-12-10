#ifndef ORBITBASICFUNC_H
#define ORBITBASICFUNC_H
#pragma once
//#include"Matrix.h"
#include<vector>

namespace ORBITBASICFUNC
{

//一段时间内HPOP轨道递推数据结构体
struct HPOP_RVdata
{
	int Year, Month, Day, Hour, Minute; //年，月，日，时，分
	double Second;					//秒
	double RJ2000[3];//J2000位置(m)
	double VJ2000[3];//J2000速度(m/s)
};

//orbitStep轨道递推函数输出
struct orbitStepOutput
{
	int number;//第几个点
	int Year, Month, Day, Hour, Minute; //年，月，日，时，分
	double Second;					//秒
	double RJ2000[3];//J2000位置(m)
	double VJ2000[3];//J2000速度(m/s)
	double lla[3];//纬经高，deg,deg,m
};

//HPOP初始化参数
struct HPOP_In
{
	bool inial;								//是否初始化
	double mass;							//质量(kg)
	double fuel;							//燃料(kg)，新增
	double thrust;							//推力(N)，新增
	double Isp;								//比冲，新增
	double Sd;								//大气迎风面积(m^2)
	double Sr;								//光压反射面积(m^2)
	double Cd;								//大气阻尼系数
	double eta;								//表面反射率
	int    Propagator_Type;					//积分器类型, 0-RK78, 1-RK56, 2-变步长龙格库塔, 3-定步长龙格库塔，10-二体动力学不采用积分
	int    Dyn_Type;						//动力学类型, 0-J2摄动, 1-所有摄动力（CForceModel中可具体设置引力场类型，是否考虑日月引力、太阳光呀、大气阻力、星历类型等等），3-J4摄动
};
//脉冲序列
struct DeltaV_In
{
	double t;															//脉冲时刻(s)
	double LVLH_dvx;													//LVLH坐标系x方向速度脉冲(m/s)
	double LVLH_dvy;													//LVLH坐标系y方向速度脉冲(m/s)
	double LVLH_dvz;													//LVLH坐标系z方向速度脉冲(m/s)
};

/********************************************************************************************
* @brief     HPOP单步递推
* @param[in]  double UTC[6]											    // 仿真开始时刻UTC时间
* @param[in]  double RVj2000[6]											// 当前j2000位置速度，m，m/s，6个
* @param[in]  double h      											// 当前仿真步长(s)
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  double UTC_[6]							                // 单步递推后的UTC时间
* @param[out]  double RVj2000_[6]										// 单步递推后的j2000位置速度，m，m/s
* @return	  void
*********************************************************************************************/
extern "C" void OrbitHPOP1(double UTC[6], double RVj2000[6], double h, HPOP_In HPOPIN, double UTC_[6], double RVj2000_[6]);
/********************************************************************************************
* @brief     HPOP单步递推
* @param[in]  double UTC[6]											    // 仿真开始时刻UTC时间
* @param[in]  double RVj2000[6]											// 当前j2000位置速度，m，m/s，6个
* @param[in]  double h      											// 当前仿真步长(s)
* @param[in]  double ax      											// J2000x方向加速度(m/s2)
* @param[in]  double ay      											// J2000y方向加速度(m/s2)
* @param[in]  double az      											// J2000z方向加速度(m/s2)
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  double UTC_[6]							                // 单步递推后的UTC时间
* @param[out]  double RVj2000_[6]										// 单步递推后的j2000位置速度，m，m/s
* @return	  void
*********************************************************************************************/
extern "C" void OrbitHPOP2(double UTC[6], double RVj2000[6], double h, double ax, double ay, double az, HPOP_In HPOPIN, double UTC_[6], double RVj2000_[6]);
/********************************************************************************************
* @brief     HPOP单步递推
* @param[in]  double UTC[6]											    // 仿真开始时刻UTC时间
* @param[in]  double RVj2000[6]											// 当前j2000位置速度，m，m/s，6个
* @param[in]  double h      											// 当前仿真步长(s)
* @param[in]  double phi      											// 滚动角(弧度)
* @param[in]  double theta      										// 俯仰角(弧度)
* @param[in]  double psi      											// 偏航角(弧度)
* @param[in]  double axl      											// 本体系x方向加速度(m/s2)
* @param[in]  double ayl      											// 本体系y方向加速度(m/s2)
* @param[in]  double azl      											// 本体系z方向加速度(m/s2)
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  double UTC_[6]							                // 单步递推后的UTC时间
* @param[out]  double RVj2000_[6]										// 单步递推后的j2000位置速度，m，m/s
* @return	  void
*********************************************************************************************/
extern "C" void OrbitHPOP3(double UTC[6], double RVj2000[6], double h, double phi, double theta, double psi, double axl, double ayl, double azl, HPOP_In HPOPIN, double UTC_[6], double RVj2000_[6]);
/********************************************************************************************
* @brief     HPOP一段时间递推
* @param[in]  double UTC0[6]											// 仿真开始时刻UTC时间
* @param[in]  double RVj2000_0[6]										// 初始j2000位置速度，m，m/s，6个
* @param[in]  double h      											// 仿真步长(s)
* @param[in]  double tmax							                    // 仿真时长(s)
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  vector<HPOP_RVdata> RVJ2000								// 每一步j2000位置速度，m，m/s
* @return	  void
*********************************************************************************************/
extern "C" void OrbitHPOP4(double UTC0[6], double RVj2000_0[6], double h, double tmax, HPOP_In HPOPIN, std::vector<HPOP_RVdata>& RVJ2000);
/********************************************************************************************
* @brief     HPOP一段时间递推
* @param[in]  double UTC0[6]											// 仿真开始时刻UTC时间
* @param[in]  double RVj2000_0[6]										// 初始j2000位置速度，m，m/s，6个
* @param[in]  double h      											// 仿真步长(s)
* @param[in]  double tmax							                    // 仿真时长(s)
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  double* RVJ2000											// HPOP_RVdata结构体数组
* @return	  void
*********************************************************************************************/
extern "C" void OrbitHPOP5(double UTC0[6], double RVj2000_0[6], double h, double tmax, HPOP_In HPOPIN, HPOP_RVdata* RVJ2000);

/********************************************************************************************
* @brief     轨道递推，步数+步长输入
* @param[in]  double UTC0[6]											// 仿真开始时刻UTC时间
* @param[in]  double RVj2000begin[6]									// 初始j2000位置速度，m，m/s，6个
* @param[in]  double step      											// 仿真步长(s)
* @param[in]  int stepNumber							                // 递推步数
* @param[in]  HPOP_In HPOPIN      										// 初始化结构体
* @param[out]  orbitStepOutput* RVJ2000out								// HPOP_RVdata结构体数组
* @return	  void
*********************************************************************************************/
extern "C" void orbitStep(double UTC0[6], double RVj2000begin[6], double step, int stepNumber, HPOP_In HPOPIN, orbitStepOutput* RVJ2000out);

/********************************************************************************************
* @brief     计算太阳光、月光、地气光入射角
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double RVSat_J2000[3]										//飞行器位置, m
* @param[in]  double pointing_J2000[3]									//飞行器指向矢量
* @param[out] double& sunAngle											//阳光入射角, rad
* @param[out] double& moonAngle											//月光入射角, rad
* @param[out] double& earthAngle										//地气光入射角, rad
* @return	  void
*********************************************************************************************/
extern "C" void CalculateLightAngle(const double TDT, const double RSat_J2000[3], const double pointing_J2000[3], double& sunAngle, double& moonAngle, double& earthAngle);
/********************************************************************************************
* @brief     计算太阳光、月光、地气光入射角
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double RVSat_J2000[3]										//飞行器位置, m
* @param[in]  double TSat_J2000[3]										//目标星位置，m
* @param[out] double& sunAngle											//阳光入射角, rad
* @param[out] double& moonAngle											//月光入射角, rad
* @param[out] double& earthAngle										//地气光入射角, rad
* @return	  void
*********************************************************************************************/
extern "C" void CalculateLightAngle_Tsat(const double TDT, const double RSat_J2000[3], const double TSat_J2000[3], double& sunAngle, double& moonAngle, double& earthAngle);


/********************************************************************************************
* @brief     WGS84->J2000位置转换
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double R_WGS84[3]										    //WGS84位置, m
* @param[out] double R_J2000[3]											//J2000位置，m
* @return	  void
*********************************************************************************************/
extern "C" void WGS842J2000_R(double TDT, double R_WGS84[3], double R_J2000[3]);

/********************************************************************************************
* @brief     J2000->WGS84位置转换
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double R_J2000[3]										    //J2000位置, m
* @param[out] double R_WGS84[3]											//WGS84位置，m
* @return	  void
*********************************************************************************************/
extern "C" void J20002WGS84_R(double TDT, double R_J2000[3], double R_WGS84[3]);

/********************************************************************************************
* @brief     WGS84->J2000位置、速度转换
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double R_WGS84[3]										    //WGS84位置, m
* @param[in]  double V_WGS84[3]										    //WGS84速度, m/s
* @param[out] double R_J2000[3]											//J2000位置，m
* @param[out] double V_J2000[3]											//J2000位置，m/s
* @return	  void
*********************************************************************************************/
extern "C" void WGS842J2000_RV(double TDT, double R_WGS84[3], double V_WGS84[3],
	double R_J2000[3], double V_J2000[3]);

/********************************************************************************************
* @brief     J2000->WGS84位置、速度转换
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double R_J2000[3]										    //J2000位置, m
* @param[in]  double V_J2000[3]										    //J2000速度, m/s
* @param[out] double R_WGS84[3]											//WGS84位置，m
* @param[out] double V_WGS84[3]											//WGS84位置，m/s
* @return	  void
*********************************************************************************************/
extern "C" void J20002WGS84_RV(double TDT, double R_J2000[3], double V_J2000[3],
	double R_WGS84[3], double V_WGS84[3]);

/********************************************************************************************
* @brief     计算卫星经度漂移率  deg/day
* @param[in]  double six[6]											    //当前六根数
* @param[out] double lonrate											//经度漂移率，deg/day
* @return	  void
*********************************************************************************************/
extern "C" void LonRateCal(double six[6], double& lonrate);

/********************************************************************************************
* @brief     是否在阴影区
* @param[in]  double TDT											    //当前时刻TDT时间
* @param[in]  double RV_J2000[6]										//卫星J2000位置,速度, m、m/s
* @param[in]  int type													//太阳光模式，0-平行光  1-放射光
* @param[out] bool& Is_Umbra											//在阴影区ture，在光照区false
* @return	  void
*********************************************************************************************/
extern "C" void Umbra(double TDT, double RV_J2000[6], int type, bool& Is_Umbra);

/********************************************************************************************
* @brief     计算任务时间段内卫星在阴影区的时间段
* @param[in]  double UTC_start[6]									    //任务起始时刻
* @param[in]  double UTC_end[6]									        //任务终止时刻
* @param[in]  double COE[6]										        //卫星初始轨道根数，弧度
* @param[in]  double step                                               //递推步长，s
* @param[in]  int type													//太阳光模式，0-平行光  1-放射光
* @param[out] OutputTime									            //在阴影区的时间,种类
* @return	  bool
*********************************************************************************************/

struct GapTime
{
	double UTC_start[6];
	double UTC_end[6];
};

struct ShadowTime
{
	std::vector<GapTime> shadowtime_earth;
	std::vector<GapTime> shadowtime_moon;
};

extern "C" bool UmbraTime(double UTC_start[6], double UTC_end[6], double COE[6], double UTC_sat[6], double step, int type, ShadowTime& OutputTime);

//计算J2000->WGS84坐标系的转换矩阵,输入为TDT儒略日
extern "C" void DCM_J2000_WGS84(double TDT,double DCM[3][3]);
//计算WGS84->J2000坐标系的转换矩阵,输入为TDT儒略日
extern "C" void DCM_WGS84_J2000(double TDT, double DCM[3][3]);//新增

//计算体系->J2000坐标系的转换矩阵,输入为轨道系姿态角，弧度、J2000位置速度，m、m/s
extern "C" void DCM_Body_to_J2000(const double Atti[3], const double RV[6], double DCM[3][3]);
//计算J2000->体系坐标系的转换矩阵,输入为轨道系姿态角，弧度、J2000位置速度，m、m/s
extern "C" void DCM_J2000_to_Body(const double Atti[3], const double RV[6], double DCM[3][3]);//新增

//计算LVLH->J2000坐标系的转换矩阵,输入为J2000位置速度，m、m/s
extern "C" void DCM_LVLH_to_J2000(const double RV[6], double DCM[3][3]);
//计算J2000->LVLH坐标系的转换矩阵,输入为J2000位置速度，m、m/s
extern "C" void DCM_J2000_to_LVLH(const double RV[6], double DCM[3][3]);//新增

//计算VVLH->J2000坐标系的转换矩阵,输入为J2000位置速度，m、m/s
extern "C" void DCM_VVLH_to_J2000(const double RV[6], double DCM[3][3]);
//计算J2000->VVLH坐标系的转换矩阵,输入为J2000位置速度，m、m/s
extern "C" void DCM_J2000_to_VVLH(const double RV[6], double DCM[3][3]);//新增

//计算本体系->VVLH坐标系的转换矩阵,输入为本体相对VVLH的三个姿态角，弧度
extern "C" void DCM_Body_to_VVLH(const double Atti[3], double DCM[3][3]);
//计算VVLH坐标系->本体系的转换矩阵,输入为本体相对VVLH的三个姿态角，弧度
extern "C" void DCM_VVLH_to_Body(const double Atti[3], double DCM[3][3]);//新增

//计算VVLH->LVLH坐标系的转换矩阵
extern "C" void DCM_VVLH_to_LVLH(double DCM[3][3]);
//计算LVLH->VVLH坐标系的转换矩阵
extern "C" void DCM_LVLH_to_VVLH(double DCM[3][3]);//新增

extern "C" void DCM_EastSouthDown_to_J2000(const double R[3], double DCM[3][3]);
extern "C" void DCM_J2000_to_EastSouthDown(const double R[3], double DCM[3][3]);//新增

extern "C" void DCM_Body_to_EastSouthDown(const double Atti[3], double DCM[3][3]);
extern "C" void DCM_EastSouthDown_to_Body(const double Atti[3], double DCM[3][3]);//新增

extern "C" void DCM_NorthEastDown_to_J2000(const double R[3], double DCM[3][3]);
extern "C" void DCM_J2000_to_NorthEastDown(const double R[3], double DCM[3][3]);//新增

//瞬根数转化为位置速度(弧度)
//输入：const double coe[]轨道根数，弧度
//输出：double rv[]J2000位置速度m、m/s
extern "C" void COE2RV(double rv[], const double coe[]);

//将卫星的位置、速度转换为轨道要素(弧度)
//输入：const double rv[]J2000位置速度m、m/s
//输出：double coe[]轨道根数，弧度
extern "C" void RV2COE(const double rv[], double coe[]);

//计算纬度精度高度LLA
//输入：double TDT      MJD格式TDT时间
//输入：double R_J2000[]R_J2000位置m
//输出：double LLA[3]维度、经度、高度，度、米
extern "C" void J20002LLA(double TDT, double R_J2000[3], double LLA[3]);//修改
//计算R_J2000位置
//输入：double TDT      MJD格式TDT时间
//输入：double LLA[3]维度、经度、高度，度、米
//输出：double R_J2000[]R_J2000位置m
extern "C" void LLA2J2000(double TDT, double LLA[3], double R_J2000[3]);//修改
//计算纬度精度高度LLA
//输入：const double R_WGS84[]WGS84位置m
//输出：double LLA[3]维度、经度、高度，度、米
extern "C" void WGS842LLA(double R_WGS84[3], double LLA[3]);//新增
//计算WGS84位置
//输入：double LLA[3]维度、经度、高度，度、米
//输出：double R_WGS84[3]WGS84位置m
extern "C" void LLA2WGS84(double LLA[3], double R_WGS84[3]);//新增

//计算真近点角(if flag ==0, Newton迭代, else 级数展开)
extern "C" double m2f(double e, double M, int flag = 0);

//计算平近点角
extern "C" double f2m(double f, double e);



//地球引力常数
extern "C" double Earth_Gravity_Constant();

//月球引力常数
extern "C" double Lunar_Gravity_Constant();

//太阳引力常数
extern "C" double Solar_Gravity_Constant();

//地球赤道半径
extern "C" double Earth_Equator_Radius();

//地球平均半径
extern "C" double Earth_Average_Radius();

//地球J2项系数
extern "C" double Earth_J2_Coefficient();

//地球J3项系数
extern "C" double Earth_J3_Coefficient();

//地球J4项系数
extern "C" double Earth_J4_Coefficient();

//地球自转角速率
extern "C" double omegaE();

//地球扁率
extern "C" double alphaE();

//判断是否椭圆轨道
extern "C" bool isellipseorbit(const double* rv);

//计算太阳在J2000系下的位置
extern "C" void SolarPosition_J2000(double TDT, double RS_J2000[3]);

//计算月球在J2000系下的位置
extern "C" void LunarPosition_J2000(double TDT, double RL_J2000[3]);




//年月日时分秒转化为儒略日
extern "C" double JulianDay(double DT[6]);

//年月日时分秒转化为MJD儒略日
extern "C" double JulianDay_Modified(double DT[6]);

//UTC时间转换为TDT时间. 输入: UTC MJD格式的UTC.
extern "C" double UTC_TDT(double UTC);

//TDT时间转换为MJD格式的UTC时间. 输入: TDT MJD格式的TDT.
extern "C" double TDT_UTC(double TDT);

//MJD时间转换为UTC时间. 输入: MJD时间.
extern "C" void MJD_UTC(double MJD, double UTC[6]);//新增

//弧度转化为角度
extern "C" double RAD2DEG(double Alpha);

//角度转化为弧度
extern "C" double DEG2RAD(double Alpha);

}
#endif
