
#ifndef LAMBERT_SOLVER_H
#define LAMBERT_SOLVER_H

#include <vector>

namespace astrodynamics {

/**
 * @brief 求解兰伯特问题 (Lambert's Problem)。
 * 
 * 该函数计算在给定的中心天体引力下，从一个位置矢量 r1 到达另一个位置矢量 r2 所需的初始速度矢量 v1。
 * 算法基于普适变量 (Universal Variables) 和牛顿迭代法。
 *
 * @param r1_vec 初始位置矢量 [x, y, z] (单位: m)。
 * @param r2_vec 最终位置矢量 [x, y, z] (单位: m)。
 * @param transfer_time 转移时间 (单位: s)。
 * @param mu 中心天体的标准引力常数 (例如，地球的 mu 约为 3.986004418e14 m^3/s^2)。
 * @param is_prograde 轨道方向是否为顺行 (true) 或逆行 (false)。通常为true。
 * @param max_iter 最大迭代次数。
 * @param tolerance 求解器的收敛容差。
 * @return std::vector<double> 计算出的初始速度矢量 v1 [vx, vy, vz] (单位: m/s)。
 * @throws std::runtime_error 如果在指定迭代次数内未收敛。
 */
std::vector<double> solve_lambert(
    const std::vector<double>& r1_vec,
    const std::vector<double>& r2_vec,
    double transfer_time,
    double mu,
    bool is_prograde = true,
    int max_iter = 100,
    double tolerance = 1e-8
);

} // namespace astrodynamics

#endif // LAMBERT_SOLVER_H
