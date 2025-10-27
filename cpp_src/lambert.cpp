#include "lambert.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

// Define M_PI if not defined (e.g., in MSVC)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace astrodynamics {

// 内部辅助函数：计算矢量点积
double dot(const std::vector<double>& v1, const std::vector<double>& v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// 内部辅助函数：计算矢量模长
double norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

// Stumpff C2(z) 函数
double stumpff_c2(double z) {
    if (z > 1e-6) {
        return (1.0 - std::cos(std::sqrt(z))) / z;
    } else if (z < -1e-6) {
        return (std::cosh(std::sqrt(-z)) - 1.0) / (-z);
    } else {
        return 0.5 - z / 24.0 + z * z / 720.0;
    }
}

// Stumpff C3(z) 函数
double stumpff_c3(double z) {
    if (z > 1e-6) {
        return (std::sqrt(z) - std::sin(std::sqrt(z))) / (z * std::sqrt(z));
    } else if (z < -1e-6) {
        return (std::sinh(std::sqrt(-z)) - std::sqrt(-z)) / (-z * std::sqrt(-z));
    } else {
        return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0;
    }
}

std::vector<double> solve_lambert(
    const std::vector<double>& r1_vec,
    const std::vector<double>& r2_vec,
    double transfer_time,
    double mu,
    bool is_prograde,
    int max_iter,
    double tolerance
) {
    double r1_norm = norm(r1_vec);
    double r2_norm = norm(r2_vec);

    double d_theta = std::acos(dot(r1_vec, r2_vec) / (r1_norm * r2_norm));

    if (!is_prograde) {
        d_theta = 2.0 * M_PI - d_theta;
    }

    double A = std::sin(d_theta) * std::sqrt(r1_norm * r2_norm / (1.0 - std::cos(d_theta)));

    double z = 0.0; // 初始猜测值
    double y, C2, C3, F, G, F_dot;

    for (int i = 0; i < max_iter; ++i) {
        y = r1_norm + r2_norm + A * (z * stumpff_c3(z) - 1.0) / std::sqrt(stumpff_c2(z));
        if (y < 0) { // 确保y为正
            z += 0.1;
            continue;
        }

        C2 = stumpff_c2(z);
        C3 = stumpff_c3(z);

        double sqrt_y_div_mu = std::sqrt(y/mu);
        F = (sqrt_y_div_mu * y) * C3 + A * std::sqrt(y) - transfer_time;

        if (std::abs(F) < tolerance) {
            // 收敛成功
            double f = 1.0 - y / r1_norm * C2;
            double g = A * std::sqrt(y / mu);
            double g_dot = 1.0 - y / r2_norm * C2;

            std::vector<double> v1_vec(3);
            for (int j = 0; j < 3; ++j) {
                v1_vec[j] = (r2_vec[j] - f * r1_vec[j]) / g;
            }
            return v1_vec;
        }

        if (z == 0) {
            F_dot = (std::sqrt(2.0) / 40.0) * std::pow(y, 1.5) + (A / 8.0) * (std::sqrt(y) + A * std::sqrt(1.0 / (2.0 * y)));
        } else {
            F_dot = (std::pow(y/mu, 1.5) * (1.0 / (2.0 * z) * (C2 - 3.0 * C3 / (2.0 * C2))) + 
                    (A / 8.0) * (3.0 * C3 / C2 * std::sqrt(y) + A * std::sqrt(mu / y)));
        }

        z = z - F / F_dot; // 牛顿迭代
    }

    throw std::runtime_error("Lambert solver did not converge within the maximum number of iterations.");
}

} // namespace astrodynamics

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(lambert_solver, m) {
    m.doc() = "Lambert problem solver";
    m.def("solve_lambert", &astrodynamics::solve_lambert, "Solves the Lambert problem",
          pybind11::arg("r1"), pybind11::arg("r2"), pybind11::arg("t"), pybind11::arg("mu"),
          pybind11::arg("is_prograde") = true, pybind11::arg("max_iter") = 100, pybind11::arg("tolerance") = 1e-6);
}
