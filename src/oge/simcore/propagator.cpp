//
// Created by baoyicui on 2/21/26.
//

#include "oge/simcore/propagator.h"


double oge::kepler_E(const double e, const double M, const double error)
{
    double E;

    if (M < M_PI)
    {
        E = M + e * 0.5;
    }
    else
    {
        E = M - e * 0.5;
    }

    double ratio = 1.0;

    while (abs(ratio) > error)
    {
        ratio = (E - e * sin(E) - M) / (1 - e * cos(E));
        E = E - ratio;
    }

    return E;
}

double oge::kepler_U(double dt, double ro, double vro, double a)
{
    constexpr double error = 1E-8;
    constexpr int nMax = 1000;

    // starting value for x
    double x = sqrt(oge::MU) * abs(a) * dt;

    int n = 0;
    double ratio = 1.0;

    double dFdx, F;

    while (abs(ratio) > error && n <= nMax)
    {
        ++n;
        const double c = stumpC(a * pow(x, 2));
        const double s = stumpS(a * pow(x, 2));
        F = ro * vro / sqrt(oge::MU) * pow(x, 2) * c + (1 - a * ro) * pow(x, 3) * s + ro * x -
            sqrt(oge::MU) * dt;
        dFdx = ro * vro / sqrt(oge::MU) * x * (1 - a * pow(x, 2) * s) + (1 - a * ro) * pow(x, 2) * c + ro;
        ratio = F / dFdx;
        x = x - ratio;
    }

    // Deliver a value for x, but report that nMax was reached
    if (n > nMax)
    {
        // std::string msg = fmt::format(
        //                               F / dFdx);
        std::ostringstream oss;
        oss << "Kepler solver failed to converge after " << n <<
            " iterations. Residual: " << F / dFdx;
        throw std::runtime_error(oss.str());
    }
    return x;
}

void oge::rv_from_r0v0(
    const Eigen::Vector3d& R0,
    const Eigen::Vector3d& V0,
    const double t,
    Eigen::Vector3d& R,
    Eigen::Vector3d& V
)
{
    // magnitudes of R0 and V0
    const double r0 = R0.norm();
    const double v0 = V0.norm();
    // Initial radial velocity
    const double vr0 = R0.dot(V0) / r0;

    const double alpha = 2 / r0 - pow(v0, 2) / MU;
    double x = kepler_U(t, r0, vr0, alpha);

    double f, g;
    f_and_g(x, t, r0, alpha, f, g);

    R = f * R0 + g * V0;
    const double r = R.norm();

    double fdot, gdot;

    fDot_and_gDot(x, r, r0, alpha, fdot, gdot);

    V = fdot * R0 + gdot * V0;
}
