//
// Created by baoyicui on 2/21/26.
//
#include "oge/simcore/utils.h"

namespace oge
{
    double stumpS(const double z)
    {
        double s;
        if (z > 0.0)
        {
            s = (sqrt(z) - sin(sqrt(z))) / pow(sqrt(z), 3);
        }
        else if (z < 0.0)
        {
            s = (sinh(sqrt(-z)) - sqrt(-z)) / pow(sqrt(-z), 3);
        }
        else
        {
            s = 1.0 / 6.0;
        }
        return s;
    }

    double stumpC(double z)
    {
        double c;
        if (z > 0.0)
        {
            c = (1.0 - cos(sqrt(z))) / z;
        }
        else if (z < 0.0)
        {
            c = (cosh(sqrt(-z)) - 1.0) / (-z);
        }
        else
        {
            c = 1.0 / 2.0;
        }
        return c;
    }

    void f_and_g(
        const double x,
        const double t,
        const double ro,
        const double a,
        double& f,
        double& g)
    {
        const double z = a * pow(x, 2);

        f = 1 - pow(x, 2) / ro * stumpC(z);
        g = t - 1 / sqrt(MU) * pow(x, 3) * stumpS(z);
    }

    void fDot_and_gDot(
        const double x,
        const double r,
        const double ro,
        const double a,
        double& fdot,
        double& gdot)
    {
        const double z = a * pow(x, 2);
        fdot = sqrt(MU) / r / ro * (z * stumpS(z) - 1) * x;
        gdot = 1 - pow(x, 2) / r * stumpC(z);
    }

    void coe2rv(const Eigen::Matrix<double, 6, 1>& coe, Eigen::Vector3d& R, Eigen::Vector3d& V)
    {
        const double a = coe(0);
        const double e = coe(1);
        const double incl = coe(2);
        const double RA = coe(3);
        const double w = coe(4);
        const double TA = coe(5);

        const double h = sqrt(a * MU * (1.0 - e * e));

        Eigen::Vector3d rp = (h * h / MU) * (1.0 / (1.0 + e * cos(TA))) * (cos(TA) * Eigen::Vector3d::UnitX() +
            sin(TA) *
            Eigen::Vector3d::UnitY());
        Eigen::Vector3d vp = (MU / h) * (-sin(TA) * Eigen::Vector3d::UnitX() + (e + cos(TA)) *
            Eigen::Vector3d::UnitY());

        Eigen::Matrix3d Q_pX = (Eigen::AngleAxisd(RA, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(incl, Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(w, Eigen::Vector3d::UnitZ())).toRotationMatrix();

        R = Q_pX * rp;
        V = Q_pX * vp;
    }

    void rv2coe(const Eigen::Vector3d& R, const Eigen::Vector3d& V, Eigen::Matrix<double, 6, 1>& coe)
    {
        const double eps = 1e-10;
        const double r = R.norm();
        const double v = V.norm();

        double vr = R.dot(V) / r;

        Eigen::Vector3d H = R.cross(V);
        double h = H.norm();

        double incl = (h > eps) ? std::acos(std::clamp(H(2) / h, -1.0, 1.0)) : 0.0;

        Eigen::Vector3d N = Eigen::Vector3d(0, 0, 1).cross(H);
        double n = N.norm();

        double RA = 0.0;
        if (std::abs(n) > eps)
        {
            RA = std::acos(std::clamp(N(0) / n, -1.0, 1.0));
            if (N(1) < 0)
            {
                RA = 2.0 * M_PI - RA;
            }
        }
        else
        {
            RA = 0.0;
        }

        Eigen::Vector3d E = 1.0 / MU * ((v * v - MU / r) * R - r * vr * V);
        double e = E.norm();

        double w = 0.0;
        if (std::abs(n) > eps)
        {
            if (e > eps)
            {
                w = std::acos(std::clamp(N.dot(E) / n / e, -1.0, 1.0));
                if (E(2) < 0)
                {
                    w = 2 * M_PI - w;
                }
            }
            else
            {
                w = 0.0;
            }
        }
        else
        {
            w = 0.0;
        }

        double TA = 0.0;
        if (e > eps)
        {
            TA = std::acos(std::clamp(E.dot(R) / e / r, -1.0, 1.0));
            if (vr < 0)
            {
                TA = 2 * M_PI - TA;
            }
        }
        else
        {
            if (std::abs(n) > eps)
            {
                // Circular inclined orbit: TA measured from ascending node
                Eigen::Vector3d cp = N.cross(R);
                if (cp(2) >= 0)
                {
                    TA = std::acos(std::clamp(N.dot(R) / n / r, -1.0, 1.0));
                }
                else
                {
                    TA = 2 * M_PI - std::acos(std::clamp(N.dot(R) / n / r, -1.0, 1.0));
                }
            }
            else
            {
                // Circular equatorial orbit: use true longitude (angle of R from x-axis)
                TA = std::atan2(R(1), R(0));
                if (TA < 0.0) TA += 2.0 * M_PI;
            }
        }
        double a = h * h / MU / (1 - e * e);

        coe << a, e, incl, RA, w, TA;
    }

    void DCM_J2000_to_LVLH(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM)
    {
        const Eigen::Vector3d x_hat = RRefJ2000.normalized();
        const Eigen::Vector3d z_hat = RRefJ2000.cross(VRefJ2000).normalized();
        const Eigen::Vector3d y_hat = z_hat.cross(x_hat);

        DCM.row(0) = x_hat;
        DCM.row(1) = y_hat;
        DCM.row(2) = z_hat;
    }

    void DCM_LVLH_to_J2000(const Eigen::Vector3d& RRefJ2000, const Eigen::Vector3d& VRefJ2000, Eigen::Matrix3d& DCM)
    {
        Eigen::Matrix3d DCM_J2000_LVLH;
        DCM_J2000_to_LVLH(RRefJ2000, VRefJ2000, DCM_J2000_LVLH);
        DCM = DCM_J2000_LVLH.transpose();
    }

    void RV_J20002LVLH(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& RJ2000,
        const Eigen::Vector3d& VJ2000,
        Eigen::Vector3d& RLVLH,
        Eigen::Vector3d& VLVLH)
    {
        // construct LVLH frame
        Eigen::Matrix3d DCM;
        DCM_J2000_to_LVLH(RRefJ2000, VRefJ2000, DCM);
        // calculate relative position
        Eigen::Vector3d RRel = RJ2000 - RRefJ2000;
        RLVLH = DCM * RRel;
        // calculate relative velocity
        Eigen::Vector3d wJ2000 = RRefJ2000.cross(VRefJ2000) / pow(RRefJ2000.norm(), 2);
        Eigen::Vector3d VRel = (VJ2000 - VRefJ2000) - wJ2000.cross(RRel);
        VLVLH = DCM * VRel;
    }

    void RV_LVLH2J2000(
        const Eigen::Vector3d& RRefJ2000,
        const Eigen::Vector3d& VRefJ2000,
        const Eigen::Vector3d& R_LVLH,
        const Eigen::Vector3d& V_LVLH,
        Eigen::Vector3d& R_J2000,
        Eigen::Vector3d& V_J2000)
    {
        Eigen::Matrix3d DCM;
        DCM_LVLH_to_J2000(RRefJ2000, VRefJ2000, DCM);
        // relative position in J2000
        const Eigen::Vector3d RRel = DCM * R_LVLH;
        R_J2000 = RRel + RRefJ2000;
        // angular velocity of LVLH frame in J2000
        const Eigen::Vector3d wJ2000 = RRefJ2000.cross(VRefJ2000) / pow(RRefJ2000.norm(), 2);
        V_J2000 = DCM * V_LVLH + VRefJ2000 + wJ2000.cross(RRel);
    }
}
