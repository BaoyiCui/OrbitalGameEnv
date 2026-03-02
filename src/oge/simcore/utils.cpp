//
// Created by baoyicui on 2/21/26.
//
#include "oge/simcore/utils.h"

double oge::stumpS(const double z) {
    double s;
    if (z > 0.0) {
        s = (sqrt(z) - sin(sqrt(z))) / pow(sqrt(z), 3);
    } else if (z < 0.0) {
        s = (sinh(sqrt(-z)) - sqrt(-z)) / pow(sqrt(-z), 3);
    } else {
        s = 1.0 / 6.0;
    }
    return s;
}

double oge::stumpC(double z) {
    double c;
    if (z > 0.0) {
        c = (1.0 - cos(sqrt(z))) / z;
    } else if (z < 0.0) {
        c = (cosh(sqrt(-z)) - 1.0) / (-z);
    } else {
        c = 1.0 / 2.0;
    }
    return c;
}

void oge::f_and_g(
    const double x,
    const double t,
    const double ro,
    const double a,
    double &f,
    double &g) {
    const double z = a * pow(x, 2);

    f = 1 - pow(x, 2) / ro * stumpC(z);
    g = t - 1 / sqrt(oge::MU) * pow(x, 3) * stumpS(z);
}

void oge::fDot_and_gDot(
    const double x,
    const double r,
    const double ro,
    const double a,
    double &fdot,
    double &gdot) {
    const double z = a * pow(x, 2);
    fdot = sqrt(oge::MU) / r / ro * (z * stumpS(z) - 1) * x;
    gdot = 1 - pow(x, 2) / r * stumpC(z);
}
