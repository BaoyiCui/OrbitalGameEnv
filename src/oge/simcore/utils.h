//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_UTILS_H
#define ORBITALGAMEENV_UTILS_H

#include <cmath>

#include "../common/constants.h"
#include "oge/simcore/propagator.h"

namespace oge {
    /**
     * This function evaluates the Stumpff function S(z) according to Eq 3.49.
     * @param z input argument
     * @return value of S(z)
     */
    double stumpS(double z);

    /**
     * This function evaluates the Stumpff function C(z) according to Eq 3.50.
     * @param z input argument
     * @return value of C(z)
     */
    double stumpC(double z);

    /**
     *
     * @param [input] x     the universal anomaly after time t (kmˆ0.5)
     * @param [input] t     the time elapsed since t (s)
     * @param [input] ro    the radial position at time t (km)
     * @param [input] a     reciprocal of the semimajor axis (1/km)
     * @param [output] f    the Lagrange f coefficient (dimensionless)
     * @param [output] g    the Lagrange g coefficient (s)
     */
    void f_and_g(double x, double t, double ro, double a, double &f, double &g);

    void fDot_and_gDot(double x, double r, double ro, double a, double &fdot, double &gdot);
}

#endif //ORBITALGAMEENV_UTILS_H
