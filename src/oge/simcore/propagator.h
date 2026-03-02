//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_PROPAGATOR_H
#define ORBITALGAMEENV_PROPAGATOR_H

#include "iostream"

#include "Eigen/Dense"

#include "../common/constants.h"
#include "oge/simcore/utils.h"

namespace oge {
    /**
     * This function uses Newton's method to solve Kepler's
     * equation E - e*sin(E) = M for the eccentric anomaly,
     * given the eccentricity and the mean anomaly.
     * @param e         eccentricity
     * @param M         mean anomaly (rad)
     * @param error     error
     * @return          eccentric anomaly (rad)
     */
    double kepler_E(double e, double M, double error = 1E-8);

    /**
     *
     * @param dt time since x = 0 (s)
     * @param ro radial position (km) when x = 0
     * @param vro radial velocity (km/s) when x = 0
     * @param a reciprocal of the semimajor axis (1/km)
     * @return the universal anomaly (km^0.5)
     */
    double kepler_U(double dt, double ro, double vro, double a);

    /**
         * This function computes the state vector (R, V) from the
         * initial state vector (R0, V0) and the elapsed time.
         * @param [input] R0 initial position vector (km)
         * @param [input] V0 initial velocity vector (km/s)
         * @param [input] t elapsed time (s)
         * @param [output] R final position vector (km)
         * @param [output] V final velocity vector (km)
         */
    void rv_from_r0v0(
        const Eigen::Vector3d &R0,
        const Eigen::Vector3d &V0,
        double t,
        Eigen::Vector3d &R,
        Eigen::Vector3d &V
    );
}

#endif //ORBITALGAMEENV_PROPAGATOR_H
