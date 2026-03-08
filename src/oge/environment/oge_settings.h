//
// Created by baoyicui on 3/6/26.
//

#ifndef ORBITALGAMEENV_SETTINGS_H
#define ORBITALGAMEENV_SETTINGS_H

#include <stdexcept>

namespace oge
{
    struct OGESettings
    {
        int random_seed;

        int num_pursuers;
        int num_evaders = 1;

        // initial conditions
        double sma_base; // base semi-major axis (km)
        double ecc_base; // base eccentricity
        double incl_base; // base inclination (rad)
        double RA_base; // base right ascension of the ascending node (rad)
        double w_base; // base argument of perigee (rad)
        double TA_base; // base true anomaly (rad)

        double dv_max_per_step_p;
        double dv_max_per_step_e;
        double capture_distance;
        double timestep;
        double terminal_time;

        double reward_time_weight;
        double reward_formation_weight;
        double reward_fuel_weight;
        double reward_capture_weight;
        double reward_timeout_weight;
        double reward_fuelout_weight;
        double reward_advantage_weight;
        double reward_phase_dist_weight;

        // advantage reward
        int advantage_reward_horizon;

        // phase dist reward
        double phase_dist_transition_dist;

        void validate() const;
    };
}
#endif //ORBITALGAMEENV_SETTINGS_H
