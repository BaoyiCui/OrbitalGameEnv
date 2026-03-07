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

        void validate() const
        {
            if (num_evaders != 1)
                throw std::invalid_argument("OGESettings: num_evaders must be 1");
            if (num_pursuers <= 0)
                throw std::invalid_argument("OGESettings: num_pursuers must be > 0");
        }

        double dv_max_per_step_p;
        double dv_max_per_step_e;
        double capture_distance;
        double timestep;
        double terminal_time;

        double reward_time_weight;
        double reward_formation_weight;
        double reward_fuel_weight;
        double capture_reward;
        double reward_timeout_penalty;
        double reward_fuelout_penalty;
        double fuel_penalty_weight;

        double reward_advantage_weight;
        int advantage_reward_horizon;

        double reward_phase_dist_weight;
    };
}
#endif //ORBITALGAMEENV_SETTINGS_H
