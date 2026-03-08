//
// Created by baoyicui on 3/6/26.
//

#include "oge_settings.h"


namespace oge
{
    void OGESettings::validate() const
    {
        if (num_evaders != 1)
            throw std::invalid_argument("OGESettings: num_evaders must be 1");
        if (num_pursuers <= 0)
            throw std::invalid_argument("OGESettings: num_pursuers must be > 0");

        if (sma_base <= 0.0)
            throw std::invalid_argument("OGESettings: sma_base must be > 0");
        if (ecc_base < 0.0 || ecc_base >= 1.0)
            throw std::invalid_argument("OGESettings: ecc_base must be in [0, 1)");

        if (dv_max_per_step_p < 0.0)
            throw std::invalid_argument("OGESettings: dv_max_per_step_p must be >= 0");
        if (dv_max_per_step_e < 0.0)
            throw std::invalid_argument("OGESettings: dv_max_per_step_e must be >= 0");
        if (capture_distance <= 0.0)
            throw std::invalid_argument("OGESettings: capture_distance must be > 0");
        if (timestep <= 0.0)
            throw std::invalid_argument("OGESettings: timestep must be > 0");
        if (terminal_time <= 0.0)
            throw std::invalid_argument("OGESettings: terminal_time must be > 0");
        if (terminal_time < timestep)
            throw std::invalid_argument("OGESettings: terminal_time must be >= timestep");

        if (reward_time_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_time_weight must be >= 0");
        if (reward_formation_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_formation_weight must be >= 0");
        if (reward_fuel_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_fuel_weight must be >= 0");
        if (reward_capture_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_capture_weight must be >= 0");
        if (reward_timeout_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_timeout_weight must be >= 0");
        if (reward_fuelout_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_fuelout_weight must be >= 0");
        if (reward_advantage_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_advantage_weight must be >= 0");
        if (reward_phase_dist_weight < 0.0)
            throw std::invalid_argument("OGESettings: reward_phase_dist_weight must be >= 0");

        if (advantage_reward_horizon <= 0)
            throw std::invalid_argument("OGESettings: advantage_reward_horizon must be > 0");
    }
}
