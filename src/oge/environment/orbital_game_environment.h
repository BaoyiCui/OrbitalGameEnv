//
// Created by baoyicui on 2/22/26.
//

#ifndef ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
#define ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H

#include "oge/simcore/propagator.h"
#include "oge/environment/oge_state.h"
#include "oge/environment/oge_settings.h"

#include <string>
#include <vector>
#include <random>

namespace oge
{
    class OrbitalGameEnvironment
    {
    public:
        explicit OrbitalGameEnvironment(OGESettings& settings_);

        /** Reset the environment to its start state. */
        void reset();

        void act(
            std::vector<Eigen::Vector3d>& agents_actions,
            std::vector<double>& agents_rewards
        );

        bool isTerminal() const;

        bool isTruncated() const;

        void getObservations(std::vector<Eigen::VectorXd>& observations);
        void getRewards(std::vector<double>& rewards);

        /** Returns the observation vector size for agent at index agent_idx. */
        int getObsSize(int agent_idx) const;

        static bool almost_equal(double a, double b, double epsilon = 1e-12)
        {
            return std::abs(a - b) < epsilon;
        }

    private:
        void processDynamics(
            std::vector<Eigen::Vector3d>& actions
        );
        void checkAlive();

        double getFormationReward();


    private:
        const OGESettings& settings;

        // simulation settings
        const double dv_max_per_step_p; // pursuer's max dv per step, km/s
        const double dv_max_per_step_e; // evader's max dv per step, km/s
        const double capture_distance; // km
        const double timestep; // s
        const double terminal_time; // s
        // Agents' states
        const int num_pursuers;
        const int num_evaders;
        const int num_agents;
        std::vector<std::string> agent_ids;
        std::vector<SatState> agents_states; // the first num_evaders elements of agents_states are evaders' states

        double current_time; // s

        // random generator
        std::mt19937 _rng;
    };
}

#endif //ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
