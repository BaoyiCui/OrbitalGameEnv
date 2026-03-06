//
// Created by baoyicui on 2/22/26.
//

#ifndef ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
#define ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H



#include "oge/simcore/propagator.h"
#include "oge/environment/oge_state.h"

#include <string>
#include <vector>

namespace oge {
    class OrbitalGameEnvironment {
    public:
        OrbitalGameEnvironment();

        /** Reset the environment to its start state. */
        void reset();

        void act(
            std::vector<Eigen::Vector3d> &agents_actions,
            std::vector<double> &agents_rewards
        );

        bool isTerminal() const;

        bool isTruncated() const;

    private:
        void processDynamics(
            std::vector<Eigen::Vector3d>& actions
        );

    private:
        // Agents' states
        std::vector<std::string> agent_ids;
        std::vector<SatState> agents_states; // agents_states[0] is evader's states

        const double dv_max_per_step_p; // pursuer's max dv per step, km/s
        const double dv_max_per_step_e; // evader's max dv per step, km/s
        const double capture_distance;  // km
        const double timestep;          // s
        const double terminal_time;     // s
        double current_time;            // s
    };
}

#endif //ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
