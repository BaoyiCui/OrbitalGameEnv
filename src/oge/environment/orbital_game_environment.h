//
// Created by baoyicui on 2/22/26.
//

#ifndef ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
#define ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H

#include "unordered_map"
#include "string"
#include "vector"

#include "oge/simcore/propagator.h"
#include "oge/environment/oge_state.h"


namespace oge {
    class OrbitalGameEnvironment {
    public:
        OrbitalGameEnvironment();

        /** Reset the environment to its start state. */
        void reset();

        void act(
            std::vector<Eigen::Vector3d> pursuers_actions,
            Eigen::Vector3d evader_actions,
            std::vector<double> &pursuers_rewards
        );

        bool isTerminal();

        bool isTruncated() const;

    private:
        void processDynamics(
            std::vector<Eigen::Vector3d> actions
        );

    private:
        // Agents' states
        std::vector<std::string> agent_ids;
        std::vector<SatState> agent_states; // agent_states[0] is evader's states

        const double dv_max_per_step;
        const double timestep;
        const double terminal_time;
        double current_time;
    };
}

#endif //ORBITALGAMEENV_ORBITAL_GAME_ENVIRONMENT_H
