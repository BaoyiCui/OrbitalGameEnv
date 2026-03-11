//
// Created by baoyicui on 3/9/26.
//

#ifndef ORBITALGAMEENV_PREPROCESSED_ENV_H
#define ORBITALGAMEENV_PREPROCESSED_ENV_H

#include <memory>
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <algorithm>

// SIMD intrinsics for maxpooling optimization
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <oge/oge_interface.h>
#include "utils.h"

namespace oge::vector
{
    class PreprocessedEnv
    {
    public:
        PreprocessedEnv(
            const int env_id
        ) : env_id_(env_id)
        {
            env_ = std::make_unique<OGEInterface>();
            // TODO: how to set configurations here
            env_->init();
            reset();
        }

        void set_actions(const EnvironmentAction& actions)
        {
            current_actions_ = actions;
        }

        void reset()
        {
            env_.reset();
            env_->getObservations(current_observations_);
            current_rewards_.assign(current_rewards_.size(), 0.0);
            current_terminated_ = env_->getTerminal();
            current_truncated_ = env_->getTruncated();
            current_time_ = 0.0;
        }

        bool is_episode_over() const
        {
            return current_terminated_ || current_truncated_;
        }

        /**
         * Steps the environment using the current action
         */
        void step()
        {
            const std::vector<Eigen::Vector3d> actions = current_actions_.actions;

            env_->act(actions);

            env_->getObservations(current_observations_);
            env_->getRewards(actions, current_rewards_);
            current_terminated_ = env_->getTerminal();
            current_truncated_ = env_->getTruncated();
            current_time_ = env_->getCurrentTime();
        }

        std::tuple<int, int> get_obs_shape() const
        {
            const int num_agents = env_->getInt("num_agents");
            const int obs_size = env_->environment->getObsSize(0);
            return {num_agents, obs_size};
        }

        /**
         * Get the current observation
         */
        Timestep get_timestep() const
        {
            Timestep timestep;
            timestep.env_id = env_id_;

            timestep.rewards = current_rewards_;
            timestep.terminated = current_terminated_;
            timestep.truncated = current_truncated_;

            timestep.observations = current_observations_;
            timestep.current_time = current_time_;

            return timestep;
        }

    private:
        int env_id_; // Unique ID for this environment
        std::unique_ptr<OGEInterface> env_; // OGE interface
        std::vector<Eigen::VectorXd> current_observations_;
        EnvironmentAction current_actions_;
        std::vector<double> current_rewards_;
        bool current_terminated_;
        bool current_truncated_;
        double current_time_;
    };
}
#endif //ORBITALGAMEENV_PREPROCESSED_ENV_H
