//
// Created by baoyicui on 3/9/26.
//

#ifndef ORBITALGAMEENV_VECTOR_PREPROCESSED_ENV_H
#define ORBITALGAMEENV_VECTOR_PREPROCESSED_ENV_H

#include <memory>
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <algorithm>
#include <functional>

// SIMD intrinsics for maxpooling optimization
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <oge/oge_interface.h>
#include "oge/vector/utils.h"

namespace oge::vector
{
    class PreprocessedEnv
    {
    public:
        using ConfigureFn = std::function<void(oge::OGEInterface&)>;

        PreprocessedEnv(
            const int env_id,
            const ConfigureFn& configure = nullptr
        ) : env_id_(env_id)
        {
            env_ = std::make_unique<OGEInterface>();
            if (configure) configure(*env_);
            env_->init();
            reset();
        }

        void set_actions(const EnvironmentAction& actions)
        {
            current_actions_ = actions;
        }

        void reset()
        {
            env_->reset();
            env_->getObservations(current_observations_);
            current_rewards_.assign(current_observations_.size(), 0.0);
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
            const int num_agents = env_->getInt("num_pursuers") + env_->getInt("num_evaders");
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

        int getInt(const std::string& key, bool strict) const
        {
            return env_->getInt(key, strict);
        }

        float getFloat(const std::string& key, bool strict) const
        {
            return env_->getFloat(key, strict);
        }

        bool getBool(const std::string& key, bool strict) const
        {
            return env_->getBool(key, strict);
        }

        const std::string& getString(const std::string& key, bool strict) const
        {
            return env_->getString(key, strict);
        }

        void setInt(const std::string& key, const int value)
        {
            env_->setInt(key, value);
        }

        void setFloat(const std::string& key, const float value)
        {
            env_->setFloat(key, value);
        }

        void setBool(const std::string& key, const bool value)
        {
            env_->setBool(key, value);
        }

        void setString(const std::string& key, const std::string& value)
        {
            env_->setString(key, value);
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
#endif //ORBITALGAMEENV_VECTOR_PREPROCESSED_ENV_H
