//
// Created by baoyicui on 3/9/26.
//

#ifndef ORBITALGAMEENV_VECTOR_ASYNC_VECTORIZER_H
#define ORBITALGAMEENV_VECTOR_ASYNC_VECTORIZER_H

#include <vector>
#include <thread>

#ifndef MOODYCAMEL_DELETE_FUNCTION
#define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "oge/external/ThreadPool.h"
#include "oge/vector/utils.h"
#include "oge/vector/preprocessed_env.h"

namespace oge::vector
{
    class AsyncVectorizer
    {
    public:
        explicit AsyncVectorizer(
            const int num_envs,
            const int batch_size = 0,
            const int num_threads = 0,
            const int thread_affinity_offset = -1,
            const std::function<std::unique_ptr<PreprocessedEnv>(int)>& env_factory = nullptr,
            const AutoresetMode autoreset_mode = AutoresetMode::NextStep
        ) :
            num_envs_(num_envs),
            batch_size_(batch_size > 0 ? batch_size : num_envs),
            autoreset_mode_(autoreset_mode),
            stop_(false),
            action_queue_(new ActionQueue(num_envs_)),
            state_buffer_(new StateBuffer(batch_size_, num_envs_))
        {
            // Create environments
            envs_.resize(num_envs_);
            for (int i = 0; i < num_envs_; ++i)
            {
                envs_[i] = env_factory(i);
            }
            // Setup worker threads
            const std::size_t processor_count = std::thread::hardware_concurrency();
            if (num_threads <= 0)
            {
                num_threads_ = std::min<int>(batch_size_, static_cast<int>(processor_count));
            }
            else
            {
                num_threads_ = num_threads;
            }
            // Start worker threads
            for (int i = 0; i < num_threads_; ++i)
            {
                workers_.emplace_back([this]
                {
                    worker_function();
                });
            }

            // Set thread affinity if requested
            if (thread_affinity_offset >= 0)
            {
                set_thread_affinity(thread_affinity_offset, processor_count);
            }
        }

        ~AsyncVectorizer()
        {
            stop_ = true;
            // Send empty actions to wake up and terminate all worker threads
            const std::vector<ActionSlice> empty_actions(workers_.size());
            action_queue_->enqueue_bulk(empty_actions);
            for (auto& worker : workers_)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }


        void reset(const std::vector<int>& reset_indices, const std::vector<int>& seeds)
        {
            std::vector<ActionSlice> reset_actions;
            reset_actions.reserve(reset_indices.size());
            for (size_t i = 0; i < reset_indices.size(); ++i)
            {
                const int env_id = reset_indices[i];
                // envs_[env_id]->set_seed(seeds[i]);
                // TODO: NEED TO SET SEED IN RESET?

                ActionSlice action;
                action.env_id = env_id;
                action.force_reset = true;

                reset_actions.emplace_back(action);
            }

            action_queue_->enqueue_bulk(reset_actions);
        }

        void send(const std::vector<EnvironmentAction>& actions)
        {
            std::vector<ActionSlice> action_slices;
            action_slices.reserve(actions.size());

            for (size_t i = 0; i < actions.size(); i++)
            {
                const int env_id = actions[i].env_id;
                envs_[env_id]->set_actions(actions[i]);

                ActionSlice action;
                action.env_id = env_id;
                action.force_reset = false;

                action_slices.emplace_back(action);
            }

            action_queue_->enqueue_bulk(action_slices);
        }

        std::vector<Timestep> recv()
        {
            std::vector<Timestep> timesteps = state_buffer_->collect();
            return timesteps;
        }

        std::vector<Timestep> step(const std::vector<EnvironmentAction>& actions)
        {
            send(actions);
            return recv();
        }

        int get_num_envs() const
        {
            return num_envs_;
        }

        int get_batch_size() const
        {
            return batch_size_;
        }

        AutoresetMode get_autoreset_mode()
        {
            return autoreset_mode_;
        }

        std::tuple<int, int> get_obs_shape() const
        {
            return envs_[0]->get_obs_shape();
        }

    private:
        int num_envs_; // Number of parallel environments
        int batch_size_; // Batch size for processing
        int num_threads_; // Number of worker threads
        AutoresetMode autoreset_mode_; // How to reset sub-environments after an episode ends

        std::atomic<bool> stop_; // Signal to stop worker threads
        std::vector<std::thread> workers_; // Worker threads
        std::unique_ptr<ActionQueue> action_queue_; // Queue for actions
        std::unique_ptr<StateBuffer> state_buffer_; // Queue for observations
        std::vector<std::unique_ptr<PreprocessedEnv>> envs_; // Environment instances

        // mutable std::vector<std::vector<Eigen::VectorXd>> final_obs_storage_; // For same-step autoreset

        /**
         * Worker thread function that processes environment steps
         */
        void worker_function() const
        {
            while (!stop_)
            {
                try
                {
                    ActionSlice action = action_queue_->dequeue();
                    if (stop_)
                    {
                        break;
                    }

                    const int env_id = action.env_id;
                    if (autoreset_mode_ == AutoresetMode::NextStep)
                    {
                        if (action.force_reset || envs_[env_id]->is_episode_over())
                        {
                            envs_[env_id]->reset();
                        }
                        else
                        {
                            envs_[env_id]->step();
                        }
                        // Get timestep and write to state buffer
                        Timestep timestep = envs_[env_id]->get_timestep();
                        timestep.final_observations = std::nullopt;
                        state_buffer_->write(timestep);
                    }
                    else if (autoreset_mode_ == AutoresetMode::SameStep)
                    {
                        if (action.force_reset)
                        {
                            // on standard reset
                            envs_[env_id]->reset();
                            Timestep timestep = envs_[env_id]->get_timestep();
                            timestep.final_observations = std::nullopt;
                            state_buffer_->write(timestep);
                        }
                        else
                        {
                            envs_[env_id]->step();
                            Timestep step_timestep = envs_[env_id]->get_timestep();;
                            // if episode over, autoreset
                            if (envs_[env_id]->is_episode_over())
                            {
                                // final_obs_storage_[env_id] = step_timestep.observations;
                                envs_[env_id]->reset();
                                Timestep reset_timestep = envs_[env_id]->get_timestep();

                                reset_timestep.final_observations = step_timestep.observations;
                                reset_timestep.rewards = step_timestep.rewards;
                                reset_timestep.terminated = step_timestep.terminated;
                                reset_timestep.truncated = step_timestep.truncated;

                                // Write the reset timestep with the some of the step timestep data
                                state_buffer_->write(reset_timestep);
                            }
                            else
                            {
                                step_timestep.final_observations = std::nullopt;
                                state_buffer_->write(step_timestep);
                            }
                        }
                    }
                    else
                    {
                        throw std::runtime_error("Invalid autoreset mode");
                    }
                }
                catch (const std::exception& e)
                {
                    // log error but continue procession
                    std::cerr << "Error in worker thread: " << e.what() << std::endl;
                }
            }
        }

        /**
         * Set thread affinity for worker threads
         */
        void set_thread_affinity(const int thread_affinity_offset, const int processor_count)
        {
            for (size_t tid = 0; tid < workers_.size(); ++tid)
            {
                size_t core_id = (thread_affinity_offset + tid) % processor_count;

#if defined(__linux__)
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(core_id, &cpuset);
                pthread_setaffinity_np(workers_[tid].native_handle(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
                DWORD_PTR mask = (static_cast<DWORD_PTR>(1) << core_id);
                SetThreadAffinityMask(workers_[tid].native_handle(), mask);
#elif defined(__APPLE__)
                thread_affinity_policy_data_t policy = {static_cast<integer_t>(core_id)};
                thread_port_t mach_thread = pthread_mach_thread_np(workers_[tid].native_handle());
                thread_policy_set(mach_thread, THREAD_AFFINITY_POLICY,
                                  (thread_policy_t) & policy, THREAD_AFFINITY_POLICY_COUNT);
#endif
            }
        }
    };
}

#endif //ORBITALGAMEENV_VECTOR_ASYNC_VECTORIZER_H
