//
// Created by baoyicui on 2/21/26.
//
#include "oge_vector_python_interface.h"


namespace nb = nanobind;

void init_vector_module(nb::module_& m)
{
    // Define OGEVectorInterface class
    nb::class_<oge::vector::OGEVectorInterface>(m, "OGEVectorInterface")
        .def("__init__",
             [](oge::vector::OGEVectorInterface* self,
                int num_envs, int batch_size, int num_threads,
                int thread_affinity_offset, const std::string& autoreset_mode)
             {
                 new (self) oge::vector::OGEVectorInterface(
                     num_envs, batch_size, num_threads, thread_affinity_offset, autoreset_mode);
             },
             nb::arg("num_envs"),
             nb::arg("batch_size") = 0,
             nb::arg("num_threads") = 0,
             nb::arg("thread_affinity_offset") = -1,
             nb::arg("autoreset_mode") = "NextStep")
        .def("__init__",
             [](oge::vector::OGEVectorInterface* self,
                int num_envs, int batch_size, int num_threads,
                int thread_affinity_offset, const std::string& autoreset_mode,
                oge::OGESettings& settings)
             {
                 auto configure_fn = [&settings](oge::OGEInterface& iface)
                 {
                     settings.copyTo(*iface.settings);
                 };
                 new (self) oge::vector::OGEVectorInterface(
                     num_envs, batch_size, num_threads, thread_affinity_offset,
                     autoreset_mode, configure_fn);
             },
             nb::arg("num_envs"),
             nb::arg("batch_size") = 0,
             nb::arg("num_threads") = 0,
             nb::arg("thread_affinity_offset") = -1,
             nb::arg("autoreset_mode") = "NextStep",
             nb::arg("settings"))
        .def("reset", [](oge::vector::OGEVectorInterface& self, const std::vector<int> reset_indices,
                         const std::vector<int> reset_seeds)
        {
            nb::gil_scoped_release release;
            auto timesteps = self.reset(reset_indices, reset_seeds);
            nb::gil_scoped_acquire acquire;

            const int batch_size = static_cast<int>(timesteps.size());
            const auto [num_agents, obs_size] = self.get_observation_shape();
            const size_t obs_per_env = num_agents * obs_size;

            // Allocate data arrays (unique_ptr ensures no leak if an exception is thrown)
            auto obs_raw = std::unique_ptr<double[]>(new double[batch_size * obs_per_env]);
            auto env_ids_raw = std::unique_ptr<int[]>(new int[batch_size]);

            // Copy data from timesteps
            for (int i = 0; i < batch_size; ++i)
            {
                const auto& ts = timesteps[i];
                env_ids_raw[i] = ts.env_id;

                if (ts.observations.size() != num_agents)
                {
                    throw std::runtime_error(
                        "reset() returned wrong number of agent observations for env_id = " + std::to_string(ts.env_id)
                        + ", expected " + std::to_string(num_agents)
                        + " but got " + std::to_string(ts.observations.size()));
                }

                for (int j = 0; j < num_agents; ++j)
                {
                    if (ts.observations[j].size() != obs_size)
                    {
                        throw std::runtime_error(
                            "reset() returned wrong observation size for env_id=" +
                            std::to_string(ts.env_id) +
                            ", agent=" + std::to_string(j) +
                            ", expected=" + std::to_string(obs_size) +
                            ", got=" + std::to_string(ts.observations[j].size())
                        );
                    }
                    std::memcpy(
                        obs_raw.get() + i * obs_per_env + j * obs_size,
                        ts.observations[j].data(),
                        obs_size * sizeof(double)
                    );
                }
            }

            // Transfer ownership to capsules
            auto* obs_data = obs_raw.release();
            auto* env_ids_data = env_ids_raw.release();
            nb::capsule obs_owner(obs_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            nb::capsule env_ids_owner(env_ids_data, [](void* p) noexcept { delete[] static_cast<int*>(p); });

            // Build numpy arrays
            size_t obs_shape[3] = {(size_t)batch_size, (size_t)num_agents, (size_t)obs_size};
            size_t batch_shape[1] = {(size_t)batch_size};

            auto observations = nb::ndarray<nb::numpy, double>(obs_data, 3, obs_shape, obs_owner);
            auto env_ids = nb::ndarray<nb::numpy, int>(env_ids_data, 1, batch_shape, env_ids_owner);

            // Create info dict
            nb::dict info;
            info["env_id"] = env_ids;

            return nb::make_tuple(observations, info);
        })
        .def("send", [](
             oge::vector::OGEVectorInterface& self,
             nb::ndarray<nb::numpy, const double, nb::ndim<3>> actions
         )
             {
                 if (actions.ndim() != 3)
                 {
                     throw std::runtime_error("send() expects a 3D numpy array with shape (batch_size, num_agent, 3)");
                 }
                 const size_t batch_size = actions.shape(0);
                 const size_t num_agents = actions.shape(1);
                 const size_t action_size = actions.shape(2);
                 if (action_size != 3)
                 {
                     throw std::runtime_error(
                         "send() expects the last dimension to be 3, got " + std::to_string(action_size));
                 }

                 std::vector<std::vector<Eigen::Vector3d>> batched_actions(batch_size);
                 auto view = actions.view();
                 for (size_t i = 0; i < batch_size; ++i)
                 {
                     batched_actions[i].resize(num_agents);
                     for (size_t j = 0; j < num_agents; ++j)
                     {
                         batched_actions[i][j] = Eigen::Vector3d(view(i, j, 0), view(i, j, 1), view(i, j, 2));
                     }
                 }
                 nb::gil_scoped_release release;
                 self.send(batched_actions);
             })
        .def("recv", [](oge::vector::OGEVectorInterface& self)
        {
            nb::gil_scoped_release release;
            const auto timesteps = self.recv();
            nb::gil_scoped_acquire acquire;

            // Get shape information
            const size_t batch_size = timesteps.size();
            const auto [num_agents, obs_size] = self.get_observation_shape();

            // Get autoreset mode
            const auto autoreset_mode = self.get_autoreset_mode();

            // Allocate memory for arrays
            const size_t obs_total_size = batch_size * num_agents * obs_size;
            auto obs_raw = std::unique_ptr<double[]>(new double[obs_total_size]);
            auto rewards_raw = std::unique_ptr<double[]>(new double[batch_size * num_agents]);
            auto terminated_raw = std::unique_ptr<bool[]>(new bool[batch_size]);
            auto truncated_raw = std::unique_ptr<bool[]>(new bool[batch_size]);
            auto env_ids_raw = std::unique_ptr<int[]>(new int[batch_size]);
            auto current_times_raw = std::unique_ptr<double[]>(new double[batch_size]);

            // Copy data from timesteps to arrays
            const size_t obs_per_env = num_agents * obs_size;
            for (int i = 0; i < batch_size; ++i)
            {
                const auto& ts = timesteps[i];
                env_ids_raw[i] = ts.env_id;
                terminated_raw[i] = ts.terminated;
                truncated_raw[i] = ts.truncated;
                current_times_raw[i] = ts.current_time;

                if (ts.observations.size() != num_agents)
                {
                    throw std::runtime_error(
                        "recv() returned wrong number of agent observations for env_id = " + std::to_string(ts.env_id)
                        + ", expected " + std::to_string(num_agents)
                        + " but got " + std::to_string(ts.observations.size()));
                }

                if (ts.rewards.size() != num_agents)
                {
                    throw std::runtime_error(
                        "recv() returned wrong number of rewards for env_id=" +
                        std::to_string(ts.env_id) +
                        ", expected=" + std::to_string(num_agents) +
                        ", got=" + std::to_string(ts.rewards.size())
                    );
                }
                for (int j = 0; j < num_agents; ++j)
                {
                    if (ts.observations[j].size() != obs_size)
                    {
                        throw std::runtime_error(
                            "recv() returned wrong observation size for env_id=" +
                            std::to_string(ts.env_id) +
                            ", agent=" + std::to_string(j) +
                            ", expected=" + std::to_string(obs_size) +
                            ", got=" + std::to_string(ts.observations[j].size())
                        );
                    }
                    std::memcpy(
                        obs_raw.get() + i * obs_per_env + j * obs_size,
                        ts.observations[j].data(),
                        obs_size * sizeof(double)
                    );
                    rewards_raw[i * num_agents + j] = ts.rewards[j];
                }
            }

            bool any_terminated = std::any_of(
                terminated_raw.get(),
                terminated_raw.get() + batch_size,
                [](bool b) { return b; }
            );
            bool any_truncated = std::any_of(
                truncated_raw.get(),
                truncated_raw.get() + batch_size,
                [](bool b) { return b; }
            );

            // Create capsules for cleanup
            auto* obs_data = obs_raw.release(); // transfer ownership to capsules
            auto* rewards_data = rewards_raw.release();
            auto* terminated_data = terminated_raw.release();
            auto* truncated_data = truncated_raw.release();
            auto* env_ids_data = env_ids_raw.release();
            auto* current_times_data = current_times_raw.release();

            nb::capsule obs_owner(obs_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            nb::capsule rewards_owner(rewards_data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            nb::capsule terminated_owner(terminated_data, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
            nb::capsule truncated_owner(truncated_data, [](void* p) noexcept { delete[] static_cast<bool*>(p); });
            nb::capsule env_ids_owner(env_ids_data, [](void* p) noexcept { delete[] static_cast<int*>(p); });
            nb::capsule current_times_owner(
                current_times_data,
                [](void* p) noexcept { delete[] static_cast<double*>(p); }
            );


            // Create numpy arrays with allocated data
            size_t obs_shape[3] = {(size_t)batch_size, (size_t)num_agents, (size_t)obs_size};
            size_t reward_shape[2] = {(size_t)batch_size, (size_t)num_agents};
            size_t batch_shape[1] = {(size_t)batch_size};
            auto observations = nb::ndarray<nb::numpy, double>(obs_data, 3, obs_shape, obs_owner);
            auto rewards = nb::ndarray<nb::numpy, double>(rewards_data, 2, reward_shape, rewards_owner);
            auto terminated = nb::ndarray<nb::numpy, bool>(terminated_data, 1, batch_shape, terminated_owner);
            auto truncated = nb::ndarray<nb::numpy, bool>(truncated_data, 1, batch_shape, truncated_owner);
            auto env_ids = nb::ndarray<nb::numpy, int>(env_ids_data, 1, batch_shape, env_ids_owner);
            auto current_times = nb::ndarray<
                nb::numpy, double>(current_times_data, 1, batch_shape, current_times_owner);

            // Create info dict
            nb::dict info;
            info["env_id"] = env_ids;
            info["current_times"] = current_times;

            if (autoreset_mode == oge::vector::AutoresetMode::SameStep)
            {
                if (any_terminated || any_truncated)
                {
                    auto final_obs_raw = std::unique_ptr<double[]>(new double[obs_total_size]);
                    for (int i = 0; i < batch_size; ++i)
                    {
                        const auto& ts = timesteps[i];

                        // Use final_observation if available, otherwise use current observation
                        const std::vector<Eigen::VectorXd>& src_obs =
                            ((ts.terminated || ts.truncated) && ts.final_observations.has_value())
                                ? ts.final_observations.value()
                                : ts.observations;
                        if (src_obs.size() != num_agents)
                        {
                            throw std::runtime_error(
                                "recv() returned wrong number of final observations for env_id=" +
                                std::to_string(ts.env_id) +
                                ", expected=" + std::to_string(num_agents) +
                                ", got=" + std::to_string(src_obs.size())
                            );
                        }
                        for (int j = 0; j < num_agents; ++j)
                        {
                            if (src_obs[j].size() != obs_size)
                            {
                                throw std::runtime_error(
                                    "recv() returned wrong final observation size for env_id=" +
                                    std::to_string(ts.env_id) +
                                    ", agent=" + std::to_string(j) +
                                    ", expected=" + std::to_string(obs_size) +
                                    ", got=" + std::to_string(src_obs[j].size())
                                );
                            }
                            std::memcpy(
                                final_obs_raw.get() + i * obs_per_env + j * obs_size,
                                src_obs[j].data(),
                                obs_size * sizeof(double)
                            );
                        }
                    }

                    auto* final_obs_data = final_obs_raw.release();
                    nb::capsule final_obs_owner(final_obs_data, [](void* p) noexcept
                    {
                        delete[] static_cast<double*>(p);
                    });
                    auto final_observation = nb::ndarray<nb::numpy, double>(
                        final_obs_data, 3, obs_shape, final_obs_owner);
                    info["final_obs"] = final_observation;
                }
            }
            return nb::make_tuple(observations, rewards, terminated, truncated, info);
        })
        .def("get_num_envs", &oge::vector::OGEVectorInterface::get_num_envs)
        .def("get_single_observation_size", &oge::vector::OGEVectorInterface::get_observation_shape)
    .def("handle", [](oge::vector::OGEVectorInterface& self)
        {
            // Get the raw pointer to the AsyncVectorizer
            auto ptr = self.get_vectorizer();

            // Allocate memory for handle array
            auto handle_raw = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(ptr)]);
            std::memcpy(handle_raw.get(), &ptr, sizeof(ptr));

            // Create capsule for cleanup
            auto* handle_data = handle_raw.release();
            nb::capsule handle_owner(handle_data, [](void* p)noexcept { delete[] static_cast<uint8_t*>(p); });

            // Create numpy array
            size_t shape[1] = {sizeof(ptr)};
            return nb::ndarray<nb::numpy, uint8_t>(handle_data, 1, shape, handle_owner);
        });
}

namespace oge::vector
{
    OGEVectorInterface::OGEVectorInterface(
        const int num_envs,
        const int batch_size,
        const int num_threads,
        const int thread_affinity_offset,
        const std::string& autoreset_mode,
        ConfigureFn configure_fn
    ) : num_envs_(num_envs),
        received_env_ids_(batch_size > 0 ? batch_size : num_envs)
    {
        // create environment factory
        auto env_factory = [configure_fn](int env_id)
        {
            return std::make_unique<PreprocessedEnv>(env_id, configure_fn);
        };

        if (autoreset_mode == "NextStep")
        {
            autoreset_mode_ = AutoresetMode::NextStep;
        }
        else if (autoreset_mode == "SameStep")
        {
            autoreset_mode_ = AutoresetMode::SameStep;
        }
        else
        {
            throw std::invalid_argument(
                "Invalid autoreset_mode: " + autoreset_mode + ", expected values: 'NextStep' or 'SameStep'");
        }

        // create vectorizer
        vectorizer_ = std::make_unique<AsyncVectorizer>(
            num_envs,
            batch_size,
            num_threads,
            thread_affinity_offset,
            env_factory,
            autoreset_mode_
        );
    }

    std::vector<Timestep> OGEVectorInterface::reset(const std::vector<int>& reset_indices,
                                                    const std::vector<int>& reset_seeds)
    {
        vectorizer_->reset(reset_indices, reset_seeds);
        return recv();
    }

    const std::vector<Timestep> OGEVectorInterface::recv()
    {
        std::vector<Timestep> timesteps = vectorizer_->recv();
        for (auto i = 0; i < timesteps.size(); i++)
        {
            received_env_ids_[i] = timesteps[i].env_id;
        }
        return timesteps;
    }

    void OGEVectorInterface::send(
        const std::vector<std::vector<Eigen::Vector3d>>& batched_actions
    )
    {
        if (batched_actions.size() != received_env_ids_.size())
        {
            throw std::invalid_argument(
                "The size of actions_batch is different from the expected batch size, "
                "actions_batch length=" + std::to_string(batched_actions.size()) +
                ", expected length=" + std::to_string(received_env_ids_.size())
            );
        }

        std::vector<EnvironmentAction> environment_actions;
        environment_actions.reserve(batched_actions.size());

        for (auto i = 0; i < batched_actions.size(); ++i)
        {
            environment_actions.push_back(
                EnvironmentAction{
                    .env_id = received_env_ids_[i],
                    .actions = batched_actions[i]
                }
            );
        }

        vectorizer_->send(environment_actions);
    }

    int OGEVectorInterface::get_num_envs() const
    {
        return num_envs_;
    }

    const std::tuple<int, int> OGEVectorInterface::get_observation_shape() const
    {
        return vectorizer_->get_obs_shape();
    }


    AutoresetMode OGEVectorInterface::get_autoreset_mode() const
    {
        return autoreset_mode_;
    }

    const AsyncVectorizer* OGEVectorInterface::get_vectorizer() const
    {
        return vectorizer_.get();
    }
}
