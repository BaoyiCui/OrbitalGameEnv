//
// Created by baoyicui on 2/21/26.
//
#include "oge_vector_python_interface.h"


namespace nb = nanobind;

void init_vector_module(nb::module_& m)
{
    // Define OGEVectorInterface class
    nb::class_<oge::vector::OGEVectorInterface>(m, "OGEVectorInterface")
        .def(nb::init<int, int, int, int, std::string>(),
             nb::arg("num_envs"),
             nb::arg("batch_size") = 0,
             nb::arg("num_threads") = 0,
             nb::arg("thread_affinity_offset") = -1,
             nb::arg("autoreset_mode") = "NextStep")
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

            return nb::make_tuple(observations, env_ids);
        })
        .def("send")
        .def("recv")
        .def("get_num_envs")
        .def("get_autoreset_mode")
        .def("handle");
}

namespace oge::vector
{
    OGEVectorInterface::OGEVectorInterface(
        const int num_envs,
        const int batch_size,
        const int num_threads,
        const int thread_affinity_offset,
        const std::string& autoreset_mode
    ) : num_envs_(num_envs),
        received_env_ids_(batch_size > 0 ? batch_size : num_envs)
    {
        // create environment factory
        auto env_factory = [this](int env_id)
        {
            return std::make_unique<PreprocessedEnv>(env_id);
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
