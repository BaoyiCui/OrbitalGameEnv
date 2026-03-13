//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H
#define ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <functional>

#include "oge/vector/async_vectorizer.h"
#include "oge/vector/preprocessed_env.h"
#include "oge/vector/utils.h"
#include "oge/common/log.h"
#include "oge/environment/oge_settings.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

namespace oge::vector
{
    /**
     * OGEVectorInterface provides a vectorized interface to the Orbital Game Environment.
     */
    class OGEVectorInterface
    {
    public:
        using ConfigureFn = std::function<void(oge::OGEInterface&)>;

        OGEVectorInterface(
            const int num_envs,
            const int batch_size,
            const int num_threads,
            const int thread_affinity_offset = -1,
            const std::string& autoreset_mode = "NextStep",
            ConfigureFn configure_fn = nullptr
        );

        std::vector<Timestep> reset(
            const std::vector<int>& reset_indices,
            const std::vector<int>& reset_seeds
        );

        void send(const std::vector<std::vector<Eigen::Vector3d>>& batched_actions);

        /**
         * Returns the environments' data
         */
        const std::vector<Timestep> recv();

        int get_num_envs() const;

        const std::tuple<int, int> get_observation_shape() const;

        AutoresetMode get_autoreset_mode() const;

        const AsyncVectorizer* get_vectorizer() const;

    private:
        int num_envs_;
        AutoresetMode autoreset_mode_;
        std::vector<int> received_env_ids_;
        std::unique_ptr<AsyncVectorizer> vectorizer_;
    };
}

#endif //ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H
