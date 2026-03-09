//
// Created by baoyicui on 2/21/26.
//
#include "oge/python/oge_python_interface.h"

namespace oge
{
    nb::ndarray<nb::numpy, double> OGEPythonInterface::getRewards(nb::ndarray<nb::numpy, const double> actions) const
    {
        if (actions.ndim() != 2)
            throw std::runtime_error("Expected a numpy array with two dimensions.");
        if (actions.shape(1) != 3)
            throw std::runtime_error("Expected actions array with shape (num_agents, 3).");

        auto view = actions.view<const double, nb::ndim<2>>();
        const int n = static_cast<int>(view.shape(0));

        std::vector<Eigen::Vector3d> acts(n);
        for (int i = 0; i < n; i++)
            acts[i] = Eigen::Vector3d(view(i, 0), view(i, 1), view(i, 2));

        std::vector<double> rewards;
        OGEInterface::getRewards(acts, rewards);

        auto* data = new double[rewards.size()];
        std::copy(rewards.begin(), rewards.end(), data);

        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = { rewards.size() };
        return {data, 1, shape, owner};
    }

    nb::ndarray<nb::numpy, double> OGEPythonInterface::getObservations() const
    {
        std::vector<Eigen::VectorXd> observations;
        OGEInterface::getObservations(observations);

        size_t total = 0;
        for (const auto& obs : observations)
            total += obs.size();

        auto* data = new double[total];
        size_t offset = 0;
        for (const auto& obs : observations)
        {
            std::copy(obs.data(), obs.data() + obs.size(), data + offset);
            offset += obs.size();
        }

        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        size_t shape[1] = { total };
        return {data, 1, shape, owner};
    }

    bool OGEPythonInterface::getTerminal() const
    {
        return OGEInterface::getTerminal();
    }

    bool OGEPythonInterface::getTruncated() const
    {
        return OGEInterface::getTruncated();
    }

    void OGEPythonInterface::act(nb::ndarray<nb::numpy, double> actions)
    {
        if (actions.ndim() != 2 || actions.shape(1) != 3)
            throw std::runtime_error("Expected actions array with shape (num_agents, 3).");

        auto view = actions.view<double, nb::ndim<2>>();
        const int n = static_cast<int>(view.shape(0));
        std::vector<Eigen::Vector3d> acts(n);
        for (int i = 0; i < n; i++)
            acts[i] = Eigen::Vector3d(view(i, 0), view(i, 1), view(i, 2));
        OGEInterface::act(acts);
    }

    void OGEPythonInterface::reset()
    {
        OGEInterface::reset();
    }
}
