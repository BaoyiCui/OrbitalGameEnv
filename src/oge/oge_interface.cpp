//
// Created by baoyicui on 2/21/26.
//

#include "oge/oge_interface.h"

namespace oge
{
    void OGEInterface::getRewards(
        const std::vector<Eigen::Vector3d>& actions,
        std::vector<double>& rewards
    ) const
    {
        environment->getRewards(actions, rewards);
    }

    void OGEInterface::getObservations(std::vector<Eigen::VectorXd>& observations) const
    {
        environment->getObservations(observations);
    }

    bool OGEInterface::getTerminal() const
    {
        return environment->isTerminal();
    }

    bool OGEInterface::getTruncated() const
    {
        return environment->isTruncated();
    }

    void OGEInterface::act(std::vector<Eigen::Vector3d>& actions)
    {
        environment->act(actions);
    }

    void OGEInterface::reset()
    {
        environment->reset();
    }
}
