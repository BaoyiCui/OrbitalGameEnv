//
// Created by baoyicui on 2/21/26.
//

#include "oge/oge_interface.h"

namespace oge
{
    OGEInterface::OGEInterface()
    {
        settings = std::make_unique<OGESettings>();
    }

    void OGEInterface::init()
    {
        environment = std::make_unique<OrbitalGameEnvironment>(*settings);
    }

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

    int OGEInterface::getInt(const std::string& key, bool strict) const
    {
        return settings->getInt(key, strict);
    }

    float OGEInterface::getFloat(const std::string& key, bool strict) const
    {
        return settings->getFloat(key, strict);
    }

    bool OGEInterface::getBool(const std::string& key, bool strict) const
    {
        return settings->getBool(key, strict);
    }

    const std::string& OGEInterface::getString(const std::string& key, bool strict) const
    {
        return settings->getString(key, strict);
    }

    void OGEInterface::setInt(const std::string& key, const int value)
    {
        settings->setInt(key, value);
    }

    void OGEInterface::setFloat(const std::string& key, const float value)
    {
        settings->setFloat(key, value);
    }

    void OGEInterface::setBool(const std::string& key, const bool value)
    {
        settings->setBool(key, value);
    }

    void OGEInterface::setString(const std::string& key, const std::string& value)
    {
        settings->setString(key, value);
    }
}
