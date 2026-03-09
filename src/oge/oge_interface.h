//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_INTERFACE_H
#define ORBITALGAMEENV_OGE_INTERFACE_H

#include "oge/environment/orbital_game_environment.h"

#include <Eigen/Dense>
#include <memory>

namespace oge
{
    class OGEInterface
    {
    public:
        OGEInterface();
        ~OGEInterface();

        void getRewards(const std::vector<Eigen::Vector3d>& actions, std::vector<double>& rewards) const;
        void getObservations(std::vector<Eigen::VectorXd>& observations) const;
        bool getTerminal() const;
        bool getTruncated() const;
        void act(std::vector<Eigen::Vector3d>& actions);
        void reset();

    public:
        std::unique_ptr<oge::OrbitalGameEnvironment> environment;
        std::unique_ptr<oge::OGESettings> settings;
    };
}


#endif //ORBITALGAMEENV_OGE_INTERFACE_H
