//
// Created by baoyicui on 2/23/26.
//

#ifndef ORBITALGAMEENV_OGE_WRAPPER_H
#define ORBITALGAMEENV_OGE_WRAPPER_H

#include <vector>
#include <Eigen/Dense>


namespace oge
{
    class OrbitalGameEnvironment;

    class OrbitalGameEnvironmentWrapper
    {
    public:
        OrbitalGameEnvironmentWrapper(OrbitalGameEnvironment& environment);
        void act(
            std::vector<Eigen::Vector3d>);

        OrbitalGameEnvironment& environment_;
    };
}

#endif //ORBITALGAMEENV_OGE_WRAPPER_H
