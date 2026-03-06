//
// Created by baoyicui on 2/23/26.
//

#include "oge/environment/oge_wrapper.h"
#include "oge/environment/orbital_game_environment.h" //OrbitalGameEnvironment在这

namespace oge
{
    OrbitalGameEnvironmentWrapper::OrbitalGameEnvironmentWrapper(
        OrbitalGameEnvironment& environment
    )
        : environment_(environment)
    {
    }
}
