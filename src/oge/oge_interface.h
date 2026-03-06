//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_INTERFACE_H
#define ORBITALGAMEENV_OGE_INTERFACE_H

#include "oge/environment/orbital_game_environment.h"

#include <memory>

class OGEInterface
{
public:
    OGEInterface();
    ~OGEInterface();

    // TODO:
    void act();

    // Indicates if the game has ended
    bool game_over(bool with_truncation = true) const;

    // Indicates if the game has been truncated
    bool game_truncated() const;

    // Reset the game
    void reset_game();

    void getStates(); // TODO

public:
    std::unique_ptr<oge::OrbitalGameEnvironment> environment;
};
#endif //ORBITALGAMEENV_OGE_INTERFACE_H
