//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_INTERFACE_H
#define ORBITALGAMEENV_OGE_INTERFACE_H

#include "oge/environment/orbital_game_environment.h"

#include <Eigen/Dense>
#include <memory>
#include <unordered_map>

namespace oge
{
    class OGEInterface
    {
    public:
        OGEInterface();
        ~OGEInterface() = default;

        /** Validate settings and construct the environment. Must be called before any other method. */
        void init();

        void getRewards(const std::vector<Eigen::Vector3d>& actions, std::vector<double>& rewards) const;
        void getObservations(std::vector<Eigen::VectorXd>& observations) const;
        bool getTerminal() const;
        bool getTruncated() const;
        double getCurrentTime() const;
        bool isCaptured() const;
        void act(const std::vector<Eigen::Vector3d>& actions);
        void reset();
        std::unordered_map<std::string, SatState> getSatStates() const;
        void resetWithStates(const std::unordered_map<std::string, SatState>& states);

        int getInt(const std::string& key, bool strict = false) const;
        float getFloat(const std::string& key, bool strict = false) const;
        bool getBool(const std::string& key, bool strict = false) const;
        const std::string& getString(const std::string& key, bool strict = false) const;
        void setInt(const std::string& key, const int value);
        void setFloat(const std::string& key, const float value);
        void setBool(const std::string& key, const bool value);
        void setString(const std::string& key, const std::string& value);

    public:
        std::unique_ptr<oge::OrbitalGameEnvironment> environment;
        std::unique_ptr<oge::OGESettings> settings;
    };
}


#endif //ORBITALGAMEENV_OGE_INTERFACE_H
