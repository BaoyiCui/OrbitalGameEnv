//
// Created by baoyicui on 3/6/26.
//

#include "oge_settings.h"

#include <cmath>
#include <stdexcept>
#include <sstream>

#include "oge/common/log.h"

namespace oge
{
    // void OGESettings::validate() const
    // {
    //     if (num_evaders != 1)
    //         throw std::invalid_argument("OGESettings: num_evaders must be 1");
    //     if (num_pursuers <= 0)
    //         throw std::invalid_argument("OGESettings: num_pursuers must be > 0");
    //
    //     if (sma_base <= 0.0)
    //         throw std::invalid_argument("OGESettings: sma_base must be > 0");
    //     if (ecc_base < 0.0 || ecc_base >= 1.0)
    //         throw std::invalid_argument("OGESettings: ecc_base must be in [0, 1)");
    //     if (incl_base < 0.0 || incl_base > M_PI)
    //         throw std::invalid_argument("OGESettings: incl_base must be in [0, pi] rad");
    //     if (RA_base < 0.0 || RA_base > 2.0 * M_PI)
    //         throw std::invalid_argument("OGESettings: RA_base must be in [0, 2*pi] rad");
    //     if (w_base < 0.0 || w_base > 2.0 * M_PI)
    //         throw std::invalid_argument("OGESettings: w_base must be in [0, 2*pi] rad");
    //     if (TA_base < 0.0 || TA_base > 2.0 * M_PI)
    //         throw std::invalid_argument("OGESettings: TA_base must be in [0, 2*pi] rad");
    //
    //     if (dv_init_p <= 0.0)
    //         throw std::invalid_argument("OGESettings: dv_init_p must be > 0");
    //     if (dv_init_e <= 0.0)
    //         throw std::invalid_argument("OGESettings: dv_init_e must be > 0");
    //     if (dv_max_per_step_p <= 0.0)
    //         throw std::invalid_argument("OGESettings: dv_max_per_step_p must be > 0");
    //     if (dv_max_per_step_e <= 0.0)
    //         throw std::invalid_argument("OGESettings: dv_max_per_step_e must be > 0");
    //     if (capture_distance <= 0.0)
    //         throw std::invalid_argument("OGESettings: capture_distance must be > 0");
    //     if (timestep <= 0.0)
    //         throw std::invalid_argument("OGESettings: timestep must be > 0");
    //     if (terminal_time <= 0.0)
    //         throw std::invalid_argument("OGESettings: terminal_time must be > 0");
    //     if (terminal_time < timestep)
    //         throw std::invalid_argument("OGESettings: terminal_time must be >= timestep");
    //
    //     if (sma_perturb_max <= 0.0)
    //         throw std::invalid_argument("OGESettings: sma_perturb_max must be > 0");
    //     if (sma_perturb_max >= sma_base)
    //         throw std::invalid_argument("OGESettings: sma_perturb_max must be < sma_base");
    //     if (dist_init_offset_min < 0.0)
    //         throw std::invalid_argument("OGESettings: dist_init_offset_min must be >= 0");
    //     if (dist_init_offset_min <= 0.0)
    //         throw std::invalid_argument("OGESettings: dist_init_offset_min must be > 0");
    //     if (dist_init_offset_min >= dist_init_offset_max)
    //         throw std::invalid_argument("OGESettings: dist_init_offset_min must be < dist_init_offset_max");
    //
    //
    //     if (reward_time_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_time_weight must be >= 0");
    //     if (reward_formation_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_formation_weight must be >= 0");
    //     if (reward_fuel_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_fuel_weight must be >= 0");
    //     if (reward_capture_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_capture_weight must be >= 0");
    //     if (reward_timeout_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_timeout_weight must be >= 0");
    //     if (reward_fuelout_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_fuelout_weight must be >= 0");
    //     if (reward_advantage_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_advantage_weight must be >= 0");
    //     if (reward_phase_dist_weight < 0.0)
    //         throw std::invalid_argument("OGESettings: reward_phase_dist_weight must be >= 0");
    //
    //     if (advantage_reward_horizon <= 0)
    //         throw std::invalid_argument("OGESettings: advantage_reward_horizon must be > 0");
    //     if (phase_dist_transition_dist <= 0.0)
    //         throw std::invalid_argument("OGESettings: phase_dist_transition_dist must be > 0");
    // }

    OGESettings::OGESettings()
    {
    }

    void OGESettings::validate() const
    {
    }

    void OGESettings::setInt(const std::string& key, const int value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(intSettings, key);
            setExternal(key, stream.str());
        }
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    void OGESettings::setFloat(const std::string& key, const float value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(floatSettings, key);
            setExternal(key, stream.str());
        }
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    void OGESettings::setBool(const std::string& key, const bool value)
    {
        std::ostringstream stream;
        stream << value;

        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, stream.str(), idx);
        }
        else
        {
            verifyVariableExistence(boolSettings, key);
            setExternal(key, stream.str());
        }
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    void OGESettings::setString(const std::string& key, const std::string& value)
    {
        if (int idx = getInternalPos(key) != -1)
        {
            setInternal(key, value, idx);
        }
        else
        {
            verifyVariableExistence(stringSettings, key);
            setExternal(key, value);
        }
    }


    int OGESettings::getInt(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            return (int)atoi(myInternalSettings[idx].value.c_str());
        }
        else
        {
            if ((idx = getExternalPos(key)) != -1)
            {
                return (int)atoi(myExternalSettings[idx].value.c_str());
            }
            else
            {
                if (strict)
                {
                    oge::Logger::Error << "No value found for key: " << key << ". ";
                    oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                    exit(-1);
                }
                else
                {
                    return -1;
                }
            }
        }
    }


    float OGESettings::getFloat(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            return (float)atof(myInternalSettings[idx].value.c_str());
        }
        else
        {
            if ((idx = getExternalPos(key)) != -1)
            {
                return (float)atof(myExternalSettings[idx].value.c_str());
            }
            else
            {
                if (strict)
                {
                    oge::Logger::Error << "No value found for key: " << key << ". ";
                    oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                    exit(-1);
                }
                else
                {
                    return -1.0;
                }
            }
        }
    }


    bool OGESettings::getBool(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            const std::string& value = myInternalSettings[idx].value;
            if (value == "1" || value == "true" || value == "True")
                return true;
            else if (value == "0" || value == "false" || value == "False")
                return false;
            else
                return false;
        }
        else if ((idx = getExternalPos(key)) != -1)
        {
            const std::string& value = myExternalSettings[idx].value;
            if (value == "1" || value == "true")
                return true;
            else if (value == "0" || value == "false")
                return false;
            else
                return false;
        }
        else
        {
            if (strict)
            {
                oge::Logger::Error << "No value found for key: " << key << ". ";
                oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            }
            else
            {
                return false;
            }
        }
    }


    const std::string& OGESettings::getString(const std::string& key, bool strict) const
    {
        // Try to find the named setting and answer its value
        int idx = -1;
        if ((idx = getInternalPos(key)) != -1)
        {
            return myInternalSettings[idx].value;
        }
        else if ((idx = getExternalPos(key)) != -1)
        {
            return myExternalSettings[idx].value;
        }
        else
        {
            if (strict)
            {
                oge::Logger::Error << "No value found for key: " << key << ". ";
                oge::Logger::Error << "Make sure all the settings files are loaded." << std::endl;
                exit(-1);
            }
            else
            {
                static std::string EmptyString("");
                return EmptyString;
            }
        }
    }


    int OGESettings::getInternalPos(const std::string& key) const
    {
        for (unsigned int i = 0; i < myInternalSettings.size(); ++i)
            if (myInternalSettings[i].key == key)
                return i;

        return -1;
    }


    int OGESettings::getExternalPos(const std::string& key) const
    {
        for (unsigned int i = 0; i < myExternalSettings.size(); ++i)
            if (myExternalSettings[i].key == key)
                return i;

        return -1;
    }


    int OGESettings::setInternal(const std::string& key, const std::string& value,
                                 int pos, bool useAsInitial)
    {
        int idx = -1;

        if (pos != -1 && pos >= 0 && pos < (int)myInternalSettings.size() &&
            myInternalSettings[pos].key == key)
        {
            idx = pos;
        }
        else
        {
            for (unsigned int i = 0; i < myInternalSettings.size(); ++i)
            {
                if (myInternalSettings[i].key == key)
                {
                    idx = i;
                    break;
                }
            }
        }

        if (idx != -1)
        {
            myInternalSettings[idx].key = key;
            myInternalSettings[idx].value = value;
            if (useAsInitial) myInternalSettings[idx].initialValue = value;

            /*cerr << "modify internal: key = " << key
                 << ", value  = " << value
                 << ", ivalue = " << myInternalSettings[idx].initialValue
                 << " @ index = " << idx
                 << endl;*/
        }
        else
        {
            Setting setting;
            setting.key = key;
            setting.value = value;
            if (useAsInitial) setting.initialValue = value;

            myInternalSettings.push_back(setting);
            idx = myInternalSettings.size() - 1;

            /*cerr << "insert internal: key = " << key
                 << ", value  = " << value
                 << ", ivalue = " << setting.initialValue
                 << " @ index = " << idx
                 << endl;*/
        }

        return idx;
    }


    int OGESettings::setExternal(const std::string& key, const std::string& value,
                                 int pos, bool useAsInitial)
    {
        int idx = -1;

        if (pos != -1 && pos >= 0 && pos < (int)myExternalSettings.size() &&
            myExternalSettings[pos].key == key)
        {
            idx = pos;
        }
        else
        {
            for (unsigned int i = 0; i < myExternalSettings.size(); ++i)
            {
                if (myExternalSettings[i].key == key)
                {
                    idx = i;
                    break;
                }
            }
        }

        if (idx != -1)
        {
            myExternalSettings[idx].key = key;
            myExternalSettings[idx].value = value;
            if (useAsInitial) myExternalSettings[idx].initialValue = value;

            /*cerr << "modify external: key = " << key
                 << ", value = " << value
                 << " @ index = " << idx
                 << endl;*/
        }
        else
        {
            Setting setting;
            setting.key = key;
            setting.value = value;
            if (useAsInitial) setting.initialValue = value;

            myExternalSettings.push_back(setting);
            idx = myExternalSettings.size() - 1;

            /*cerr << "insert external: key = " << key
                 << ", value = " << value
                 << " @ index = " << idx
                 << endl;*/
        }

        return idx;
    }

    OGESettings::OGESettings(const OGESettings&)
    {
    }

    OGESettings& OGESettings::operator=(const OGESettings&)
    {
    }
}
