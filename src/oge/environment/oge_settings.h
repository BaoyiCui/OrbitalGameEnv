//
// Created by baoyicui on 3/6/26.
//

#ifndef ORBITALGAMEENV_SETTINGS_H
#define ORBITALGAMEENV_SETTINGS_H
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

namespace oge
{
    class OGESettings
    {
    public:
        OGESettings();
        virtual ~OGESettings();
        OGESettings(const OGESettings&) = delete;
        OGESettings& operator =(const OGESettings&) = delete;

        void validate() const;

        /**
          Get the value assigned to the specified key.  If the key does
          not exist then -1 is returned.

          @param key The key of the setting to lookup
          @return The integer value of the setting
        */
        int getInt(const std::string& key, bool strict = false) const;

        /**
          Get the value assigned to the specified key.  If the key does
          not exist then -1.0 is returned.

          @param key The key of the setting to lookup
          @return The floating point value of the setting
        */
        float getFloat(const std::string& key, bool strict = false) const;

        /**
          Get the value assigned to the specified key.  If the key does
          not exist then false is returned.

          @param key The key of the setting to lookup
          @return The boolean value of the setting
        */
        bool getBool(const std::string& key, bool strict = false) const;

        /**
          Get the value assigned to the specified key.  If the key does
          not exist then the empty string is returned.

          @param key The key of the setting to lookup
          @return The string value of the setting
        */
        const std::string& getString(const std::string& key, bool strict = false) const;

        /**
      Set the value associated with key to the given value.

      @param key   The key of the setting
      @param value The value to assign to the setting
    */
        void setInt(const std::string& key, const int value);

        /**
          Set the value associated with key to the given value.

          @param key   The key of the setting
          @param value The value to assign to the setting
        */
        void setFloat(const std::string& key, const float value);

        /**
          Set the value associated with key to the given value.

          @param key   The key of the setting
          @param value The value to assign to the setting
        */
        void setBool(const std::string& key, const bool value);

        /**
          Set the value associated with key to the given value.

          @param key   The key of the setting
          @param value The value to assign to the setting
        */
        void setString(const std::string& key, const std::string& value);

        /** Copy all settings from this object into dst. */
        void copyTo(OGESettings& dst) const;

    private:
        // Trim leading and following witespace from a string
        static std::string trim(std::string& str)
        {
            std::string::size_type first = str.find_first_not_of(' ');
            return (first == std::string::npos)
                       ? std::string()
                       : str.substr(first, str.find_last_not_of(' ') - first + 1);
        }

        static std::string toString(int value)
        {
            std::ostringstream os;
            os << value;
            return os.str();
        }

        static std::string toString(float value)
        {
            std::ostringstream os;
            os << value;
            return os.str();
        }

        static std::string toString(double value)
        {
            std::ostringstream os;
            os << value;
            return os.str();
        }

    protected:
        struct Setting
        {
            std::string key;
            std::string value;
            std::string initialValue;
        };

        typedef std::vector<Setting> SettingsArray;

        const SettingsArray& getInternalSettings() const
        {
            return myInternalSettings;
        }

        const SettingsArray& getExternalSettings() const
        {
            return myExternalSettings;
        }

        /** Get position in specified array of 'key' */
        int getInternalPos(const std::string& key) const;
        int getExternalPos(const std::string& key) const;

        /** Add key, value pair to specified array at specified position */
        int setInternal(const std::string& key, const std::string& value,
                        int pos = -1, bool useAsInitial = false);
        int setExternal(const std::string& key, const std::string& value,
                        int pos = -1, bool useAsInitial = false);

    private:
        // Maps containing all external settings an user can 
        // define and their respectives default values.
        std::map<std::string, int> intSettings;
        std::map<std::string, bool> boolSettings;
        std::map<std::string, float> floatSettings;
        std::map<std::string, std::string> stringSettings;
        template <typename ValueType>
        void verifyVariableExistence(
            const std::map<std::string, ValueType>& dict,
            const std::string& key) const;

        SettingsArray myInternalSettings;
        SettingsArray myExternalSettings;
    };
}
#endif //ORBITALGAMEENV_SETTINGS_H
