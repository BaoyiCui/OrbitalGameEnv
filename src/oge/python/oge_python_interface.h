//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
#define ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H

#include <optional>
#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/eigen/dense.h>

#include "oge/oge_interface.h"
#include "version.h"

#ifdef BUILD_VECTOR_LIB
#include "oge_vector_python_interface.h"
#endif

namespace nb = nanobind;
using namespace nb::literals;
void init_vector_module(nb::module_& m);

namespace oge
{
    class OGEPythonInterface : public OGEInterface
    {
    public:
        using OGEInterface::OGEInterface;

        nb::ndarray<nb::numpy, double> getRewards(nb::ndarray<nb::numpy, const double> actions) const;
        nb::ndarray<nb::numpy, double> getObservations() const;
        void act(nb::ndarray<nb::numpy, double> actions);
    };
}

NB_MODULE(_oge_py, m)
{
    m.attr("__version__") = OGE_VERSION;

    nb::class_<oge::SatState>(m, "SatState")
        .def(nb::init<>())
        .def_rw("r_j2000", &oge::SatState::r_j2000)
        .def_rw("v_j2000", &oge::SatState::v_j2000)
        .def_rw("dv_remain", &oge::SatState::dv_remain)
        .def_rw("is_alive", &oge::SatState::is_alive)
        .def("__repr__", [](const oge::SatState& s)
        {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });

    nb::class_<oge::OGESettings>(m, "OGESettings")
        .def(nb::init<>())
        .def("validate", &oge::OGESettings::validate)
        .def("set_int", &oge::OGESettings::setInt)
        .def("set_float", &oge::OGESettings::setFloat)
        .def("set_bool", &oge::OGESettings::setBool)
        .def("set_string", &oge::OGESettings::setString)
        .def("get_int", &oge::OGESettings::getInt)
        .def("get_float", &oge::OGESettings::getFloat)
        .def("get_bool", &oge::OGESettings::getBool)
        .def("get_string", &oge::OGESettings::getString);

    nb::class_<oge::OGEPythonInterface>(m, "OGEInterface")
        .def(nb::init<>())
        .def_prop_ro("settings", [](oge::OGEPythonInterface& self) -> oge::OGESettings&
        {
            return *self.settings;
        }, nb::rv_policy::reference_internal)
        .def("init", &oge::OGEPythonInterface::init)
        .def("get_rewards", &oge::OGEPythonInterface::getRewards)
        .def("get_observations", &oge::OGEPythonInterface::getObservations)
        .def("act", &oge::OGEPythonInterface::act)
        .def("reset", &oge::OGEPythonInterface::reset)
        .def("init", &oge::OGEPythonInterface::init)
        .def("setInt", &oge::OGEPythonInterface::setInt)
        .def("setFloat", &oge::OGEPythonInterface::setFloat)
        .def("setBool", &oge::OGEPythonInterface::setBool)
        .def("setString", &oge::OGEPythonInterface::setString)
        .def("getInt", &oge::OGEPythonInterface::getInt)
        .def("getFloat", &oge::OGEPythonInterface::getFloat)
        .def("getBool", &oge::OGEPythonInterface::getBool)
        .def("getString", &oge::OGEPythonInterface::getString);

#ifdef BUILD_VECTOR_LIB
    init_vector_module(m);
#endif
}

#endif //ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
