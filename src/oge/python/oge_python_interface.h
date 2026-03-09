//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H
#define ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/filesystem.h>

#include "oge/oge_interface.h"

#ifdef BUILD_VECTOR_LIB
    #include "oge_python_interface.h"
#endif

namespace nb = nanobind;
void init_vector_module(nb::module_ &m);

namespace oge
{
    class OGEPythonInterface : public OGEInterface
    {
    public:
        using OGEInterface::OGEInterface;

        void getRewards(const std::vector<Eigen::Vector3d>& actions, std::vector<double>& rewards) const;
        void getObservations(std::vector<Eigen::VectorXd>& observations) const;
        bool getTerminal() const;
        bool getTruncated() const;
        void act(std::vector<Eigen::Vector3d>& actions);
        void reset();

        nb::ndarray<nb::numpy, double> getRewards(nb::ndarray<nb::numpy, double> actions);
        nb::ndarray<nb::numpy, double> getObservations();
    };
}

NB_MODULE(_oge_py, m)
{
    m.attr("__version__" = );
}

#endif //ORBITALGAMEENV_OGE_PYTHON_INTERFACE_H