//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H
#define ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H

#include <vector>
#include <stdexcept>

#include "oge/vector/async_vectorizer.h"
#include "oge/vector/preprocessed_env.h"
#include "oge/vector/utils.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

namespace oge::vector
{
    /**
     * OGEVectorInterface provides a vectorized interface to the Orbital Game Environment.
     */
    class OGEVectorInterface
    {
    public:
        // TODO: COMPLETE THIS
    };
}

#endif //ORBITALGAMEENV_OGE_VECTOR_PYTHON_INTERFACE_H
