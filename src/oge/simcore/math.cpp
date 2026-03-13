//
// Created by baoyicui on 3/13/26.
//
#include "oge/simcore/math.h"

namespace oge
{
    double signed_log(double x)
    {
        const double sign = (x > 0.0) - (x < 0.0);
        return sign * std::log(std::abs(x) + 1.0);
    }
}
