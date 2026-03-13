//
// Created by baoyicui on 3/13/26.
//

#ifndef OGE_MATH_H
#define OGE_MATH_H

#include <cmath>
#include <Eigen/Dense>

namespace oge
{
    double signed_log(double x);

    /**
     * Element-wise symmetric logarithm: sign(x) * log(|x| + 1).
     * Maps large-magnitude values (e.g. km-scale positions) into a compact range
     * while preserving sign and passing through zero.
     */
    template <typename Derived>
    auto signed_log(const Eigen::MatrixBase<Derived>& x)
    {
        return x.unaryExpr([](typename Derived::Scalar v)
        {
            return std::copysign(std::log(std::abs(v) + 1.0), v);
        });
    }
}


#endif //OGE_MATH_H
