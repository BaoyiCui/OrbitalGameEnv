//
// Created by baoyicui on 2/21/26.
//

#ifndef ORBITALGAMEENV_OGE_STATE_H
#define ORBITALGAMEENV_OGE_STATE_H

#include <Eigen/Dense>

namespace oge
{
    // satellite's state
    struct SatState
    {
        Eigen::Vector3d r_j2000;
        Eigen::Vector3d v_j2000;
        double dv_remain;
        bool is_alive;

        friend std::ostream& operator<<(std::ostream& os, const SatState& s);
    };

    std::ostream& operator<<(std::ostream& os, const SatState& s);
}

#endif //ORBITALGAMEENV_OGE_STATE_H
