//
// Created by baoyicui on 2/21/26.
//
#include "oge_state.h"

#include <iomanip>

namespace oge
{
    std::ostream& operator<<(std::ostream& os, const SatState& s)
    {
        // 保存原始格式，防止污染全局 cout 状态
        std::ios_base::fmtflags f(os.flags());

        os << std::fixed << std::setprecision(4)
            << "[SatState]\n"
            << "    R: [" << s.r_j2000.transpose() << "]"
            << "    V: [" << s.v_j2000.transpose() << "]"
            << "    dv_remain: " << s.dv_remain
            << "    Status: " << (s.is_alive ? "Alive" : "Dead")
            << "\n";

        os.flags(f);
        return os;
    }
}
