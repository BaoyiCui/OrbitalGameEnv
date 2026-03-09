//
// Created by baoyicui on 3/9/26.
//

#ifndef ORBITALGAMEENV_PREPROCESSED_ENV_H
#define ORBITALGAMEENV_PREPROCESSED_ENV_H

#include <memory>
#include <vector>
#include <deque>
#include <random>
#include <string>
#include <algorithm>

// SIMD intrinsics for maxpooling optimization
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <oge/oge_interface.h>
#include "utils.h"

namespace oge::vector
{
    class PreprocessedEnv
    {
    public:
        PreprocessedEnv(
            const int env_id,
            const OGESettings& settings
        )
        {

        }

        void set_seed(const int seed);

        void reset();

    private:
        int env_id_;
        std::unique_ptr<OGEInterface> env_;

    };
}
#endif //ORBITALGAMEENV_PREPROCESSED_ENV_H
