//
// Created by baoyicui on 3/9/26.
//

#ifndef ORBITALGAMEENV_ASYNC_VECTORIZER_H
#define ORBITALGAMEENV_ASYNC_VECTORIZER_H

#include <vector>
#include <thread>

#include "oge/external/ThreadPool.h"
#include "utils.h"
#include "preprocessed_env.h"

namespace oge::vector
{
    class AsyncVectorizer
    {
        explicit AsyncVectorizer(
            const int num_envs,
            const int batch_size = 0,
            const int num_threads = 0,
            const int thread_affinity_offset = -1,
            const
        );
    };
}

#endif //ORBITALGAMEENV_ASYNC_VECTORIZER_H
