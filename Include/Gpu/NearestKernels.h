#pragma once

#include "Utils.h"
#include <vector>

namespace Gpu
{
    // Early warm-up for the CUDA runtime context
    void gpu_warmup();

    // For each query point, the k nearest survivor ids.
    // neighbour_ids has size query_count * k. Missing slots are -1.
    void nearest_surviving_customers(
        const std::vector<Point>& nodes,
        const std::vector<int>&   survivor_ids,
        const std::vector<int>&   query_ids,
        int                       k,
        std::vector<int>&         neighbour_ids);
}
