#pragma once

#include <vector>

namespace Gpu
{
    // For each query point, the k nearest survivor ids.
    // neighbour_ids has size query_count * k. Missing slots are -1.
    void nearest_surviving_customers(
        const std::vector<double>& survivor_x,
        const std::vector<double>& survivor_y,
        const std::vector<int>&    survivor_ids,
        const std::vector<double>& query_x,
        const std::vector<double>& query_y,
        int                        k,
        std::vector<int>&          neighbour_ids);
}
