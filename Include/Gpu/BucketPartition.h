#pragma once

#include "Utils.h"
#include <vector>

namespace Bucket_Partitioned_MDS
{
    class CVRP;
}

namespace Gpu
{
    /*
     * create_buckets: GPU bucket assignment + compaction.
     * Fills buckets[b] with depot followed by customers in angular wedge b.
     * Memory: O(N) device storage (coordinates + bucket metadata only).
     */
    void create_buckets(
        const Bucket_Partitioned_MDS::CVRP& cvrp,
        double                              alpha,
        std::vector<std::vector<node_t>>&   buckets);
}
