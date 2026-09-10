#pragma once

#include "Gpu/DeviceData.h"
#include <vector>

namespace Gpu
{
    /*
     * Boruvka MST on one bucket. Uses on-the-fly squared Euclidean distance
     * from global coordinate arrays — O(k) memory, O(k^2 log k) work per bucket.
     *
     * @param mst_adj[out] adjacency lists indexed by local bucket position (0..k-1)
     */
    void construct_mst_boruvka(
        const DeviceCVRP&                   device,
        const int*                          d_bucket_nodes,
        const int*                          d_bucket_offsets,
        int                                 bucket_id,
        int                                 max_bucket_size,
        int*                                d_parent,
        double*                             d_cheapest_w,
        int*                                d_cheapest_u,
        int*                                d_cheapest_v,
        int*                                d_mst_u,
        int*                                d_mst_v,
        int*                                d_mst_edge_count,
        int*                                d_component_count,
        std::vector<std::vector<node_t>>&   mst_adj);
}
