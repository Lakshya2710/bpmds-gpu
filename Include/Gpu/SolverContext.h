#pragma once

#include "Utils.h"
#include <memory>
#include <vector>

namespace Bucket_Partitioned_MDS
{
    class CVRP;
}

namespace Gpu
{
    /*
     * SolverContext: GPU session for one solve() call.
     * Holds O(N) device data, bucket layout, and reusable MST scratch buffers.
     */
    class SolverContext
    {
    public:
        SolverContext(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha);
        ~SolverContext();

        SolverContext(const SolverContext&)            = delete;
        SolverContext& operator=(const SolverContext&) = delete;

        void create_buckets(std::vector<std::vector<node_t>>& buckets);
        void construct_mst(int bucket_id, std::vector<std::vector<node_t>>& mst_adj);

        int num_buckets() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
