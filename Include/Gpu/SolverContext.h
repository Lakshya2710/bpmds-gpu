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
     * GPU session for one solve() call.
     * Keeps one CUDA stream and one MstBucketScratch per bucket so MST and
     * rho work can overlap across buckets.
     */
    class SolverContext
    {
    public:
        SolverContext(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha);
        ~SolverContext();

        SolverContext(const SolverContext&)            = delete;
        SolverContext& operator=(const SolverContext&) = delete;

        void partition();

        void build_all_msts_streamed();
        void run_all_route_trials_streamed(int rho);

        void fetch_best_routes_for_bucket(
            int                               bucket_id,
            int                               rho,
            std::vector<std::vector<node_t>>& routes,
            double&                           cost);

        int num_buckets() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
