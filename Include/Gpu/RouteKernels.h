#pragma once

#include "Gpu/MstKernels.h"
#include "Gpu/DeviceData.h"
#include <cuda_runtime.h>
#include <vector>

namespace Gpu
{
    struct RouteTrialStorage
    {
        double* trial_costs     = nullptr; // num_buckets * rho
        int*    trial_num_routes = nullptr; // num_buckets * rho
        int*    route_offsets   = nullptr; // num_buckets * rho * (max_k + 1)
        int*    route_nodes     = nullptr; // num_buckets * rho * max_k (global node ids, concatenated)
        int*    trial_scratch   = nullptr; // num_buckets * rho * scratch_stride
    };

    void allocate_route_trial_storage(RouteTrialStorage& storage, int num_buckets, int rho, int max_k);
    void free_route_trial_storage(RouteTrialStorage& storage);

    int route_trial_scratch_stride(int max_k);

    /*
     * Step B: one CUDA stream per bucket, all rho trials launched at once (no waves).
     * Each thread runs one independent randomized DFS trial.
     */
    void run_all_route_trials_on_streams(
        const DeviceCVRP&       device,
        const int*              d_bucket_offsets,
        const std::vector<int>& h_bucket_k,
        const MstDeviceStorage& mst_storage,
        int                     rho,
        int                     max_k,
        double                  capacity,
        RouteTrialStorage&      storage,
        cudaStream_t*           streams);

    /*
     * Copy the lowest-cost trial for bucket_id back to host route vectors.
     */
    void fetch_best_routes_from_device(
        int                               bucket_id,
        int                               rho,
        int                               max_k,
        const RouteTrialStorage&          storage,
        std::vector<std::vector<node_t>>& routes,
        double&                           cost);
}
