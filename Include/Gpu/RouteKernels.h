#pragma once

#include "Gpu/MstKernels.h"
#include "Gpu/DeviceData.h"
#include <cuda_runtime.h>
#include <vector>

namespace Gpu
{
    /*
     * Per-bucket trial costs plus the single winning route set (not rho copies).
     */
    struct RouteTrialStorage
    {
        double* trial_costs   = nullptr; // num_buckets * rho
        int*    num_routes    = nullptr; // num_buckets (winner only)
        int*    route_offsets = nullptr; // num_buckets * (max_k + 1) (winner only)
        int*    route_nodes   = nullptr; // num_buckets * max_k (winner only)
        float2* bcoords       = nullptr; // num_buckets * max_k, local-order (x, y)
        double* bdemand       = nullptr; // num_buckets * max_k, local-order demand
        int*    trial_scratch = nullptr; // contiguous scratch for all buckets
    };

    void allocate_route_trial_storage(RouteTrialStorage& storage, int num_buckets, int rho, int max_k, size_t total_nodes = 0);
    void free_route_trial_storage(RouteTrialStorage& storage);

    /*
     * One CUDA stream per bucket. Each bucket launches min(rho, 2048) threads.
     */
    void run_all_route_trials_on_streams(
        const DeviceCVRP&       device,
        const std::vector<int>& h_offsets,
        const MstDeviceStorage& mst_storage,
        int                     rho,
        int                     max_k,
        double                  capacity,
        RouteTrialStorage&      storage,
        cudaStream_t*           streams);

    void fetch_best_routes_from_device(
        int                               bucket_id,
        int                               rho,
        int                               max_k,
        const RouteTrialStorage&          storage,
        std::vector<std::vector<node_t>>& routes,
        double&                           cost);
}
