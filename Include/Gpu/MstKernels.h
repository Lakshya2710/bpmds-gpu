#pragma once

#include "Gpu/DeviceData.h"
#include <cuda_runtime.h>
#include <vector>

namespace Gpu
{
    struct MstBucketScratch
    {
        int*    parent          = nullptr;
        double* cheapest_w      = nullptr;
        int*    cheapest_u      = nullptr;
        int*    cheapest_v      = nullptr;
        int*    mst_u           = nullptr;
        int*    mst_v           = nullptr;
        int*    mst_edge_count  = nullptr;
        int*    component_count = nullptr;
    };

    struct MstDeviceStorage
    {
        int* row_offsets = nullptr; // num_buckets * (max_k + 1), padded CSR rows
        int* cols        = nullptr; // num_buckets * 2 * max_k, padded CSR cols
        int* bucket_k    = nullptr; // num_buckets actual sizes
    };

    void allocate_mst_bucket_scratch(MstBucketScratch& scratch, int max_k);
    void free_mst_bucket_scratch(MstBucketScratch& scratch);

    void allocate_mst_device_storage(MstDeviceStorage& storage, int num_buckets, int max_k);
    void free_mst_device_storage(MstDeviceStorage& storage);

    /*
     * Build Boruvka MST for one bucket on a CUDA stream.
     * Writes CSR into storage row_offsets/cols at bucket_id slice.
     */
    void construct_mst_boruvka_streamed(
        const DeviceCVRP&     device,
        int                   bucket_id,
        int                   bucket_offset,
        int                   k,
        int                   max_k,
        MstBucketScratch&     scratch,
        MstDeviceStorage&     storage,
        cudaStream_t          stream);

    /*
     * Legacy host adjacency output (kept for compatibility).
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

    /*
     * Step A: one CUDA stream per bucket, all MSTs built concurrently (no waves).
     */
    void build_all_msts_on_streams(
        const DeviceCVRP&     device,
        const int*            d_bucket_offsets,
        const std::vector<int>& h_bucket_k,
        int                   max_k,
        MstBucketScratch*     per_bucket_scratch,
        MstDeviceStorage&     storage,
        cudaStream_t*         streams);
}
