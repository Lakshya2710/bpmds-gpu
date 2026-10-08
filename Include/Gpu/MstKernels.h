#pragma once

#include "Gpu/DeviceData.h"
#include <cuda_runtime.h>
#include <vector>

namespace Gpu
{
    /*
     * Temporary Boruvka workspace for one in-flight bucket MST.
     * Parallel MSTs need one copy per concurrent bucket (size B with B streams).
     */
    struct MstBucketScratch
    {
        int*                parent          = nullptr;
        unsigned long long* cheapest_edge   = nullptr;
        int*                cheapest_v      = nullptr;
        int*                mst_u           = nullptr;
        int*                mst_v           = nullptr;
        int*                mst_edge_count  = nullptr;
        int*                component_count = nullptr;
    };

    /*
     * Final per-bucket MST in padded CSR.
     * Bucket b uses row_offsets[b * (max_k+1) ...] and cols[b * 2 * max_k ...].
     * Only the first k+1 offsets and 2*(k-1) cols are valid for that bucket.
     */
    struct MstDeviceStorage
    {
        int* row_offsets = nullptr;
        int* cols        = nullptr;
    };

    void allocate_mst_bucket_scratch(MstBucketScratch& scratch, int max_k);
    void free_mst_bucket_scratch(MstBucketScratch& scratch);

    void allocate_mst_device_storage(MstDeviceStorage& storage, int num_buckets, int max_k);
    void free_mst_device_storage(MstDeviceStorage& storage);

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
     * One CUDA stream per bucket. Host threads keep each bucket's Boruvka
     * while-loop feeding its own stream so MSTs overlap.
     */
    void build_all_msts_on_streams(
        const DeviceCVRP&       device,
        const std::vector<int>& h_offsets,
        int                     max_k,
        MstBucketScratch*       per_bucket_scratch,
        MstDeviceStorage&       storage,
        cudaStream_t*           streams);
}
