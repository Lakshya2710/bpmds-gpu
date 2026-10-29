#include "Gpu/NearestKernels.h"
#include "Gpu/CudaCommon.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Gpu
{
    namespace
    {
        constexpr int kMaxNeighbours = 32;
        constexpr int kTileSize      = 256;

        inline size_t align256(size_t bytes)
        {
            return (bytes + 255) & ~size_t(255);
        }

        __device__ __forceinline__ void evaluate_tile_candidates(
            const float2* __restrict__ survivor_coords,
            const int*    __restrict__ survivor_ids,
            int start, int end,
            float qx, float qy,
            int k,
            float* best_dist,
            int* best_id)
        {
            #pragma unroll 4
            for (int s = start; s < end; ++s)
            {
                const float2 cs = __ldg(&survivor_coords[s]);
                const float dx = cs.x - qx;
                const float dy = cs.y - qy;
                const float dist2 = dx * dx + dy * dy;

                if (dist2 < best_dist[k - 1])
                {
                    float current_dist = dist2;
                    int   current_id   = __ldg(&survivor_ids[s]);

                    #pragma unroll
                    for (int t = 0; t < kMaxNeighbours; ++t)
                    {
                        if (t >= k) break;

                        if (current_dist < best_dist[t])
                        {
                            const float tmp_dist = best_dist[t];
                            const int   tmp_id   = best_id[t];

                            best_dist[t] = current_dist;
                            best_id[t]   = current_id;

                            current_dist = tmp_dist;
                            current_id   = tmp_id;
                        }
                    }
                }
            }
        }

        __global__ void nearest_surviving_customers_kernel(
            const float2* __restrict__ survivor_coords,
            const int*    __restrict__ survivor_ids,
            const float4* __restrict__ tile_boxes,
            int                        num_tiles,
            int                        n_survivors,
            const float2* __restrict__ query_coords,
            int                        n_queries,
            int                        k,
            int*          __restrict__ neighbour_ids)
        {
            const int query = blockIdx.x * blockDim.x + threadIdx.x;
            if (query >= n_queries)
            {
                return;
            }

            // Force arrays into hardware registers
            float best_dist[kMaxNeighbours];
            int   best_id[kMaxNeighbours];

            #pragma unroll
            for (int i = 0; i < kMaxNeighbours; ++i)
            {
                best_dist[i] = 1e30f;
                best_id[i]   = -1;
            }

            const float2 q = __ldg(&query_coords[query]);
            const float qx = q.x;
            const float qy = q.y;

            // Pass 1: Find the tile whose bounding box is closest to (qx, qy).
            // All threads in a warp load tile_boxes[t] simultaneously with L1 warp broadcast.
            int   best_tile    = 0;
            float min_box_dist = 1e30f;

            for (int t = 0; t < num_tiles; ++t)
            {
                const float4 box = __ldg(&tile_boxes[t]);
                const float dx = fmaxf(0.0f, fmaxf(box.x - qx, qx - box.y));
                const float dy = fmaxf(0.0f, fmaxf(box.z - qy, qy - box.w));
                const float box_dist2 = dx * dx + dy * dy;

                if (box_dist2 < min_box_dist)
                {
                    min_box_dist = box_dist2;
                    best_tile    = t;
                }
            }

            // Evaluate best tile first!
            // Since this tile is geographically closest to (qx, qy), this immediately
            // initializes best_dist[k - 1] to a tight local bound.
            {
                const int start = best_tile * kTileSize;
                const int end   = min(start + kTileSize, n_survivors);
                evaluate_tile_candidates(
                    survivor_coords, survivor_ids,
                    start, end,
                    qx, qy, k,
                    best_dist, best_id);
            }

            // Pass 2: Check all other tiles.
            // If the tile bounding-box distance >= best_dist[k - 1], skip the entire tile!
            for (int t = 0; t < num_tiles; ++t)
            {
                if (t == best_tile) continue;

                const float4 box = __ldg(&tile_boxes[t]);
                const float dx = fmaxf(0.0f, fmaxf(box.x - qx, qx - box.y));
                const float dy = fmaxf(0.0f, fmaxf(box.z - qy, qy - box.w));
                const float box_dist2 = dx * dx + dy * dy;

                if (box_dist2 >= best_dist[k - 1])
                {
                    continue; // Pruned!
                }

                const int start = t * kTileSize;
                const int end   = min(start + kTileSize, n_survivors);
                evaluate_tile_candidates(
                    survivor_coords, survivor_ids,
                    start, end,
                    qx, qy, k,
                    best_dist, best_id);
            }

            int* out = neighbour_ids + query * k;
            for (int t = 0; t < k; ++t)
            {
                out[t] = best_id[t];
            }
        }
    }

    void gpu_warmup()
    {
        cudaFree(0);
    }

    void nearest_surviving_customers(
        const std::vector<Point>& nodes,
        const std::vector<int>&   survivor_ids,
        const std::vector<int>&   query_ids,
        int                       k,
        std::vector<int>&         neighbour_ids)
    {
        const int n_survivors = static_cast<int>(survivor_ids.size());
        const int n_queries   = static_cast<int>(query_ids.size());
        neighbour_ids.assign(static_cast<size_t>(n_queries) * static_cast<size_t>(k), -1);

        if (n_queries == 0 || n_survivors == 0 || k <= 0)
        {
            return;
        }
        if (k > kMaxNeighbours)
        {
            throw std::runtime_error("nearest neighbour k exceeds kernel capacity");
        }

        const int num_tiles = (n_survivors + kTileSize - 1) / kTileSize;

        // Precompute tile bounding boxes and pack survivor coordinates directly from nodes
        std::vector<float2> survivor_coords(n_survivors);
        std::vector<float4> tile_boxes(num_tiles);

        for (int t = 0; t < num_tiles; ++t)
        {
            const int start = t * kTileSize;
            const int end   = std::min(start + kTileSize, n_survivors);
            float min_x = 1e30f, max_x = -1e30f;
            float min_y = 1e30f, max_y = -1e30f;

            for (int i = start; i < end; ++i)
            {
                const int sid = survivor_ids[i];
                const float x = static_cast<float>(nodes[sid].x);
                const float y = static_cast<float>(nodes[sid].y);
                survivor_coords[i] = make_float2(x, y);

                if (x < min_x) min_x = x;
                if (x > max_x) max_x = x;
                if (y < min_y) min_y = y;
                if (y > max_y) max_y = y;
            }
            tile_boxes[t] = make_float4(min_x, max_x, min_y, max_y);
        }

        // Pack query coordinates directly from nodes
        std::vector<float2> query_coords(n_queries);
        for (int i = 0; i < n_queries; ++i)
        {
            const int qid = query_ids[i];
            query_coords[i] = make_float2(
                static_cast<float>(nodes[qid].x),
                static_cast<float>(nodes[qid].y));
        }

        // Single contiguous device allocation (cuts down 6 separate cudaMalloc calls)
        const size_t sz_survivor_coords = align256(n_survivors * sizeof(float2));
        const size_t sz_survivor_ids    = align256(n_survivors * sizeof(int));
        const size_t sz_query_coords    = align256(n_queries * sizeof(float2));
        const size_t sz_tile_boxes      = align256(num_tiles * sizeof(float4));
        const size_t sz_out             = align256(neighbour_ids.size() * sizeof(int));

        const size_t total_bytes = sz_survivor_coords + sz_survivor_ids +
                                   sz_query_coords + sz_tile_boxes + sz_out;

        char* d_blob = nullptr;
        CUDA_CHECK(cudaMalloc(&d_blob, total_bytes));

        float2* d_survivor_coords = reinterpret_cast<float2*>(d_blob);
        int*    d_survivor_ids    = reinterpret_cast<int*>(d_blob + sz_survivor_coords);
        float2* d_query_coords    = reinterpret_cast<float2*>(d_blob + sz_survivor_coords + sz_survivor_ids);
        float4* d_tile_boxes      = reinterpret_cast<float4*>(d_blob + sz_survivor_coords + sz_survivor_ids + sz_query_coords);
        int*    d_out             = reinterpret_cast<int*>(d_blob + sz_survivor_coords + sz_survivor_ids + sz_query_coords + sz_tile_boxes);

        CUDA_CHECK(cudaMemcpy(d_survivor_coords, survivor_coords.data(), n_survivors * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_survivor_ids, survivor_ids.data(), n_survivors * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_query_coords, query_coords.data(), n_queries * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tile_boxes, tile_boxes.data(), num_tiles * sizeof(float4), cudaMemcpyHostToDevice));

        constexpr int threads = 256;
        const int blocks = (n_queries + threads - 1) / threads;

        nearest_surviving_customers_kernel<<<blocks, threads>>>(
            d_survivor_coords, d_survivor_ids, d_tile_boxes, num_tiles,
            n_survivors, d_query_coords, n_queries, k, d_out);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(neighbour_ids.data(), d_out, neighbour_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));

        cudaFree(d_blob);
    }
}
