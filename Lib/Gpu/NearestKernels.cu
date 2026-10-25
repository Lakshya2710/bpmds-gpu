#include "Gpu/NearestKernels.h"
#include "Gpu/CudaCommon.h"

namespace Gpu
{
    namespace
    {
        constexpr int kMaxNeighbours = 32;

        __global__ void nearest_surviving_customers_kernel(
            const double* __restrict__ survivor_x,
            const double* __restrict__ survivor_y,
            const int*    __restrict__ survivor_ids,
            int                        n_survivors,
            const double* __restrict__ query_x,
            const double* __restrict__ query_y,
            int                        n_queries,
            int                        k,
            int*                       neighbour_ids)
        {
            const int query = blockIdx.x * blockDim.x + threadIdx.x;
            if (query >= n_queries)
            {
                return;
            }

            // Force arrays into hardware registers 
            double best_dist[kMaxNeighbours];
            int    best_id[kMaxNeighbours];

            #pragma unroll
            for (int i = 0; i < kMaxNeighbours; ++i)
            {
                best_dist[i] = 1e30; 
                best_id[i]   = -1;
            }

            const double qx = __ldg(&query_x[query]);
            const double qy = __ldg(&query_y[query]);

            for (int i = 0; i < n_survivors; ++i)
            {
                // __ldg is CRITICAL here: the entire warp reads the same 'i' simultaneously, 
                // allowing the L1 cache to broadcast the coordinates to all 32 threads instantly.
                const double sx = __ldg(&survivor_x[i]);
                const double sy = __ldg(&survivor_y[i]);
                
                const double dx = sx - qx;
                const double dy = sy - qy;
                const double dist2 = dx * dx + dy * dy;

                // Only engage the update logic if it beats our worst tracked neighbor
                if (dist2 < best_dist[k - 1])
                {
                    double current_dist = dist2;
                    // Delay reading the ID from RAM until we mathematically know it's a hit
                    int current_id = __ldg(&survivor_ids[i]); 
                    
                    // Fully unrolled "Bubble Down" insertion sort
                    #pragma unroll
                    for (int t = 0; t < kMaxNeighbours; ++t)
                    {
                        if (t >= k) break; 
                        
                        if (current_dist < best_dist[t])
                        {
                            double tmp_dist = best_dist[t];
                            int tmp_id = best_id[t];
                            
                            best_dist[t] = current_dist;
                            best_id[t] = current_id;
                            
                            current_dist = tmp_dist;
                            current_id = tmp_id;
                        }
                    }
                }
            }

            int* out = neighbour_ids + query * k;
            for (int t = 0; t < k; ++t)
            {
                out[t] = best_id[t];
            }
        }
    }

    void nearest_surviving_customers(
        const std::vector<double>& survivor_x,
        const std::vector<double>& survivor_y,
        const std::vector<int>&    survivor_ids,
        const std::vector<double>& query_x,
        const std::vector<double>& query_y,
        int                        k,
        std::vector<int>&          neighbour_ids)
    {
        const int n_survivors = static_cast<int>(survivor_ids.size());
        const int n_queries = static_cast<int>(query_x.size());
        neighbour_ids.assign(static_cast<size_t>(n_queries) * static_cast<size_t>(k), -1);

        if (n_queries == 0 || n_survivors == 0 || k <= 0)
        {
            return;
        }
        if (k > kMaxNeighbours)
        {
            throw std::runtime_error("nearest neighbour k exceeds kernel capacity");
        }

        double* d_sx = nullptr;
        double* d_sy = nullptr;
        int*    d_ids = nullptr;
        double* d_qx = nullptr;
        double* d_qy = nullptr;
        int*    d_out = nullptr;

        CUDA_CHECK(cudaMalloc(&d_sx, n_survivors * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sy, n_survivors * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ids, n_survivors * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_qx, n_queries * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_qy, n_queries * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_out, neighbour_ids.size() * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_sx, survivor_x.data(), n_survivors * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sy, survivor_y.data(), n_survivors * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ids, survivor_ids.data(), n_survivors * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qx, query_x.data(), n_queries * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_qy, query_y.data(), n_queries * sizeof(double), cudaMemcpyHostToDevice));

        constexpr int threads = 128;
        const int blocks = (n_queries + threads - 1) / threads;
        nearest_surviving_customers_kernel<<<blocks, threads>>>(
            d_sx, d_sy, d_ids, n_survivors, d_qx, d_qy, n_queries, k, d_out);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(neighbour_ids.data(), d_out, neighbour_ids.size() * sizeof(int), cudaMemcpyDeviceToHost));

        cudaFree(d_sx);
        cudaFree(d_sy);
        cudaFree(d_ids);
        cudaFree(d_qx);
        cudaFree(d_qy);
        cudaFree(d_out);
    }
}
