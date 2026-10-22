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

            double best_dist[kMaxNeighbours];
            int    best_id[kMaxNeighbours];
            int    filled = 0;
            int    farthest = 0;

            const double qx = query_x[query];
            const double qy = query_y[query];

            for (int i = 0; i < n_survivors; ++i)
            {
                const double dx = survivor_x[i] - qx;
                const double dy = survivor_y[i] - qy;
                const double dist2 = dx * dx + dy * dy;

                if (filled < k)
                {
                    best_dist[filled] = dist2;
                    best_id[filled] = survivor_ids[i];
                    if (dist2 > best_dist[farthest])
                    {
                        farthest = filled;
                    }
                    ++filled;
                    continue;
                }

                if (dist2 >= best_dist[farthest])
                {
                    continue;
                }

                best_dist[farthest] = dist2;
                best_id[farthest] = survivor_ids[i];
                farthest = 0;
                for (int t = 1; t < k; ++t)
                {
                    if (best_dist[t] > best_dist[farthest])
                    {
                        farthest = t;
                    }
                }
            }

            int* out = neighbour_ids + query * k;
            for (int t = 0; t < k; ++t)
            {
                out[t] = (t < filled) ? best_id[t] : -1;
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
