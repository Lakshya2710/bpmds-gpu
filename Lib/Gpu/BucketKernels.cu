#include "Gpu/DeviceData.h"
#include "Gpu/BucketPartition.h"
#include "Gpu/SolverContext.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cmath>

namespace Gpu
{
    namespace
    {
        __device__ bool vec_equal(double ax, double ay, double bx, double by)
        {
            return fabs(ax - bx) < EPS && fabs(ay - by) < EPS;
        }

        /*
         * Matches Unit_Vector_2D::is_in_between — true if (px,py) lies in [v1, v2).
         * All inputs are unit vectors in the depot-centered polar partition.
         */
        __device__ bool is_in_between(
            double px, double py,
            double v1x, double v1y,
            double v2x, double v2y)
        {
            if (vec_equal(v1x, v1y, v2x, v2y))
            {
                return false;
            }
            if (vec_equal(v1x, v1y, px, py))
            {
                return true;
            }
            if (vec_equal(v2x, v2y, px, py))
            {
                return false;
            }

            const double cross12 = v1x * v2y - v1y * v2x;
            const double cross1p = v1x * py - v1y * px;
            const double crossp2 = px * v2y - py * v2x;

            if (cross12 >= 0.0)
            {
                return cross1p >= 0.0 && crossp2 >= 0.0;
            }
            return !(cross1p < 0.0 && crossp2 < 0.0);
        }

        __global__ void assign_bucket_kernel(
            const double* x,
            const double* y,
            const double* sep_x,
            const double* sep_y,
            int*          bucket_id,
            int           N,
            int           num_buckets)
        {
            const int u = blockIdx.x * blockDim.x + threadIdx.x + 1;
            if (u >= N)
            {
                return;
            }

            const double dx = x[u] - x[0];
            const double dy = y[u] - y[0];
            const double norm = sqrt(dx * dx + dy * dy);

            if (norm < 1e-15)
            {
                bucket_id[u] = 0;
                return;
            }

            const double px = dx / norm;
            const double py = dy / norm;

            for (int i = 0; i < num_buckets; ++i)
            {
                if (is_in_between(px, py, sep_x[i], sep_y[i], sep_x[i + 1], sep_y[i + 1]))
                {
                    bucket_id[u] = i;
                    return;
                }
            }

            bucket_id[u] = num_buckets - 1;
        }

        __global__ void fill_sort_keys_kernel(
            int* keys,
            int* nodes,
            const int* bucket_id,
            int num_buckets,
            int N)
        {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total = num_buckets + (N - 1);

            if (idx >= total)
            {
                return;
            }

            if (idx < num_buckets)
            {
                keys[idx]  = idx;
                nodes[idx] = 0;
            }
            else
            {
                const int u = idx - num_buckets + 1;
                keys[idx]  = bucket_id[u];
                nodes[idx] = u;
            }
        }

        void launch_assign_buckets(DeviceCVRP& device)
        {
            const int N           = device.size();
            const int num_buckets = device.num_buckets();

            if (num_buckets == 1)
            {
                return;
            }

            const int threads = 256;
            const int blocks  = (N - 1 + threads - 1) / threads;

            assign_bucket_kernel<<<blocks, threads>>>(
                device.device_x(),
                device.device_y(),
                device.device_sep_x(),
                device.device_sep_y(),
                device.device_bucket_id(),
                N,
                num_buckets);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    BucketLayout assign_and_compact_buckets(DeviceCVRP& device)
    {
        const int N           = device.size();
        const int num_buckets = device.num_buckets();

        BucketLayout layout;
        layout.num_buckets = num_buckets;
        layout.offsets.resize(num_buckets + 1);

        if (num_buckets == 1)
        {
            layout.nodes.resize(N);
            for (int u = 0; u < N; ++u)
            {
                layout.nodes[u] = u;
            }
            layout.offsets[0] = 0;
            layout.offsets[1] = N;
            return layout;
        }

        launch_assign_buckets(device);

        const int total = num_buckets + (N - 1);
        thrust::device_vector<int> d_keys(total);
        thrust::device_vector<int> d_nodes(total);

        const int threads = 256;
        const int blocks  = (total + threads - 1) / threads;

        fill_sort_keys_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_nodes.data()),
            device.device_bucket_id(),
            num_buckets,
            N);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_nodes.begin());

        std::vector<int> h_keys(total);
        std::vector<int> h_nodes(total);
        CUDA_CHECK(cudaMemcpy(
            h_keys.data(),
            thrust::raw_pointer_cast(d_keys.data()),
            total * sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_nodes.data(),
            thrust::raw_pointer_cast(d_nodes.data()),
            total * sizeof(int),
            cudaMemcpyDeviceToHost));

        for (int b = 0; b <= num_buckets; ++b)
        {
            layout.offsets[b] = static_cast<int>(
                std::lower_bound(h_keys.begin(), h_keys.end(), b) - h_keys.begin());
        }

        layout.nodes.assign(h_nodes.begin(), h_nodes.end());
        return layout;
    }

    void create_buckets(
        const Bucket_Partitioned_MDS::CVRP& cvrp,
        double                              alpha,
        std::vector<std::vector<node_t>>&   buckets)
    {
        SolverContext ctx(cvrp, alpha);
        ctx.create_buckets(buckets);
    }
}
