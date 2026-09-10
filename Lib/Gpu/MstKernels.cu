#include "Gpu/MstKernels.h"

#include <cfloat>
#include <algorithm>
#include <vector>

namespace Gpu
{
    namespace
    {
        __device__ inline double dist2_global(
            int           gu,
            int           gv,
            const double* x,
            const double* y)
        {
            const double dx = x[gu] - x[gv];
            const double dy = y[gu] - y[gv];
            return dx * dx + dy * dy;
        }

        __device__ int uf_find(const int* parent, int i)
        {
            while (parent[i] != i)
            {
                i = parent[i];
            }
            return i;
        }

        __device__ void uf_union(int* parent, int a, int b)
        {
            while (a != b)
            {
                a = uf_find(parent, a);
                b = uf_find(parent, b);
                if (a == b)
                {
                    return;
                }
                if (a < b)
                {
                    const int old = atomicCAS(&parent[b], b, a);
                    if (old == b)
                    {
                        return;
                    }
                    b = old;
                }
                else
                {
                    const int old = atomicCAS(&parent[a], a, b);
                    if (old == a)
                    {
                        return;
                    }
                    a = old;
                }
            }
        }

        __device__ void atomic_min_edge(
            double* best_w,
            int*    best_u,
            int*    best_v,
            int     comp,
            double  w,
            int     u,
            int     v)
        {
            unsigned long long* addr =
                reinterpret_cast<unsigned long long*>(&best_w[comp]);
            unsigned long long assumed = *addr;
            unsigned long long old;

            do
            {
                old      = assumed;
                const double old_w = __longlong_as_double(old);
                if (w >= old_w)
                {
                    return;
                }
                assumed = atomicCAS(addr, old, __double_as_longlong(w));
            } while (assumed != old);

            best_u[comp] = u;
            best_v[comp] = v;
        }

        __global__ void init_parent_kernel(int* parent, int k)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < k)
            {
                parent[i] = i;
            }
        }

        __global__ void reset_cheapest_kernel(
            double* cheapest_w,
            int*    cheapest_u,
            int*    cheapest_v,
            int     k)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < k)
            {
                cheapest_w[i] = DBL_MAX;
                cheapest_u[i] = -1;
                cheapest_v[i] = -1;
            }
        }

        __global__ void count_components_kernel(
            const int* parent,
            int        k,
            int*       component_count)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < k && parent[i] == i)
            {
                atomicAdd(component_count, 1);
            }
        }

        /*
         * One thread per unordered pair (u, v), u < v.
         * Updates cheapest outgoing edge for each component.
         */
        __global__ void boruvka_find_cheapest_kernel(
            const int*    bucket_nodes,
            int           bucket_offset,
            const double* x,
            const double* y,
            const int*    parent,
            int           k,
            double*       cheapest_w,
            int*          cheapest_u,
            int*          cheapest_v)
        {
            const int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
            const int total_pairs = k * (k - 1) / 2;
            if (pair_idx >= total_pairs)
            {
                return;
            }

            int u         = 0;
            int remaining = pair_idx;
            while (u < k - 1)
            {
                const int count = k - 1 - u;
                if (remaining < count)
                {
                    break;
                }
                remaining -= count;
                ++u;
            }
            const int v = u + 1 + remaining;

            const int cu = uf_find(parent, u);
            const int cv = uf_find(parent, v);
            if (cu == cv)
            {
                return;
            }

            const int gu = bucket_nodes[bucket_offset + u];
            const int gv = bucket_nodes[bucket_offset + v];
            const double w = dist2_global(gu, gv, x, y);

            atomic_min_edge(cheapest_w, cheapest_u, cheapest_v, cu, w, u, v);
            atomic_min_edge(cheapest_w, cheapest_u, cheapest_v, cv, w, u, v);
        }

        __global__ void boruvka_merge_kernel(
            int*    parent,
            int     k,
            double* cheapest_w,
            int*    cheapest_u,
            int*    cheapest_v,
            int*    mst_u,
            int*    mst_v,
            int*    mst_edge_count)
        {
            const int c = blockIdx.x * blockDim.x + threadIdx.x;
            if (c >= k)
            {
                return;
            }

            if (cheapest_u[c] < 0 || cheapest_w[c] >= DBL_MAX)
            {
                return;
            }

            const int u = cheapest_u[c];
            const int v = cheapest_v[c];
            if (u < 0 || v < 0 || u >= k || v >= k)
            {
                return;
            }

            const int cu = uf_find(parent, u);
            const int cv = uf_find(parent, v);
            if (cu == cv)
            {
                return;
            }

            const int edge_idx = atomicAdd(mst_edge_count, 1);
            mst_u[edge_idx] = u;
            mst_v[edge_idx] = v;
            uf_union(parent, u, v);
        }

        int launch_blocks(int n, int threads)
        {
            return (n + threads - 1) / threads;
        }
    }

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
        std::vector<std::vector<node_t>>&   mst_adj)
    {
        int h_offsets[2];
        CUDA_CHECK(cudaMemcpy(
            h_offsets,
            d_bucket_offsets + bucket_id,
            2 * sizeof(int),
            cudaMemcpyDeviceToHost));

        const int bucket_offset = h_offsets[0];
        const int k             = h_offsets[1] - h_offsets[0];

        mst_adj.clear();
        mst_adj.resize(k);

        if (k <= 1)
        {
            return;
        }

        if (k == 2)
        {
            mst_adj[0].push_back(1);
            mst_adj[1].push_back(0);
            return;
        }

        const int threads = 256;

        init_parent_kernel<<<launch_blocks(k, threads), threads>>>(d_parent, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(d_mst_edge_count, 0, sizeof(int)));

        int h_components = 0;
        while (true)
        {
            CUDA_CHECK(cudaMemset(d_component_count, 0, sizeof(int)));
            count_components_kernel<<<launch_blocks(k, threads), threads>>>(
                d_parent, k, d_component_count);
            CUDA_CHECK(cudaMemcpy(
                &h_components,
                d_component_count,
                sizeof(int),
                cudaMemcpyDeviceToHost));

            if (h_components <= 1)
            {
                break;
            }

            reset_cheapest_kernel<<<launch_blocks(k, threads), threads>>>(
                d_cheapest_w, d_cheapest_u, d_cheapest_v, k);

            const int total_pairs = k * (k - 1) / 2;
            boruvka_find_cheapest_kernel<<<launch_blocks(total_pairs, threads), threads>>>(
                d_bucket_nodes,
                bucket_offset,
                device.device_x(),
                device.device_y(),
                d_parent,
                k,
                d_cheapest_w,
                d_cheapest_u,
                d_cheapest_v);

            boruvka_merge_kernel<<<launch_blocks(k, threads), threads>>>(
                d_parent,
                k,
                d_cheapest_w,
                d_cheapest_u,
                d_cheapest_v,
                d_mst_u,
                d_mst_v,
                d_mst_edge_count);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        int edge_count = 0;
        CUDA_CHECK(cudaMemcpy(
            &edge_count,
            d_mst_edge_count,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        if (edge_count <= 0)
        {
            return;
        }

        edge_count = std::min(edge_count, k - 1);

        std::vector<int> h_mst_u(edge_count);
        std::vector<int> h_mst_v(edge_count);
        CUDA_CHECK(cudaMemcpy(
            h_mst_u.data(),
            d_mst_u,
            edge_count * sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_mst_v.data(),
            d_mst_v,
            edge_count * sizeof(int),
            cudaMemcpyDeviceToHost));

        for (int e = 0; e < edge_count; ++e)
        {
            const int u = h_mst_u[e];
            const int v = h_mst_v[e];
            if (u < 0 || v < 0 || u >= k || v >= k)
            {
                continue;
            }
            mst_adj[u].push_back(v);
            mst_adj[v].push_back(u);
        }
    }
}
