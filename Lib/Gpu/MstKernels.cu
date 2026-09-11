#include "Gpu/MstKernels.h"

#include <cfloat>
#include <algorithm>
#include <thread>
#include <vector>

namespace Gpu
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

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
                old                = assumed;
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
            const int pair_idx    = blockIdx.x * blockDim.x + threadIdx.x;
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

        __global__ void init_mst_k2_kernel(int* row_offsets, int* cols, int max_k)
        {
            row_offsets[0] = 0;
            row_offsets[1] = 1;
            row_offsets[2] = 2;
            cols[0]        = 1;
            cols[1]        = 0;
            for (int i = 3; i <= max_k; ++i)
            {
                row_offsets[i] = 2;
            }
        }

        __global__ void zero_degrees_kernel(int* degrees, int k)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < k)
            {
                degrees[i] = 0;
            }
        }

        __global__ void count_mst_degrees_kernel(
            const int* mst_u,
            const int* mst_v,
            int        edge_count,
            int*       degrees)
        {
            const int e = blockIdx.x * blockDim.x + threadIdx.x;
            if (e >= edge_count)
            {
                return;
            }
            atomicAdd(&degrees[mst_u[e]], 1);
            atomicAdd(&degrees[mst_v[e]], 1);
        }

        __global__ void build_row_offsets_kernel(const int* degrees, int k, int* row_offsets)
        {
            if (threadIdx.x != 0 || blockIdx.x != 0)
            {
                return;
            }
            row_offsets[0] = 0;
            for (int i = 0; i < k; ++i)
            {
                row_offsets[i + 1] = row_offsets[i] + degrees[i];
            }
        }

        __global__ void fill_mst_cols_kernel(
            const int* mst_u,
            const int* mst_v,
            int        edge_count,
            int*       degrees,
            int*       row_offsets,
            int*       cols)
        {
            const int e = blockIdx.x * blockDim.x + threadIdx.x;
            if (e >= edge_count)
            {
                return;
            }

            const int u = mst_u[e];
            const int v = mst_v[e];

            const int pos_u = atomicAdd(&degrees[u], 1);
            const int pos_v = atomicAdd(&degrees[v], 1);

            cols[row_offsets[u] + pos_u] = v;
            cols[row_offsets[v] + pos_v] = u;
        }

        __global__ void set_bucket_k_kernel(int* bucket_k, int bucket_id, int k)
        {
            if (threadIdx.x == 0 && blockIdx.x == 0)
            {
                bucket_k[bucket_id] = k;
            }
        }

        int launch_blocks(int n, int threads)
        {
            return (n + threads - 1) / threads;
        }

        void build_csr_from_edges(
            int*       row_offsets,
            int*       cols,
            int*       degrees,
            const int* mst_u,
            const int* mst_v,
            int        edge_count,
            int        k,
            cudaStream_t stream)
        {
            zero_degrees_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                degrees, k);

            count_mst_degrees_kernel<<<launch_blocks(edge_count, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                mst_u, mst_v, edge_count, degrees);

            build_row_offsets_kernel<<<1, 1, 0, stream>>>(degrees, k, row_offsets);

            zero_degrees_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                degrees, k);

            fill_mst_cols_kernel<<<launch_blocks(edge_count, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                mst_u, mst_v, edge_count, degrees, row_offsets, cols);
        }
    }

    void allocate_mst_bucket_scratch(MstBucketScratch& scratch, int max_k)
    {
        if (max_k <= 1)
        {
            return;
        }
        CUDA_CHECK(cudaMalloc(&scratch.parent, max_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.cheapest_w, max_k * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&scratch.cheapest_u, max_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.cheapest_v, max_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.mst_u, (max_k - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.mst_v, (max_k - 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.mst_edge_count, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&scratch.component_count, sizeof(int)));
    }

    void free_mst_bucket_scratch(MstBucketScratch& scratch)
    {
        cudaFree(scratch.parent);
        cudaFree(scratch.cheapest_u);
        cudaFree(scratch.cheapest_v);
        cudaFree(scratch.cheapest_w);
        cudaFree(scratch.mst_u);
        cudaFree(scratch.mst_v);
        cudaFree(scratch.mst_edge_count);
        cudaFree(scratch.component_count);
        scratch = {};
    }

    void allocate_mst_device_storage(MstDeviceStorage& storage, int num_buckets, int max_k)
    {
        CUDA_CHECK(cudaMalloc(&storage.row_offsets, num_buckets * (max_k + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.cols, num_buckets * 2 * max_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.bucket_k, num_buckets * sizeof(int)));
        CUDA_CHECK(cudaMemset(storage.row_offsets, 0, num_buckets * (max_k + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(storage.cols, 0, num_buckets * 2 * max_k * sizeof(int)));
        CUDA_CHECK(cudaMemset(storage.bucket_k, 0, num_buckets * sizeof(int)));
    }

    void free_mst_device_storage(MstDeviceStorage& storage)
    {
        cudaFree(storage.row_offsets);
        cudaFree(storage.cols);
        cudaFree(storage.bucket_k);
        storage = {};
    }

    void construct_mst_boruvka_streamed(
        const DeviceCVRP& device,
        int               bucket_id,
        int               bucket_offset,
        int               k,
        int               max_k,
        MstBucketScratch& scratch,
        MstDeviceStorage& storage,
        cudaStream_t      stream)
    {
        int* row_offsets = storage.row_offsets + bucket_id * (max_k + 1);
        int* cols        = storage.cols + bucket_id * 2 * max_k;

        set_bucket_k_kernel<<<1, 1, 0, stream>>>(storage.bucket_k, bucket_id, k);

        if (k <= 1)
        {
            return;
        }

        if (k == 2)
        {
            init_mst_k2_kernel<<<1, 1, 0, stream>>>(row_offsets, cols, max_k);
            return;
        }

        init_parent_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
            scratch.parent, k);
        CUDA_CHECK(cudaMemsetAsync(scratch.mst_edge_count, 0, sizeof(int), stream));

        int h_components = 0;
        while (true)
        {
            CUDA_CHECK(cudaMemsetAsync(scratch.component_count, 0, sizeof(int), stream));
            count_components_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                scratch.parent, k, scratch.component_count);
            CUDA_CHECK(cudaMemcpyAsync(
                &h_components,
                scratch.component_count,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            if (h_components <= 1)
            {
                break;
            }

            reset_cheapest_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                scratch.cheapest_w, scratch.cheapest_u, scratch.cheapest_v, k);

            const int total_pairs = k * (k - 1) / 2;
            boruvka_find_cheapest_kernel<<<launch_blocks(total_pairs, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                device.device_bucket_nodes(),
                bucket_offset,
                device.device_x(),
                device.device_y(),
                scratch.parent,
                k,
                scratch.cheapest_w,
                scratch.cheapest_u,
                scratch.cheapest_v);

            boruvka_merge_kernel<<<launch_blocks(k, kThreadsPerBlock), kThreadsPerBlock, 0, stream>>>(
                scratch.parent,
                k,
                scratch.cheapest_w,
                scratch.cheapest_u,
                scratch.cheapest_v,
                scratch.mst_u,
                scratch.mst_v,
                scratch.mst_edge_count);
        }

        int edge_count = 0;
        CUDA_CHECK(cudaMemcpyAsync(
            &edge_count,
            scratch.mst_edge_count,
            sizeof(int),
            cudaMemcpyDeviceToHost,
            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        edge_count = std::min(edge_count, k - 1);
        if (edge_count <= 0)
        {
            return;
        }

        int* degrees = scratch.parent;
        build_csr_from_edges(
            row_offsets,
            cols,
            degrees,
            scratch.mst_u,
            scratch.mst_v,
            edge_count,
            k,
            stream);
    }

    void build_all_msts_on_streams(
        const DeviceCVRP&       device,
        const int*              d_bucket_offsets,
        const std::vector<int>& h_bucket_k,
        int                     max_k,
        MstBucketScratch*       per_bucket_scratch,
        MstDeviceStorage&       storage,
        cudaStream_t*           streams)
    {
        const int num_buckets = static_cast<int>(h_bucket_k.size());
        std::vector<int> h_offsets(2);

        std::vector<std::thread> workers;
        workers.reserve(num_buckets);

        for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id)
        {
            workers.emplace_back([&, bucket_id]()
            {
                CUDA_CHECK(cudaMemcpy(
                    h_offsets.data(),
                    d_bucket_offsets + bucket_id,
                    2 * sizeof(int),
                    cudaMemcpyDeviceToHost));

                const int bucket_offset = h_offsets[0];
                const int k             = h_bucket_k[bucket_id];

                construct_mst_boruvka_streamed(
                    device,
                    bucket_id,
                    bucket_offset,
                    k,
                    max_k,
                    per_bucket_scratch[bucket_id],
                    storage,
                    streams[bucket_id]);
                CUDA_CHECK(cudaStreamSynchronize(streams[bucket_id]));
            });
        }

        for (auto& worker : workers)
        {
            worker.join();
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
        MstBucketScratch scratch;
        scratch.parent          = d_parent;
        scratch.cheapest_w      = d_cheapest_w;
        scratch.cheapest_u      = d_cheapest_u;
        scratch.cheapest_v      = d_cheapest_v;
        scratch.mst_u           = d_mst_u;
        scratch.mst_v           = d_mst_v;
        scratch.mst_edge_count  = d_mst_edge_count;
        scratch.component_count = d_component_count;

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

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        MstDeviceStorage temp_storage;
        allocate_mst_device_storage(temp_storage, 1, max_bucket_size);

        construct_mst_boruvka_streamed(
            device,
            0,
            bucket_offset,
            k,
            max_bucket_size,
            scratch,
            temp_storage,
            stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (k == 2)
        {
            mst_adj[0].push_back(1);
            mst_adj[1].push_back(0);
            free_mst_device_storage(temp_storage);
            cudaStreamDestroy(stream);
            return;
        }

        int edge_count = 0;
        CUDA_CHECK(cudaMemcpy(
            &edge_count,
            d_mst_edge_count,
            sizeof(int),
            cudaMemcpyDeviceToHost));
        edge_count = std::min(edge_count, k - 1);

        std::vector<int> h_mst_u(edge_count);
        std::vector<int> h_mst_v(edge_count);
        CUDA_CHECK(cudaMemcpy(h_mst_u.data(), d_mst_u, edge_count * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mst_v.data(), d_mst_v, edge_count * sizeof(int), cudaMemcpyDeviceToHost));

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

        free_mst_device_storage(temp_storage);
        cudaStreamDestroy(stream);
    }
}
