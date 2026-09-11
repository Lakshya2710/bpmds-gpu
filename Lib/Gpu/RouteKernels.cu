#include "Gpu/RouteKernels.h"

#include <cfloat>
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

namespace Gpu
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        __device__ inline double dist_global(
            int           gu,
            int           gv,
            const double* x,
            const double* y)
        {
            const double dx = x[gu] - x[gv];
            const double dy = y[gu] - y[gv];
            return sqrt(dx * dx + dy * dy);
        }

        __device__ unsigned int lcg_next(unsigned int* state)
        {
            *state = (*state * 1103515245u + 12345u);
            return *state;
        }

        __device__ void shuffle_neighbors(int* arr, int n, unsigned int* rng)
        {
            for (int i = n - 1; i > 0; --i)
            {
                const int j = static_cast<int>(lcg_next(rng) % static_cast<unsigned int>(i + 1));
                const int tmp = arr[i];
                arr[i]        = arr[j];
                arr[j]        = tmp;
            }
        }

        __device__ int& stack_top(int* scratch)
        {
            return scratch[0];
        }

        __device__ int& neigh_pool_used(int* scratch)
        {
            return scratch[1];
        }

        __device__ int& curr_route_len(int* scratch)
        {
            return scratch[2];
        }

        __device__ int* visited(int* scratch, int k)
        {
            return scratch + 4;
        }

        __device__ int* stack_v(int* scratch, int k)
        {
            return scratch + 4 + k;
        }

        __device__ int* stack_pos(int* scratch, int k)
        {
            return scratch + 4 + 2 * k;
        }

        __device__ int* stack_pool_start(int* scratch, int k)
        {
            return scratch + 4 + 3 * k;
        }

        __device__ int* curr_route(int* scratch, int k)
        {
            return scratch + 4 + 4 * k;
        }

        __device__ int* neigh_pool(int* scratch, int k)
        {
            return scratch + 4 + 5 * k;
        }

        __device__ void push_stack_frame(
            int* scratch,
            int  k,
            int  v_index,
            int  deg,
            int  pool_start)
        {
            const int top         = stack_top(scratch)++;
            stack_v(scratch, k)[top]           = v_index;
            stack_pos(scratch, k)[top]         = deg - 1;
            stack_pool_start(scratch, k)[top]  = pool_start;
        }

        __device__ void get_routes_one_trial(
            const int*    bucket_nodes,
            int           bucket_offset,
            int           k,
            int           max_k,
            const int*    row_offsets,
            const int*    cols,
            const double* x,
            const double* y,
            const double* demand,
            double        capacity,
            int           bucket_id,
            int           trial_id,
            int*          scratch,
            double*       out_cost,
            int*          out_num_routes,
            int*          out_route_offsets,
            int*          out_route_nodes)
        {
            if (k <= 1)
            {
                *out_cost       = 0.0;
                *out_num_routes = 0;
                out_route_offsets[0] = 0;
                return;
            }

            unsigned int rng = static_cast<unsigned int>(bucket_id * 1000003 + trial_id * 9176u + 12345u);

            for (int i = 0; i < 4; ++i)
            {
                scratch[i] = 0;
            }
            for (int i = 0; i < k; ++i)
            {
                visited(scratch, k)[i] = 0;
            }

            const int depot_global = bucket_nodes[bucket_offset];
            double total_cost      = 0.0;
            double residue         = capacity;
            int    prev_global     = depot_global;

            int out_node_write  = 0;
            int out_route_count = 0;
            out_route_offsets[0] = 0;

            visited(scratch, k)[0] = 1;

            const int deg0 = row_offsets[1] - row_offsets[0];
            const int pool = neigh_pool_used(scratch);
            for (int i = 0; i < deg0; ++i)
            {
                neigh_pool(scratch, k)[pool + i] = cols[row_offsets[0] + i];
            }
            shuffle_neighbors(neigh_pool(scratch, k) + pool, deg0, &rng);
            neigh_pool_used(scratch) = pool + deg0;
            push_stack_frame(scratch, k, 0, deg0, pool);

            while (stack_top(scratch) > 0)
            {
                const int top        = stack_top(scratch) - 1;
                int       index      = stack_pos(scratch, k)[top];
                const int pool_start = stack_pool_start(scratch, k)[top];
                int       push_new   = 0;
                int       new_v      = -1;
                int       new_deg    = 0;
                int       new_pool   = 0;

                while (index >= 0)
                {
                    const int v_index = neigh_pool(scratch, k)[pool_start + index];
                    --index;

                    if (visited(scratch, k)[v_index])
                    {
                        continue;
                    }

                    visited(scratch, k)[v_index] = 1;
                    const int    v_global  = bucket_nodes[bucket_offset + v_index];
                    const double v_demand  = demand[v_global];

                    if (residue < v_demand)
                    {
                        for (int i = 0; i < curr_route_len(scratch); ++i)
                        {
                            out_route_nodes[out_node_write++] = curr_route(scratch, k)[i];
                        }
                        curr_route_len(scratch) = 0;
                        total_cost += dist_global(prev_global, depot_global, x, y);
                        ++out_route_count;
                        out_route_offsets[out_route_count] = out_node_write;
                        residue     = capacity;
                        prev_global = depot_global;
                    }

                    curr_route(scratch, k)[curr_route_len(scratch)++] = v_global;
                    total_cost += dist_global(prev_global, v_global, x, y);
                    residue -= v_demand;
                    prev_global = v_global;

                    const int v_deg = row_offsets[v_index + 1] - row_offsets[v_index];
                    new_pool        = neigh_pool_used(scratch);
                    for (int i = 0; i < v_deg; ++i)
                    {
                        neigh_pool(scratch, k)[new_pool + i] = cols[row_offsets[v_index] + i];
                    }
                    shuffle_neighbors(neigh_pool(scratch, k) + new_pool, v_deg, &rng);
                    neigh_pool_used(scratch) = new_pool + v_deg;

                    new_v    = v_index;
                    new_deg  = v_deg;
                    push_new = 1;
                    break;
                }

                if (index < 0)
                {
                    --stack_top(scratch);
                }
                else
                {
                    stack_pos(scratch, k)[top] = index;
                }

                if (push_new)
                {
                    push_stack_frame(scratch, k, new_v, new_deg, new_pool);
                }
            }

            if (curr_route_len(scratch) > 0)
            {
                for (int i = 0; i < curr_route_len(scratch); ++i)
                {
                    out_route_nodes[out_node_write++] = curr_route(scratch, k)[i];
                }
                curr_route_len(scratch) = 0;
                total_cost += dist_global(prev_global, depot_global, x, y);
                ++out_route_count;
                out_route_offsets[out_route_count] = out_node_write;
            }

            *out_cost       = total_cost;
            *out_num_routes = out_route_count;
        }

        __global__ void get_routes_kernel(
            const int*    bucket_nodes,
            int           bucket_offset,
            int           k,
            int           max_k,
            const int*    row_offsets,
            const int*    cols,
            const double* x,
            const double* y,
            const double* demand,
            double        capacity,
            int           bucket_id,
            int           rho,
            int           scratch_stride,
            int*          trial_scratch,
            double*       trial_costs,
            int*          trial_num_routes,
            int*          route_offsets,
            int*          route_nodes)
        {
            const int trial = blockIdx.x * blockDim.x + threadIdx.x;
            if (trial >= rho)
            {
                return;
            }

            const int flat = bucket_id * rho + trial;
            int* scratch = trial_scratch + flat * scratch_stride;

            get_routes_one_trial(
                bucket_nodes,
                bucket_offset,
                k,
                max_k,
                row_offsets,
                cols,
                x,
                y,
                demand,
                capacity,
                bucket_id,
                trial,
                scratch,
                &trial_costs[flat],
                &trial_num_routes[flat],
                route_offsets + flat * (max_k + 1),
                route_nodes + flat * max_k);
        }

        void launch_bucket_route_trials(
            const DeviceCVRP&       device,
            int                     bucket_id,
            int                     bucket_offset,
            int                     k,
            int                     max_k,
            const MstDeviceStorage& mst_storage,
            int                     rho,
            double                  capacity,
            RouteTrialStorage&      storage,
            cudaStream_t            stream)
        {
            if (k <= 1)
            {
                return;
            }

            const int* row_offsets = mst_storage.row_offsets + bucket_id * (max_k + 1);
            const int* cols        = mst_storage.cols + bucket_id * 2 * max_k;

            const int blocks = (rho + kThreadsPerBlock - 1) / kThreadsPerBlock;
            get_routes_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
                device.device_bucket_nodes(),
                bucket_offset,
                k,
                max_k,
                row_offsets,
                cols,
                device.device_x(),
                device.device_y(),
                device.device_demand(),
                capacity,
                bucket_id,
                rho,
                route_trial_scratch_stride(max_k),
                storage.trial_scratch,
                storage.trial_costs,
                storage.trial_num_routes,
                storage.route_offsets,
                storage.route_nodes);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    int route_trial_scratch_stride(int max_k)
    {
        return 4 + 5 * max_k + max_k * max_k;
    }

    void allocate_route_trial_storage(RouteTrialStorage& storage, int num_buckets, int rho, int max_k)
    {
        const int flat_trials = num_buckets * rho;
        const int stride      = route_trial_scratch_stride(max_k);

        CUDA_CHECK(cudaMalloc(&storage.trial_costs, flat_trials * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&storage.trial_num_routes, flat_trials * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.route_offsets, flat_trials * (max_k + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.route_nodes, flat_trials * max_k * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.trial_scratch, flat_trials * stride * sizeof(int)));
    }

    void free_route_trial_storage(RouteTrialStorage& storage)
    {
        cudaFree(storage.trial_costs);
        cudaFree(storage.trial_num_routes);
        cudaFree(storage.route_offsets);
        cudaFree(storage.route_nodes);
        cudaFree(storage.trial_scratch);
        storage = {};
    }

    void run_all_route_trials_on_streams(
        const DeviceCVRP&       device,
        const int*              d_bucket_offsets,
        const std::vector<int>& h_bucket_k,
        const MstDeviceStorage& mst_storage,
        int                     rho,
        int                     max_k,
        double                  capacity,
        RouteTrialStorage&      storage,
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

                launch_bucket_route_trials(
                    device,
                    bucket_id,
                    bucket_offset,
                    k,
                    max_k,
                    mst_storage,
                    rho,
                    capacity,
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

    void fetch_best_routes_from_device(
        int                               bucket_id,
        int                               rho,
        int                               max_k,
        const RouteTrialStorage&          storage,
        std::vector<std::vector<node_t>>& routes,
        double&                           cost)
    {
        routes.clear();
        cost = DBL_MAX;

        std::vector<double> h_costs(rho);
        CUDA_CHECK(cudaMemcpy(
            h_costs.data(),
            storage.trial_costs + bucket_id * rho,
            rho * sizeof(double),
            cudaMemcpyDeviceToHost));

        int best_trial = 0;
        for (int t = 1; t < rho; ++t)
        {
            if (h_costs[t] < h_costs[best_trial])
            {
                best_trial = t;
            }
        }

        cost = h_costs[best_trial];

        const int flat = bucket_id * rho + best_trial;

        int num_routes = 0;
        CUDA_CHECK(cudaMemcpy(
            &num_routes,
            storage.trial_num_routes + flat,
            sizeof(int),
            cudaMemcpyDeviceToHost));

        if (num_routes <= 0)
        {
            cost = 0.0;
            return;
        }

        std::vector<int> h_offsets(num_routes + 1);
        std::vector<int> h_nodes(max_k);

        CUDA_CHECK(cudaMemcpy(
            h_offsets.data(),
            storage.route_offsets + flat * (max_k + 1),
            (num_routes + 1) * sizeof(int),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_nodes.data(),
            storage.route_nodes + flat * max_k,
            h_offsets[num_routes] * sizeof(int),
            cudaMemcpyDeviceToHost));

        routes.resize(num_routes);
        for (int r = 0; r < num_routes; ++r)
        {
            const int start = h_offsets[r];
            const int end   = h_offsets[r + 1];
            routes[r].assign(h_nodes.begin() + start, h_nodes.begin() + end);
        }
    }
}
