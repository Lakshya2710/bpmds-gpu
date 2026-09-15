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
            const float dx = __double2float_rn(x[gu] - x[gv]);
            const float dy = __double2float_rn(y[gu] - y[gv]);
            return static_cast<double>(sqrtf(dx * dx + dy * dy));
        }

        __device__ unsigned int lcg_next(unsigned int* state)
        {
            *state = (*state * 1103515245u + 12345u);
            return *state;
        }

        // --- Streamlined Memory Accessors ---
        // Down from 7 arrays to just 3!
        __device__ int* visited(int* scratch, int k) { return scratch; }
        __device__ int* stack_v(int* scratch, int k) { return scratch + k; }
        __device__ int* curr_route(int* scratch, int k) { return scratch + 2 * k; }

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
                *out_cost = 0.0;
                if (out_num_routes) *out_num_routes = 0;
                if (out_route_offsets) out_route_offsets[0] = 0;
                return;
            }

            unsigned int rng = static_cast<unsigned int>(bucket_id * 1000003 + trial_id * 9176u + 12345u);

            for (int i = 0; i < k; ++i) visited(scratch, k)[i] = 0;

            const int depot_global = bucket_nodes[bucket_offset];
            double total_cost      = 0.0;
            double residue         = capacity;
            int    prev_global     = depot_global;

            int out_node_write  = 0;
            int out_route_count = 0;
            if (out_route_offsets) out_route_offsets[0] = 0;

            // MASSIVE SPEEDUP: These live in blazing fast hardware registers now!
            int top = 0;
            int route_len = 0;

            // Push depot to start the tree walk
            stack_v(scratch, k)[top++] = 0;

            while (top > 0)
            {
                int u_idx = stack_v(scratch, k)[--top];

                if (visited(scratch, k)[u_idx]) continue;
                visited(scratch, k)[u_idx] = 1;

                // Process the node (skip payload math for the depot itself)
                if (u_idx != 0)
                {
                    int u_global = bucket_nodes[bucket_offset + u_idx];
                    double u_demand = demand[u_global];

                    if (residue < u_demand)
                    {
                        if (out_route_nodes) {
                            for (int i = 0; i < route_len; ++i)
                                out_route_nodes[out_node_write++] = curr_route(scratch, k)[i];
                            out_route_offsets[out_route_count + 1] = out_node_write;
                        }
                        route_len = 0;
                        total_cost += dist_global(prev_global, depot_global, x, y);
                        ++out_route_count;
                        residue     = capacity;
                        prev_global = depot_global;
                    }

                    curr_route(scratch, k)[route_len++] = u_global;
                    total_cost += dist_global(prev_global, u_global, x, y);
                    residue -= u_demand;
                    prev_global = u_global;
                }

                // Push all unvisited neighbors directly to the stack
                int start_top = top;
                int row_start = row_offsets[u_idx];
                int deg = row_offsets[u_idx + 1] - row_start;

                for (int i = 0; i < deg; ++i)
                {
                    int v_idx = cols[row_start + i];
                    if (!visited(scratch, k)[v_idx])
                    {
                        stack_v(scratch, k)[top++] = v_idx;
                    }
                }

                // In-place shuffle of the newly pushed neighbors
                int count = top - start_top;
                if (count > 1)
                {
                    for (int i = count - 1; i > 0; --i)
                    {
                        int j = lcg_next(&rng) % (i + 1);
                        int tmp = stack_v(scratch, k)[start_top + i];
                        stack_v(scratch, k)[start_top + i] = stack_v(scratch, k)[start_top + j];
                        stack_v(scratch, k)[start_top + j] = tmp;
                    }
                }
            }

            if (route_len > 0)
            {
                if (out_route_nodes) {
                    for (int i = 0; i < route_len; ++i)
                        out_route_nodes[out_node_write++] = curr_route(scratch, k)[i];
                    out_route_offsets[out_route_count + 1] = out_node_write;
                }
                route_len = 0;
                total_cost += dist_global(prev_global, depot_global, x, y);
                ++out_route_count;
            }

            *out_cost = total_cost;
            if (out_num_routes) *out_num_routes = out_route_count;
        }

        // =========================================================================
        // PHASE 4: BATCHED 2-OPT AND TSP APPROX KERNELS 
        // =========================================================================

        __device__ double calc_tour_cost(const int* tour, int sz, int depot, const double* x, const double* y) {
            double cost = dist_global(depot, tour[0], x, y);
            for (int i = 1; i < sz; ++i) cost += dist_global(tour[i - 1], tour[i], x, y);
            cost += dist_global(tour[sz - 1], depot, x, y);
            return cost;
        }

        __device__ void gpu_tsp_approx(const int* original, int* tour, int sz, int depot, const double* x, const double* y) {
            for (int i = 0; i < sz; ++i) tour[i] = original[i];

            for (int i = 0; i < sz - 1; ++i) {
                double best_dist = 1e30;
                int best_pt = i;
                double prev_x = (i == 0) ? x[depot] : x[tour[i - 1]];
                double prev_y = (i == 0) ? y[depot] : y[tour[i - 1]];

                for (int j = i; j < sz; ++j) {
                    double dx = x[tour[j]] - prev_x;
                    double dy = y[tour[j]] - prev_y;
                    double d2 = dx * dx + dy * dy; 
                    if (d2 < best_dist) {
                        best_dist = d2;
                        best_pt = j;
                    }
                }
                int tmp = tour[i];
                tour[i] = tour[best_pt];
                tour[best_pt] = tmp;
            }
        }

        __device__ void gpu_tsp_2opt(int* tour, int sz, int depot, const double* x, const double* y, int* aux) {
            int improve = 0;
            while (improve < 2) {
                double best_dist = calc_tour_cost(tour, sz, depot, x, y);
                bool found = false;

                for (int i = 0; i < sz - 1; ++i) {
                    for (int k = i + 1; k < sz; ++k) {
                        for (int c = 0; c < i; ++c) aux[c] = tour[c];
                        int dec = 0;
                        for (int c = i; c <= k; ++c) aux[c] = tour[k - (dec++)];
                        for (int c = k + 1; c < sz; ++c) aux[c] = tour[c];

                        double new_dist = calc_tour_cost(aux, sz, depot, x, y);
                        if (new_dist < best_dist) {
                            improve = 0;
                            for (int j = 0; j < sz; ++j) tour[j] = aux[j];
                            best_dist = new_dist;
                            found = true;
                        }
                    }
                }
                if (!found) improve++;
            }
        }

        __global__ void batched_postprocess_kernel(
            const int* bucket_nodes, int bucket_offset,
            int* d_num_routes, int* route_offsets, int* route_nodes,
            int* flat_scratch, const double* x, const double* y)
        {
            int num_routes = *d_num_routes;
            const int r = blockIdx.x * blockDim.x + threadIdx.x;
            if (r >= num_routes) return;

            const int start = route_offsets[r];
            const int end = route_offsets[r + 1];
            const int sz = end - start;
            if (sz <= 2) return;

            const int depot = bucket_nodes[bucket_offset];
            
            int* orig  = route_nodes + start;
            int* tour1 = flat_scratch + (start * 3);
            int* tour2 = flat_scratch + (start * 3) + sz;
            int* aux   = flat_scratch + (start * 3) + 2 * sz;

            for(int i=0; i<sz; ++i) tour2[i] = orig[i];
            gpu_tsp_2opt(tour2, sz, depot, x, y, aux);
            double cost2 = calc_tour_cost(tour2, sz, depot, x, y);

            gpu_tsp_approx(orig, tour1, sz, depot, x, y);
            gpu_tsp_2opt(tour1, sz, depot, x, y, aux);
            double cost1 = calc_tour_cost(tour1, sz, depot, x, y);

            double orig_cost = calc_tour_cost(orig, sz, depot, x, y);

            if (cost1 < cost2 && cost1 < orig_cost) {
                for(int i=0; i<sz; ++i) orig[i] = tour1[i];
            } else if (cost2 < cost1 && cost2 < orig_cost) {
                for(int i=0; i<sz; ++i) orig[i] = tour2[i];
            }
        }

        // =========================================================================

        __global__ void evaluate_routes_kernel(
            const int* bucket_nodes, int bucket_offset, int k, int max_k,
            const int* row_offsets, const int* cols, const double* x, const double* y,
            const double* demand, double capacity, int bucket_id, int rho,
            int scratch_stride, int* trial_scratch, double* trial_costs)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            int* scratch = trial_scratch + tid * scratch_stride; 

            for (int trial = tid; trial < rho; trial += gridDim.x * blockDim.x)
            {
                double cost;
                get_routes_one_trial(
                    bucket_nodes, bucket_offset, k, max_k, row_offsets, cols, x, y, demand, capacity, 
                    bucket_id, trial, scratch, &cost, nullptr, nullptr, nullptr); 
                
                trial_costs[bucket_id * rho + trial] = cost;
            }
        }

        __global__ void materialize_winner_kernel(
            const int* bucket_nodes, int bucket_offset, int k, int max_k,
            const int* row_offsets, const int* cols, const double* x, const double* y,
            const double* demand, double capacity, int bucket_id, int rho,
            int scratch_stride, int* trial_scratch, double* trial_costs,
            int* out_num_routes, int* out_route_offsets, int* out_route_nodes)
        {
            if (threadIdx.x != 0 || blockIdx.x != 0) return;

            int best_trial = 0;
            double best_cost = trial_costs[bucket_id * rho + 0];
            for (int i = 1; i < rho; ++i)
            {
                if (trial_costs[bucket_id * rho + i] < best_cost) {
                    best_cost = trial_costs[bucket_id * rho + i];
                    best_trial = i;
                }
            }

            int* scratch = trial_scratch + 0 * scratch_stride;
            double dummy_cost;

            get_routes_one_trial(
                bucket_nodes, bucket_offset, k, max_k, row_offsets, cols, x, y, demand, capacity, 
                bucket_id, best_trial, scratch, &dummy_cost, out_num_routes, out_route_offsets, out_route_nodes);
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
            if (k <= 1) return;

            const int* row_offsets = mst_storage.row_offsets + bucket_id * (max_k + 1);
            const int* cols        = mst_storage.cols + bucket_id * 2 * max_k;

            int active_blocks = 4;
            int active_threads = active_blocks * kThreadsPerBlock;

            int stride = route_trial_scratch_stride(k);
            size_t scratch_bytes = (size_t)active_threads * stride * sizeof(int);
            int* d_scratch;
            CUDA_CHECK(cudaMallocAsync(&d_scratch, scratch_bytes, stream));

            evaluate_routes_kernel<<<active_blocks, kThreadsPerBlock, 0, stream>>>(
                device.device_bucket_nodes(), bucket_offset, k, max_k, row_offsets, cols,
                device.device_x(), device.device_y(), device.device_demand(), capacity,
                bucket_id, rho, stride, d_scratch, storage.trial_costs);

            materialize_winner_kernel<<<1, 1, 0, stream>>>(
                device.device_bucket_nodes(), bucket_offset, k, max_k, row_offsets, cols,
                device.device_x(), device.device_y(), device.device_demand(), capacity,
                bucket_id, rho, stride, d_scratch, storage.trial_costs,
                storage.trial_num_routes + bucket_id, 
                storage.route_offsets + bucket_id * (max_k + 1), 
                storage.route_nodes + bucket_id * max_k);

            int blocks = (k + kThreadsPerBlock - 1) / kThreadsPerBlock;
            batched_postprocess_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
                device.device_bucket_nodes(), bucket_offset,
                storage.trial_num_routes + bucket_id,
                storage.route_offsets + bucket_id * (max_k + 1),
                storage.route_nodes + bucket_id * max_k,
                d_scratch, device.device_x(), device.device_y()
            );

            CUDA_CHECK(cudaFreeAsync(d_scratch, stream));
        }
    }

    int route_trial_scratch_stride(int k)
    {
        return 3 * k; 
    }

    void allocate_route_trial_storage(RouteTrialStorage& storage, int num_buckets, int rho, int max_k)
    {
        CUDA_CHECK(cudaMalloc(&storage.trial_costs, num_buckets * rho * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&storage.trial_num_routes, num_buckets * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.route_offsets, num_buckets * (max_k + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&storage.route_nodes, num_buckets * max_k * sizeof(int)));
    }

    void free_route_trial_storage(RouteTrialStorage& storage)
    {
        cudaFree(storage.trial_costs);
        cudaFree(storage.trial_num_routes);
        cudaFree(storage.route_offsets);
        cudaFree(storage.route_nodes);
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
        std::vector<int> h_offsets(num_buckets + 1);
        CUDA_CHECK(cudaMemcpy(h_offsets.data(), d_bucket_offsets, (num_buckets + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id)
        {
            const int bucket_offset = h_offsets[bucket_id];
            const int k             = h_bucket_k[bucket_id];
            launch_bucket_route_trials(
                device, bucket_id, bucket_offset, k, max_k, mst_storage, rho, capacity, storage, streams[bucket_id]);
        }
        for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id)
        {
            CUDA_CHECK(cudaStreamSynchronize(streams[bucket_id]));
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
        cost = 0.0;

        int num_routes = 0;
        CUDA_CHECK(cudaMemcpy(&num_routes, storage.trial_num_routes + bucket_id, sizeof(int), cudaMemcpyDeviceToHost));

        if (num_routes <= 0) return;

        std::vector<double> h_costs(rho);
        CUDA_CHECK(cudaMemcpy(h_costs.data(), storage.trial_costs + bucket_id * rho, rho * sizeof(double), cudaMemcpyDeviceToHost));
        
        cost = h_costs[0];
        for (int t = 1; t < rho; ++t) {
            if (h_costs[t] < cost) cost = h_costs[t];
        }

        std::vector<int> h_offsets(num_routes + 1);
        std::vector<int> h_nodes(max_k);

        CUDA_CHECK(cudaMemcpy(h_offsets.data(), storage.route_offsets + bucket_id * (max_k + 1), (num_routes + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nodes.data(), storage.route_nodes + bucket_id * max_k, h_offsets[num_routes] * sizeof(int), cudaMemcpyDeviceToHost));

        routes.resize(num_routes);
        for (int r = 0; r < num_routes; ++r)
        {
            const int start = h_offsets[r];
            const int end   = h_offsets[r + 1];
            routes[r].assign(h_nodes.begin() + start, h_nodes.begin() + end);
        }
    }
}

