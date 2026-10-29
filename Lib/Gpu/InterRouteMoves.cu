#include "Gpu/InterRouteMoves.h"
#include "Gpu/CudaCommon.h"

#include <algorithm>
#include <chrono>
#include <vector>

namespace Gpu
{
    namespace
    {
        constexpr int kNeighbours = 20;
        constexpr int kMaxNeighbours = 32;
        constexpr double kImproveEps = 1.0e-6;

        __device__ double euc(
            double ax, double ay, double bx, double by)
        {
            const double dx = ax - bx;
            const double dy = ay - by;
            return sqrt(dx * dx + dy * dy);
        }

        __device__ void point_of(
            int local,
            const double* __restrict__ x,
            const double* __restrict__ y,
            double depot_x,
            double depot_y,
            double& ox,
            double& oy)
        {
            if (local < 0)
            {
                ox = depot_x;
                oy = depot_y;
                return;
            }
            ox = x[local];
            oy = y[local];
        }

        // Column-major neighbours: nbr[t * n + query] so a warp reading the
        // same t touches consecutive addresses.
        // Uses bounding-box tile pruning and FP32 packed coordinates.
        constexpr int kTileSize = 256;

        __device__ __forceinline__ void evaluate_tile_candidates_static(
            const float2* __restrict__ coords,
            int start, int end,
            int query, float qx, float qy,
            int k,
            float* best_dist,
            int* best_id)
        {
            #pragma unroll 4
            for (int s = start; s < end; ++s)
            {
                if (s == query) continue;

                const float2 cs = __ldg(&coords[s]);
                const float dx = cs.x - qx;
                const float dy = cs.y - qy;
                const float dist2 = dx * dx + dy * dy;

                if (dist2 < best_dist[k - 1])
                {
                    float current_dist = dist2;
                    int   current_id   = s;

                    #pragma unroll
                    for (int i = 0; i < kMaxNeighbours; ++i)
                    {
                        if (i >= k) break;

                        if (current_dist < best_dist[i])
                        {
                            const float tmp_dist = best_dist[i];
                            const int   tmp_id   = best_id[i];

                            best_dist[i] = current_dist;
                            best_id[i]   = current_id;

                            current_dist = tmp_dist;
                            current_id   = tmp_id;
                        }
                    }
                }
            }
        }

        __global__ void static_pool_neighbour_kernel(
            const float2* __restrict__ coords,
            const float4* __restrict__ tile_boxes,
            int num_tiles,
            int n,
            int k,
            int* __restrict__ nbr)
        {
            const int query = blockIdx.x * blockDim.x + threadIdx.x;
            if (query >= n)
            {
                return;
            }

            float best_dist[kMaxNeighbours];
            int   best_id[kMaxNeighbours];

            #pragma unroll
            for (int i = 0; i < kMaxNeighbours; ++i)
            {
                best_dist[i] = 1e30f;
                best_id[i]   = -1;
            }

            const float2 q = __ldg(&coords[query]);
            const float qx = q.x;
            const float qy = q.y;

            // 1. Evaluate own tile first
            const int own_tile = blockIdx.x;
            const int own_start = own_tile * kTileSize;
            const int own_end   = min(n, own_start + kTileSize);
            evaluate_tile_candidates_static(
                coords, own_start, own_end,
                query, qx, qy, k, best_dist, best_id);

            // 2. Loop through all other tiles with bounding-box pruning
            for (int t = 0; t < num_tiles; ++t)
            {
                if (t == own_tile) continue;

                const float4 box = __ldg(&tile_boxes[t]);
                const float dx_box = fmaxf(0.0f, fmaxf(box.x - qx, qx - box.y));
                const float dy_box = fmaxf(0.0f, fmaxf(box.z - qy, qy - box.w));
                const float box_dist2 = dx_box * dx_box + dy_box * dy_box;

                if (box_dist2 >= best_dist[k - 1])
                {
                    continue;
                }

                const int t_start = t * kTileSize;
                const int t_end   = min(n, t_start + kTileSize);
                evaluate_tile_candidates_static(
                    coords, t_start, t_end,
                    query, qx, qy, k, best_dist, best_id);
            }

            for (int t = 0; t < k; ++t)
            {
                nbr[t * n + query] = best_id[t];
            }
        }

        __global__ void evaluate_moves_kernel(
            const double* __restrict__ x,
            const double* __restrict__ y,
            const double* __restrict__ demand,
            const double* __restrict__ cum_load,
            const int* __restrict__ pred,
            const int* __restrict__ succ,
            const int* __restrict__ route_of,
            const double* __restrict__ route_load,
            const int* __restrict__ nbr,
            int n,
            int k,
            double depot_x,
            double depot_y,
            double capacity,
            double* __restrict__ best_delta,
            int* __restrict__ best_type,
            int* __restrict__ best_arg,
            int* __restrict__ best_route_v)
        {
            const int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n)
            {
                return;
            }

            const int ru = route_of[i];
            const double load_u = route_load[ru];
            const double dem_i = demand[i];
            const int prev = pred[i];
            const int next = succ[i];

            double px, py, nx, ny, ix, iy;
            point_of(prev, x, y, depot_x, depot_y, px, py);
            point_of(next, x, y, depot_x, depot_y, nx, ny);
            point_of(i, x, y, depot_x, depot_y, ix, iy);

            const double remove1 = euc(px, py, nx, ny) - euc(px, py, ix, iy) - euc(ix, iy, nx, ny);

            // Precompute Move 2 removal terms if node i has a successor
            double remove2 = 0.0;
            double dem_pair = 0.0;
            if (next >= 0)
            {
                const int next2 = succ[next];
                double n2x, n2y;
                point_of(next2, x, y, depot_x, depot_y, n2x, n2y);
                dem_pair = dem_i + demand[next];
                remove2 = euc(px, py, n2x, n2y) - euc(px, py, ix, iy) - euc(nx, ny, n2x, n2y);
            }

            double best = 0.0;
            int type = 0;
            int arg = -1;
            int route_v = -1;

            for (int t = 0; t < k; ++t)
            {
                const int j = nbr[t * n + i];
                if (j < 0)
                {
                    continue;
                }
                const int rv = route_of[j];
                if (rv == ru)
                {
                    continue;
                }

                const int after = succ[j];
                double jx, jy, ax, ay;
                point_of(j, x, y, depot_x, depot_y, jx, jy);
                point_of(after, x, y, depot_x, depot_y, ax, ay);

                // Move 1: single relocate
                const double insert = euc(jx, jy, ix, iy) + euc(ix, iy, ax, ay) - euc(jx, jy, ax, ay);
                const double relocate = remove1 + insert;
                if (route_load[rv] + dem_i <= capacity && relocate < best - kImproveEps)
                {
                    best = relocate;
                    type = 1;
                    arg = j;
                    route_v = rv;
                }

                // Move 2: relocate pair (i, next)
                if (next >= 0 && route_load[rv] + dem_pair <= capacity)
                {
                    const double insert2 = euc(jx, jy, ix, iy) + euc(nx, ny, ax, ay) - euc(jx, jy, ax, ay);
                    const double relocate2 = remove2 + insert2;
                    if (relocate2 < best - kImproveEps)
                    {
                        best = relocate2;
                        type = 2;
                        arg = j;
                        route_v = rv;
                    }
                }

                const double link = euc(ix, iy, jx, jy) + euc(nx, ny, ax, ay)
                                  - euc(ix, iy, nx, ny) - euc(jx, jy, ax, ay);
                const double new_a = cum_load[i] + cum_load[j];
                const double new_b = load_u + route_load[rv] - new_a;
                if (new_a <= capacity && new_b <= capacity && link < best - kImproveEps)
                {
                    best = link;
                    type = 8;
                    arg = j;
                    route_v = rv;
                }

                const double link9 = euc(ix, iy, ax, ay) + euc(jx, jy, nx, ny)
                                   - euc(ix, iy, nx, ny) - euc(jx, jy, ax, ay);
                const double new9_a = cum_load[i] + route_load[rv] - cum_load[j];
                const double new9_b = cum_load[j] + load_u - cum_load[i];
                if (new9_a <= capacity && new9_b <= capacity && link9 < best - kImproveEps)
                {
                    best = link9;
                    type = 9;
                    arg = j;
                    route_v = rv;
                }

                if (pred[j] < 0)
                {
                    double dx, dy;
                    point_of(-1, x, y, depot_x, depot_y, dx, dy);
                    const double insert_front = euc(dx, dy, ix, iy) + euc(ix, iy, jx, jy) - euc(dx, dy, jx, jy);
                    const double relocate_front = remove1 + insert_front;
                    if (route_load[rv] + dem_i <= capacity && relocate_front < best - kImproveEps)
                    {
                        best = relocate_front;
                        type = 1;
                        arg = -1;
                        route_v = rv;
                    }

                    // Move 2 front insertion
                    if (next >= 0 && route_load[rv] + dem_pair <= capacity)
                    {
                        const double insert2_front = euc(dx, dy, ix, iy) + euc(nx, ny, jx, jy) - euc(dx, dy, jx, jy);
                        const double relocate2_front = remove2 + insert2_front;
                        if (relocate2_front < best - kImproveEps)
                        {
                            best = relocate2_front;
                            type = 2;
                            arg = -1;
                            route_v = rv;
                        }
                    }

                    const double link_depot = euc(ix, iy, dx, dy) + euc(nx, ny, jx, jy)
                                            - euc(ix, iy, nx, ny) - euc(dx, dy, jx, jy);
                    const double depot_a = cum_load[i];
                    const double depot_b = load_u + route_load[rv] - depot_a;
                    if (depot_a <= capacity && depot_b <= capacity
                        && link_depot < best - kImproveEps)
                    {
                        best = link_depot;
                        type = 8;
                        arg = -1;
                        route_v = rv;
                    }

                    const double link_depot9 = euc(ix, iy, jx, jy) + euc(dx, dy, nx, ny)
                                             - euc(ix, iy, nx, ny) - euc(dx, dy, jx, jy);
                    const double depot9_a = cum_load[i] + route_load[rv];
                    const double depot9_b = load_u - cum_load[i];
                    if (depot9_a <= capacity && depot9_b <= capacity
                        && link_depot9 < best - kImproveEps)
                    {
                        best = link_depot9;
                        type = 9;
                        arg = -1;
                        route_v = rv;
                    }
                }
            }

            best_delta[i] = best;
            best_type[i] = type;
            best_arg[i] = arg;
            best_route_v[i] = route_v;
        }

        struct Candidate
        {
            double delta;
            int u;
            int type;
            int arg;
            int route_u;
            int route_v;
        };

        void apply_move1(
            std::vector<int>& route_u,
            std::vector<int>& route_v,
            int pos_u,
            int pos_v)
        {
            const int customer = route_u[pos_u];
            route_u.erase(route_u.begin() + pos_u);
            if (pos_v < 0)
            {
                route_v.insert(route_v.begin(), customer);
            }
            else
            {
                route_v.insert(route_v.begin() + pos_v + 1, customer);
            }
        }

        void apply_move2(
            std::vector<int>& route_u,
            std::vector<int>& route_v,
            int pos_u,
            int pos_v)
        {
            if (pos_u < 0 || pos_u + 1 >= static_cast<int>(route_u.size()))
            {
                return;
            }
            const int cust1 = route_u[pos_u];
            const int cust2 = route_u[pos_u + 1];
            route_u.erase(route_u.begin() + pos_u, route_u.begin() + pos_u + 2);
            if (pos_v < 0)
            {
                route_v.insert(route_v.begin(), {cust1, cust2});
            }
            else
            {
                route_v.insert(route_v.begin() + pos_v + 1, {cust1, cust2});
            }
        }

        void apply_move8(
            const std::vector<int>& route_u,
            const std::vector<int>& route_v,
            int pos_u,
            int pos_v,
            std::vector<int>& out_a,
            std::vector<int>& out_b)
        {
            std::vector<int> prefix_u(route_u.begin(), route_u.begin() + pos_u + 1);
            std::vector<int> suffix_u(route_u.begin() + pos_u + 1, route_u.end());
            std::vector<int> prefix_v;
            std::vector<int> suffix_v;
            if (pos_v < 0)
            {
                suffix_v = route_v;
            }
            else
            {
                prefix_v.assign(route_v.begin(), route_v.begin() + pos_v + 1);
                suffix_v.assign(route_v.begin() + pos_v + 1, route_v.end());
            }
            std::reverse(prefix_v.begin(), prefix_v.end());
            std::reverse(suffix_u.begin(), suffix_u.end());
            out_a = std::move(prefix_u);
            out_a.insert(out_a.end(), prefix_v.begin(), prefix_v.end());
            out_b = std::move(suffix_u);
            out_b.insert(out_b.end(), suffix_v.begin(), suffix_v.end());
        }

        void apply_move9(
            const std::vector<int>& route_u,
            const std::vector<int>& route_v,
            int pos_u,
            int pos_v,
            std::vector<int>& out_a,
            std::vector<int>& out_b)
        {
            std::vector<int> prefix_u(route_u.begin(), route_u.begin() + pos_u + 1);
            std::vector<int> suffix_u(route_u.begin() + pos_u + 1, route_u.end());
            std::vector<int> prefix_v;
            std::vector<int> suffix_v;
            if (pos_v < 0)
            {
                suffix_v = route_v;
            }
            else
            {
                prefix_v.assign(route_v.begin(), route_v.begin() + pos_v + 1);
                suffix_v.assign(route_v.begin() + pos_v + 1, route_v.end());
            }
            out_a = std::move(prefix_u);
            out_a.insert(out_a.end(), suffix_v.begin(), suffix_v.end());
            out_b = std::move(prefix_v);
            out_b.insert(out_b.end(), suffix_u.begin(), suffix_u.end());
        }
    }

    void improve_pool_routes(
        std::vector<std::vector<int>>& routes,
        const std::vector<Point>& nodes,
        double capacity,
        InterRouteStats& stats)
    {
        const auto started = std::chrono::steady_clock::now();
        stats.move1 = 0;
        stats.move2 = 0;
        stats.move8 = 0;
        stats.move9 = 0;
        stats.passes = 0;

        const int initial_routes = static_cast<int>(routes.size());
        int n = 0;
        for (const auto& route : routes)
        {
            n += static_cast<int>(route.size());
        }
        if (n < 2 || initial_routes < 2)
        {
            stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            return;
        }

        const int depot = 0;
        const int k = std::min(kNeighbours, n - 1);
        const int num_tiles = (n + kTileSize - 1) / kTileSize;

        // 1. Assign each unique pool customer a fixed local index in [0, n - 1]
        std::vector<int> local_to_global(n);
        std::vector<int> global_to_local(nodes.size(), -1);
        int local_idx = 0;
        for (const auto& route : routes)
        {
            for (int cust : route)
            {
                local_to_global[local_idx] = cust;
                global_to_local[cust] = local_idx;
                ++local_idx;
            }
        }

        // 2. Compute static customer properties and tile bounding boxes
        std::vector<double> x(n), y(n), demand(n);
        std::vector<float2> coords(n);
        for (int i = 0; i < n; ++i)
        {
            const int gid = local_to_global[i];
            x[i] = nodes[gid].x;
            y[i] = nodes[gid].y;
            demand[i] = nodes[gid].demand;
            coords[i] = make_float2(static_cast<float>(x[i]), static_cast<float>(y[i]));
        }

        std::vector<float4> tile_boxes(num_tiles);
        for (int t = 0; t < num_tiles; ++t)
        {
            float xmin = 1e30f;
            float xmax = -1e30f;
            float ymin = 1e30f;
            float ymax = -1e30f;
            const int start = t * kTileSize;
            const int end   = std::min(n, start + kTileSize);
            for (int i = start; i < end; ++i)
            {
                const float px = coords[i].x;
                const float py = coords[i].y;
                if (px < xmin) xmin = px;
                if (px > xmax) xmax = px;
                if (py < ymin) ymin = py;
                if (py > ymax) ymax = py;
            }
            tile_boxes[t] = make_float4(xmin, xmax, ymin, ymax);
        }

        // 3. Allocate device memory
        double* d_x = nullptr;
        double* d_y = nullptr;
        double* d_demand = nullptr;
        double* d_cum = nullptr;
        double* d_route_load = nullptr;
        double* d_best_delta = nullptr;
        int* d_pred = nullptr;
        int* d_succ = nullptr;
        int* d_route = nullptr;
        int* d_nbr = nullptr;
        int* d_best_type = nullptr;
        int* d_best_arg = nullptr;
        int* d_best_rv = nullptr;
        float2* d_coords = nullptr;
        float4* d_tile_boxes = nullptr;

        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_demand, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cum, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_route_load, initial_routes * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pred, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_succ, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_route, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nbr, static_cast<size_t>(k) * n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_delta, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_best_type, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_arg, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_rv, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_coords, n * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_tile_boxes, num_tiles * sizeof(float4)));

        // 4. Upload static coordinates, demands, and tile bounding boxes ONCE
        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_demand, demand.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coords, coords.data(), n * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tile_boxes, tile_boxes.data(), num_tiles * sizeof(float4), cudaMemcpyHostToDevice));

        constexpr int threads = 256;
        const int blocks = (n + threads - 1) / threads;

        // 5. Compute static spatial nearest neighbors ONCE
        static_pool_neighbour_kernel<<<blocks, threads>>>(d_coords, d_tile_boxes, num_tiles, n, k, d_nbr);
        CUDA_CHECK(cudaGetLastError());

        // Preallocate host buffers for dynamic route state
        std::vector<int> pos_in_route(n);
        std::vector<int> pred(n), succ(n), route_of(n);
        std::vector<double> cum_load(n);
        std::vector<double> route_load(initial_routes);
        std::vector<double> best_delta(n);
        std::vector<int> best_type(n), best_arg(n), best_rv(n);
        std::vector<int> winner(initial_routes);
        std::vector<Candidate> chosen;
        chosen.reserve(initial_routes);
        std::vector<char> used(initial_routes);
        std::vector<char> drop(initial_routes);
        std::vector<std::vector<int>> replacement(initial_routes);

        constexpr int kMaxPasses = 30;

        for (int pass = 0; pass < kMaxPasses; ++pass)
        {
            const int current_routes = static_cast<int>(routes.size());
            if (current_routes < 2)
            {
                break;
            }

            for (int r = 0; r < current_routes; ++r)
            {
                double running = 0.0;
                const int len = static_cast<int>(routes[r].size());
                for (int p = 0; p < len; ++p)
                {
                    const int gid = routes[r][p];
                    const int u = global_to_local[gid];
                    pos_in_route[u] = p;
                    running += demand[u];
                    cum_load[u] = running;
                    pred[u] = (p == 0) ? -1 : global_to_local[routes[r][p - 1]];
                    succ[u] = (p + 1 == len) ? -1 : global_to_local[routes[r][p + 1]];
                    route_of[u] = r;
                }
                route_load[r] = running;
            }

            // Only dynamic route topologies/loads are transferred each pass
            CUDA_CHECK(cudaMemcpy(d_cum, cum_load.data(), n * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_route_load, route_load.data(), current_routes * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pred, pred.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_succ, succ.data(), n * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_route, route_of.data(), n * sizeof(int), cudaMemcpyHostToDevice));

            evaluate_moves_kernel<<<blocks, threads>>>(
                d_x, d_y, d_demand, d_cum, d_pred, d_succ, d_route, d_route_load, d_nbr,
                n, k, nodes[depot].x, nodes[depot].y, capacity,
                d_best_delta, d_best_type, d_best_arg, d_best_rv);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(best_delta.data(), d_best_delta, n * sizeof(double), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_type.data(), d_best_type, n * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_arg.data(), d_best_arg, n * sizeof(int), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(best_rv.data(), d_best_rv, n * sizeof(int), cudaMemcpyDeviceToHost));

            winner.assign(current_routes, -1);
            for (int i = 0; i < n; ++i)
            {
                if (best_type[i] == 0 || best_rv[i] < 0 || best_rv[i] == route_of[i])
                {
                    continue;
                }
                const int r = route_of[i];
                if (winner[r] < 0 || best_delta[i] < best_delta[winner[r]])
                {
                    winner[r] = i;
                }
            }

            chosen.clear();
            for (int r = 0; r < current_routes; ++r)
            {
                if (winner[r] < 0)
                {
                    continue;
                }
                const int u = winner[r];
                chosen.push_back({best_delta[u], u, best_type[u], best_arg[u], r, best_rv[u]});
            }

            if (chosen.empty())
            {
                break;
            }

            std::sort(chosen.begin(), chosen.end(), [](const Candidate& a, const Candidate& b) {
                return a.delta < b.delta;
            });

            used.assign(current_routes, 0);
            drop.assign(current_routes, 0);
            int applied_moves = 0;

            for (const Candidate& move : chosen)
            {
                if (used[move.route_u] || used[move.route_v])
                {
                    continue;
                }
                const int pos_u = pos_in_route[move.u];
                const int pos_v = (move.arg < 0) ? -1 : pos_in_route[move.arg];
                std::vector<int> rep_u;
                std::vector<int> rep_v;

                if (move.type == 1)
                {
                    rep_u = routes[move.route_u];
                    rep_v = routes[move.route_v];
                    apply_move1(rep_u, rep_v, pos_u, pos_v);
                }
                else if (move.type == 2)
                {
                    rep_u = routes[move.route_u];
                    rep_v = routes[move.route_v];
                    apply_move2(rep_u, rep_v, pos_u, pos_v);
                }
                else if (move.type == 8)
                {
                    apply_move8(routes[move.route_u], routes[move.route_v], pos_u, pos_v, rep_u, rep_v);
                }
                else if (move.type == 9)
                {
                    apply_move9(routes[move.route_u], routes[move.route_v], pos_u, pos_v, rep_u, rep_v);
                }
                else
                {
                    continue;
                }

                // Strict capacity verification on host to ensure zero violations
                double load_u = 0.0;
                for (int c : rep_u) load_u += nodes[c].demand;
                double load_v = 0.0;
                for (int c : rep_v) load_v += nodes[c].demand;
                if (load_u > capacity || load_v > capacity)
                {
                    continue;
                }

                if (move.type == 1) ++stats.move1;
                else if (move.type == 2) ++stats.move2;
                else if (move.type == 8) ++stats.move8;
                else if (move.type == 9) ++stats.move9;
                ++applied_moves;

                if (rep_u.empty()) drop[move.route_u] = 1;
                if (rep_v.empty()) drop[move.route_v] = 1;
                replacement[move.route_u] = std::move(rep_u);
                replacement[move.route_v] = std::move(rep_v);
                used[move.route_u] = 1;
                used[move.route_v] = 1;
            }

            if (applied_moves == 0)
            {
                break;
            }

            ++stats.passes;

            std::vector<std::vector<int>> updated_routes;
            updated_routes.reserve(routes.size());
            for (int r = 0; r < current_routes; ++r)
            {
                if (drop[r])
                {
                    continue;
                }
                if (used[r])
                {
                    updated_routes.push_back(std::move(replacement[r]));
                }
                else
                {
                    updated_routes.push_back(std::move(routes[r]));
                }
            }
            routes.swap(updated_routes);
        }

        cudaFree(d_coords);
        cudaFree(d_tile_boxes);
        cudaFree(d_x);
        cudaFree(d_y);
        cudaFree(d_demand);
        cudaFree(d_cum);
        cudaFree(d_route_load);
        cudaFree(d_pred);
        cudaFree(d_succ);
        cudaFree(d_route);
        cudaFree(d_nbr);
        cudaFree(d_best_delta);
        cudaFree(d_best_type);
        cudaFree(d_best_arg);
        cudaFree(d_best_rv);

        stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
}
