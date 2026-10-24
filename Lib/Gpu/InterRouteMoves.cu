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
        __global__ void pool_neighbour_kernel(
            const double* __restrict__ x,
            const double* __restrict__ y,
            const int* __restrict__ route_of,
            int n,
            int k,
            int* __restrict__ nbr)
        {
            const int query = blockIdx.x * blockDim.x + threadIdx.x;
            if (query >= n)
            {
                return;
            }

            // Force arrays into registers by initializing them with fixed bounds
            double best_dist[kMaxNeighbours];
            int best_id[kMaxNeighbours];

            #pragma unroll
            for (int i = 0; i < kMaxNeighbours; ++i)
            {
                best_dist[i] = 1e30; // Infinity removes the need for a 'filled' tracker
                best_id[i] = -1;
            }

            // Wrap reads in __ldg() to use the instant L1 Texture Cache
            const double qx = __ldg(&x[query]);
            const double qy = __ldg(&y[query]);
            const int qroute = __ldg(&route_of[query]);

            for (int s = 0; s < n; ++s)
            {
                if (s == query)
                {
                    continue;
                }
                
                const int sroute = __ldg(&route_of[s]);
                if (sroute == qroute)
                {
                    continue;
                }
                
                const double dx = __ldg(&x[s]) - qx;
                const double dy = __ldg(&y[s]) - qy;
                const double dist2 = dx * dx + dy * dy;

                // Only engage the update logic if it beats our worst tracked neighbor
                if (dist2 < best_dist[k - 1])
                {
                    double current_dist = dist2;
                    int current_id = s;
                    
                    // A fully unrolled "Bubble Down" insertion sort
                    // This strictly forces the arrays to remain in registers.
                    #pragma unroll
                    for (int i = 0; i < kMaxNeighbours; ++i)
                    {
                        if (i >= k) break; 
                        
                        if (current_dist < best_dist[i])
                        {
                            double tmp_dist = best_dist[i];
                            int tmp_id = best_id[i];
                            
                            best_dist[i] = current_dist;
                            best_id[i] = current_id;
                            
                            current_dist = tmp_dist;
                            current_id = tmp_id;
                        }
                    }
                }
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

                const int prev = pred[i];
                const int next = succ[i];
                const int after = succ[j];
                double px, py, nx, ny, ix, iy, jx, jy, ax, ay;
                point_of(prev, x, y, depot_x, depot_y, px, py);
                point_of(next, x, y, depot_x, depot_y, nx, ny);
                point_of(i, x, y, depot_x, depot_y, ix, iy);
                point_of(j, x, y, depot_x, depot_y, jx, jy);
                point_of(after, x, y, depot_x, depot_y, ax, ay);

                const double remove = euc(px, py, nx, ny) - euc(px, py, ix, iy) - euc(ix, iy, nx, ny);
                const double insert = euc(jx, jy, ix, iy) + euc(ix, iy, ax, ay) - euc(jx, jy, ax, ay);
                const double relocate = remove + insert;
                if (route_load[rv] + dem_i <= capacity + kImproveEps && relocate < best - kImproveEps)
                {
                    best = relocate;
                    type = 1;
                    arg = j;
                    route_v = rv;
                }

                const double link = euc(ix, iy, jx, jy) + euc(nx, ny, ax, ay)
                                  - euc(ix, iy, nx, ny) - euc(jx, jy, ax, ay);
                const double new_a = cum_load[i] + cum_load[j];
                const double new_b = load_u + route_load[rv] - new_a;
                if (new_a <= capacity + kImproveEps && new_b <= capacity + kImproveEps && link < best - kImproveEps)
                {
                    best = link;
                    type = 8;
                    arg = j;
                    route_v = rv;
                }

                if (pred[j] < 0)
                {
                    double dx, dy;
                    point_of(-1, x, y, depot_x, depot_y, dx, dy);
                    const double insert_front = euc(dx, dy, ix, iy) + euc(ix, iy, jx, jy) - euc(dx, dy, jx, jy);
                    const double relocate_front = remove + insert_front;
                    if (route_load[rv] + dem_i <= capacity + kImproveEps && relocate_front < best - kImproveEps)
                    {
                        best = relocate_front;
                        type = 1;
                        arg = -1;
                        route_v = rv;
                    }
                    const double link_depot = euc(ix, iy, dx, dy) + euc(nx, ny, jx, jy)
                                            - euc(ix, iy, nx, ny) - euc(dx, dy, jx, jy);
                    const double depot_a = cum_load[i];
                    const double depot_b = load_u + route_load[rv] - depot_a;
                    if (depot_a <= capacity + kImproveEps && depot_b <= capacity + kImproveEps
                        && link_depot < best - kImproveEps)
                    {
                        best = link_depot;
                        type = 8;
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
    }

    void improve_pool_routes(
        std::vector<std::vector<int>>& routes,
        const std::vector<Point>& nodes,
        double capacity,
        InterRouteStats& stats)
    {
        const auto started = std::chrono::steady_clock::now();
        stats.move1 = 0;
        stats.move8 = 0;

        const int nroutes = static_cast<int>(routes.size());
        int n = 0;
        for (const auto& route : routes)
        {
            n += static_cast<int>(route.size());
        }
        if (n < 2 || nroutes < 2)
        {
            stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            return;
        }

        const int depot = 0;
        std::vector<int> global_id(n);
        std::vector<double> x(n), y(n), demand(n), cum_load(n), route_load(nroutes);
        std::vector<int> pred(n), succ(n), route_of(n), route_start(nroutes);
        int local = 0;
        for (int r = 0; r < nroutes; ++r)
        {
            route_start[r] = local;
            double running = 0.0;
            const int len = static_cast<int>(routes[r].size());
            for (int p = 0; p < len; ++p, ++local)
            {
                const int id = routes[r][p];
                global_id[local] = id;
                x[local] = nodes[id].x;
                y[local] = nodes[id].y;
                demand[local] = nodes[id].demand;
                running += nodes[id].demand;
                cum_load[local] = running;
                pred[local] = (p == 0) ? -1 : local - 1;
                succ[local] = (p + 1 == len) ? -1 : local + 1;
                route_of[local] = r;
            }
            route_load[r] = running;
        }

        const int k = std::min(kNeighbours, n - 1);
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

        CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_demand, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_cum, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_route_load, nroutes * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pred, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_succ, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_route, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_nbr, static_cast<size_t>(k) * n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_delta, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_best_type, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_arg, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_best_rv, n * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y, y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_demand, demand.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cum, cum_load.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_route_load, route_load.data(), nroutes * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pred, pred.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_succ, succ.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_route, route_of.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        constexpr int threads = 128;
        const int blocks = (n + threads - 1) / threads;
        pool_neighbour_kernel<<<blocks, threads>>>(d_x, d_y, d_route, n, k, d_nbr);
        CUDA_CHECK(cudaGetLastError());
        evaluate_moves_kernel<<<blocks, threads>>>(
            d_x, d_y, d_demand, d_cum, d_pred, d_succ, d_route, d_route_load, d_nbr,
            n, k, nodes[depot].x, nodes[depot].y, capacity,
            d_best_delta, d_best_type, d_best_arg, d_best_rv);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> best_delta(n);
        std::vector<int> best_type(n), best_arg(n), best_rv(n);
        CUDA_CHECK(cudaMemcpy(best_delta.data(), d_best_delta, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(best_type.data(), d_best_type, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(best_arg.data(), d_best_arg, n * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(best_rv.data(), d_best_rv, n * sizeof(int), cudaMemcpyDeviceToHost));

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

        std::vector<int> winner(nroutes, -1);
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

        std::vector<Candidate> chosen;
        chosen.reserve(nroutes);
        for (int r = 0; r < nroutes; ++r)
        {
            if (winner[r] < 0)
            {
                continue;
            }
            const int u = winner[r];
            chosen.push_back({best_delta[u], u, best_type[u], best_arg[u], r, best_rv[u]});
        }
        std::sort(chosen.begin(), chosen.end(), [](const Candidate& a, const Candidate& b) {
            return a.delta < b.delta;
        });

        std::vector<char> used(nroutes, 0);
        std::vector<std::vector<int>> replacement(nroutes);
        std::vector<char> drop(nroutes, 0);

        for (const Candidate& move : chosen)
        {
            if (used[move.route_u] || used[move.route_v])
            {
                continue;
            }
            const int pos_u = move.u - route_start[move.route_u];
            const int pos_v = (move.arg < 0) ? -1 : move.arg - route_start[move.route_v];
            if (move.type == 1)
            {
                std::vector<int> left = routes[move.route_u];
                std::vector<int> right = routes[move.route_v];
                apply_move1(left, right, pos_u, pos_v);
                replacement[move.route_u] = std::move(left);
                replacement[move.route_v] = std::move(right);
                if (replacement[move.route_u].empty())
                {
                    drop[move.route_u] = 1;
                }
                ++stats.move1;
            }
            else
            {
                std::vector<int> out_a;
                std::vector<int> out_b;
                apply_move8(routes[move.route_u], routes[move.route_v], pos_u, pos_v, out_a, out_b);
                replacement[move.route_u] = std::move(out_a);
                replacement[move.route_v] = std::move(out_b);
                if (replacement[move.route_u].empty())
                {
                    drop[move.route_u] = 1;
                }
                if (replacement[move.route_v].empty())
                {
                    drop[move.route_v] = 1;
                }
                ++stats.move8;
            }
            used[move.route_u] = 1;
            used[move.route_v] = 1;
        }

        std::vector<std::vector<int>> updated_routes;
        updated_routes.reserve(routes.size());
        for (int r = 0; r < nroutes; ++r)
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
        stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    }
}
