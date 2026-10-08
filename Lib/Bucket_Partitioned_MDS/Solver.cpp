#include "Utils.h"
#include "Bucket_Partitioned_MDS.h"
#include "Gpu/SolverContext.h"
#include <chrono>
#include <cfloat>

namespace Bucket_Partitioned_MDS
{
    distance_t Solver::get_route_distance(
        const CVRP&                 cvrp,
        const std::vector <node_t>& route) const
    {
        node_t prev_node = cvrp.depot();
        distance_t cost = 0;

        for(auto& node: route)
        {
            cost += cvrp.distance(prev_node, node);
            prev_node = node;
        }
        cost += cvrp.distance(prev_node, cvrp.depot());

        return cost;
    }

    Solver::Solver(
        const double _alpha, 
        const int _rho) 
        : alpha(_alpha), rho(_rho) 
    {
    }

    Solution Solver::solve(
        const CVRP& cvrp) const
    {
        double maxMB_before_execution = get_curr_rss_mb();

        auto start = std::chrono::high_resolution_clock::now();

        distance_t final_cost   = 0.0;
        std::vector <std::vector<int>> final_routes;

        Gpu::SolverContext gpu_ctx(cvrp, alpha);
        gpu_ctx.partition();
        const int num_buckets = gpu_ctx.num_buckets();
	
	auto end1 = std::chrono::high_resolution_clock::now();
	std::cout<<"HERE1: "<<std::chrono::duration<double>(end1 - start).count()<<std::endl;

        gpu_ctx.build_all_msts_streamed();
	
	auto end2 = std::chrono::high_resolution_clock::now();
        std::cout<<"HERE2: "<<std::chrono::duration<double>(end2 - start).count()<<std::endl;

        gpu_ctx.run_all_route_trials_streamed(rho);
	
	auto end3 = std::chrono::high_resolution_clock::now();
        std::cout<<"HERE3: "<<std::chrono::duration<double>(end3 - start).count()<<std::endl;

	for (int bucket_id = 0; bucket_id < num_buckets; ++bucket_id)
        {
            std::vector<std::vector<node_t>> low_cost_routes;
            distance_t pre_opt_cost = DBL_MAX;

            gpu_ctx.fetch_best_routes_for_bucket(bucket_id, rho, low_cost_routes, pre_opt_cost);

            if (!low_cost_routes.empty())
            {
                distance_t optimized_bucket_cost = 0.0;
                for (auto& route : low_cost_routes)
                {
                    optimized_bucket_cost += get_route_distance(cvrp, route);
                    final_routes.push_back(std::move(route));
                }
                final_cost += optimized_bucket_cost;
            }
        }

        auto end                    = std::chrono::high_resolution_clock::now();
        double maxMB_after_execution = get_peak_rss_mb();

        double execution_time = std::chrono::duration<double>(end - start).count();
        double maxMB_difference = maxMB_after_execution - maxMB_before_execution;

        return Solution(execution_time, maxMB_difference, final_cost, final_routes);
    }
}
