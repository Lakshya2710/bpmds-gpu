#include "Dynamic.h"
#include "Bucket_Partitioned_MDS.h"
#include "Gpu/NearestKernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Bucket_Partitioned_MDS
{
    namespace
    {
        // Old customers pulled into the pool with each newly added customer.
        constexpr int kPoolNeighbours = 20;

        struct VrpInstance
        {
            std::string name;
            std::string comment;
            std::string type = "CVRP";
            std::string edge_weight = "EUC_2D";
            double capacity = 0.0;
            double distance_limit = 1.0e30;
            bool has_distance = false;
            std::vector<Point> nodes;
        };

        struct AddEvent
        {
            double x;
            double y;
            double demand;
        };

        std::string trim(const std::string& text)
        {
            const size_t first = text.find_first_not_of(" \t\r\n");
            if (first == std::string::npos)
            {
                return "";
            }
            const size_t last = text.find_last_not_of(" \t\r\n");
            return text.substr(first, last - first + 1);
        }

        VrpInstance read_vrp(const std::string& path)
        {
            std::ifstream in(path);
            if (!in)
            {
                HANDLE_ERROR("Cannot open VRP file: " + path, true);
            }

            VrpInstance vrp;
            std::string line;
            int dimension = 0;
            while (std::getline(in, line))
            {
                line = trim(line);
                if (line.empty())
                {
                    continue;
                }
                if (line == "NODE_COORD_SECTION")
                {
                    break;
                }

                std::string key;
                std::string val;
                const size_t colon = line.find(':');
                if (colon != std::string::npos)
                {
                    key = trim(line.substr(0, colon));
                    val = trim(line.substr(colon + 1));
                }
                else
                {
                    std::istringstream ss(line);
                    ss >> key;
                    std::getline(ss, val);
                    val = trim(val);
                }

                if (key == "NAME")
                {
                    vrp.name = val;
                }
                else if (key == "COMMENT")
                {
                    vrp.comment = val;
                }
                else if (key == "TYPE")
                {
                    vrp.type = val;
                }
                else if (key == "DIMENSION")
                {
                    dimension = std::stoi(val);
                }
                else if (key == "EDGE_WEIGHT_TYPE")
                {
                    vrp.edge_weight = val;
                }
                else if (key == "CAPACITY")
                {
                    vrp.capacity = std::stod(val);
                }
                else if (key == "DISTANCE")
                {
                    vrp.distance_limit = std::stod(val);
                    vrp.has_distance = true;
                }
            }

            if (dimension <= 1)
            {
                HANDLE_ERROR("DIMENSION not found in VRP file", true);
            }
            if (vrp.capacity <= 0.0)
            {
                HANDLE_ERROR("CAPACITY not found in VRP file", true);
            }

            vrp.nodes.resize(dimension);
            for (int i = 0; i < dimension; ++i)
            {
                int id = 0;
                if (!(in >> id >> vrp.nodes[i].x >> vrp.nodes[i].y))
                {
                    HANDLE_ERROR("Could not read node coordinates", true);
                }
            }

            std::string section;
            in >> section;
            if (section != "DEMAND_SECTION")
            {
                HANDLE_ERROR("Expected DEMAND_SECTION, got: " + section, true);
            }
            for (int i = 0; i < dimension; ++i)
            {
                int id = 0;
                if (!(in >> id >> vrp.nodes[i].demand))
                {
                    HANDLE_ERROR("Could not read node demands", true);
                }
            }
            return vrp;
        }

        std::vector<std::vector<node_t>> read_solution(const std::string& path, int customer_count)
        {
            std::ifstream in(path);
            if (!in)
            {
                HANDLE_ERROR("Cannot open solution file: " + path, true);
            }

            std::vector<std::vector<node_t>> routes;
            std::vector<int> seen(customer_count, 0);
            std::string token;
            while (in >> token)
            {
                if (token != "Route")
                {
                    break;
                }
                in >> token;
                std::string line;
                std::getline(in, line);
                std::istringstream ss(line);
                std::vector<node_t> route;
                int customer = 0;
                while (ss >> customer)
                {
                    if (customer <= 0 || customer >= customer_count)
                    {
                        HANDLE_ERROR("Solution customer " + std::to_string(customer) + " is outside 1.." + std::to_string(customer_count - 1), true);
                    }
                    if (seen[customer] != 0)
                    {
                        HANDLE_ERROR("Customer " + std::to_string(customer) + " appears in more than one route", true);
                    }
                    seen[customer] = 1;
                    route.push_back(customer);
                }
                if (!route.empty())
                {
                    routes.push_back(std::move(route));
                }
            }

            for (int customer = 1; customer < customer_count; ++customer)
            {
                if (seen[customer] == 0)
                {
                    HANDLE_ERROR("Customer " + std::to_string(customer) + " is missing from the solution", true);
                }
            }
            return routes;
        }

        void read_events(
            const std::string& path,
            int customer_count,
            std::vector<int>& deleted,
            std::vector<AddEvent>& added)
        {
            std::ifstream in(path);
            if (!in)
            {
                HANDLE_ERROR("Cannot open events file: " + path, true);
            }

            std::string action;
            while (in >> action)
            {
                if (action == "ADD")
                {
                    AddEvent event{};
                    if (!(in >> event.x >> event.y >> event.demand))
                    {
                        HANDLE_ERROR("ADD event must be: ADD x y demand", true);
                    }
                    added.push_back(event);
                }
                else if (action == "DEL")
                {
                    int id = 0;
                    if (!(in >> id))
                    {
                        HANDLE_ERROR("DEL event must be: DEL id", true);
                    }
                    if (id <= 0 || id >= customer_count)
                    {
                        HANDLE_ERROR("DEL id " + std::to_string(id) + " is outside 1.." + std::to_string(customer_count - 1), true);
                    }
                    deleted.push_back(id);
                }
                else
                {
                    HANDLE_ERROR("Unknown event '" + action + "'. Expected ADD or DEL", true);
                }
            }

            std::sort(deleted.begin(), deleted.end());
            deleted.erase(std::unique(deleted.begin(), deleted.end()), deleted.end());
        }

        // One pass: drop deleted customers and append new ones. Surviving ids stay
        // in their previous order, which matches deleting from the highest id down.
        void apply_events(
            const VrpInstance& original,
            const std::vector<std::vector<node_t>>& original_routes,
            const std::vector<int>& deleted,
            const std::vector<AddEvent>& added,
            VrpInstance& updated,
            std::vector<std::vector<node_t>>& routes,
            std::vector<char>& route_lost_customer,
            std::vector<int>& new_customers)
        {
            const int original_count = static_cast<int>(original.nodes.size());
            std::vector<char> removed(original_count, 0);
            for (int id : deleted)
            {
                removed[id] = 1;
            }

            std::vector<int> old_to_new(original_count, -1);
            updated = original;
            updated.nodes.clear();
            updated.nodes.push_back(original.nodes[0]);
            old_to_new[0] = 0;

            for (int id = 1; id < original_count; ++id)
            {
                if (removed[id])
                {
                    continue;
                }
                old_to_new[id] = static_cast<int>(updated.nodes.size());
                updated.nodes.push_back(original.nodes[id]);
            }

            new_customers.clear();
            new_customers.reserve(added.size());
            for (const AddEvent& event : added)
            {
                const int id = static_cast<int>(updated.nodes.size());
                updated.nodes.emplace_back(event.x, event.y, event.demand);
                new_customers.push_back(id);
            }

            routes.clear();
            route_lost_customer.clear();
            for (const std::vector<node_t>& route : original_routes)
            {
                std::vector<node_t> kept;
                bool lost = false;
                for (node_t customer : route)
                {
                    if (removed[customer])
                    {
                        lost = true;
                        continue;
                    }
                    kept.push_back(old_to_new[customer]);
                }
                if (kept.empty())
                {
                    continue;
                }
                routes.push_back(std::move(kept));
                route_lost_customer.push_back(lost ? 1 : 0);
            }
        }

        distance_t route_cost(const CVRP& cvrp, const std::vector<node_t>& route)
        {
            distance_t cost = cvrp.distance(cvrp.depot(), route.front());
            for (size_t i = 1; i < route.size(); ++i)
            {
                cost += cvrp.distance(route[i - 1], route[i]);
            }
            cost += cvrp.distance(route.back(), cvrp.depot());
            return cost;
        }

        // Same rounded edge length d-cvrp.cu writes into "Cost".
        long long rounded_solution_cost(const CVRP& cvrp, const std::vector<std::vector<node_t>>& routes)
        {
            double total = 0.0;
            for (const std::vector<node_t>& route : routes)
            {
                auto edge = [&](node_t u, node_t v) {
                    const float dx = static_cast<float>(cvrp[u].x) - static_cast<float>(cvrp[v].x);
                    const float dy = static_cast<float>(cvrp[u].y) - static_cast<float>(cvrp[v].y);
                    return static_cast<double>(roundf(sqrtf(dx * dx + dy * dy)));
                };
                total += edge(cvrp.depot(), route.front());
                for (size_t i = 1; i < route.size(); ++i)
                {
                    total += edge(route[i - 1], route[i]);
                }
                total += edge(route.back(), cvrp.depot());
            }
            return static_cast<long long>(std::llround(total));
        }

        void ensure_parent_directory(const std::string& path)
        {
            const std::filesystem::path parent = std::filesystem::path(path).parent_path();
            if (parent.empty())
            {
                return;
            }
            std::error_code error;
            std::filesystem::create_directories(parent, error);
            if (error)
            {
                HANDLE_ERROR("Cannot create directory: " + parent.string(), true);
            }
        }

        void write_solution(
            const std::string& path,
            const std::vector<std::vector<node_t>>& routes,
            long long cost)
        {
            ensure_parent_directory(path);
            std::ofstream out(path);
            if (!out)
            {
                HANDLE_ERROR("Cannot open output solution: " + path, true);
            }
            for (size_t r = 0; r < routes.size(); ++r)
            {
                out << "Route #" << (r + 1) << ":";
                for (node_t customer : routes[r])
                {
                    out << " " << customer;
                }
                out << "\n";
            }
            out << "Cost " << cost << "\n";
        }

        void write_number(std::ostream& out, double value)
        {
            if (value == std::floor(value))
            {
                out << static_cast<long long>(std::llround(value));
            }
            else
            {
                out << value;
            }
        }

        void write_vrp(
            const std::string& path,
            const VrpInstance& vrp,
            int added,
            int removed)
        {
            ensure_parent_directory(path);
            std::ofstream out(path);
            if (!out)
            {
                HANDLE_ERROR("Cannot open output VRP: " + path, true);
            }

            out << "NAME : " << vrp.name << "-updated\n";
            out << "COMMENT : " << vrp.comment
                << " (Batch updated (" << added << " added, " << removed << " removed))\n";
            out << "TYPE : " << vrp.type << "\n";
            out << "DIMENSION : " << vrp.nodes.size() << "\n";
            out << "EDGE_WEIGHT_TYPE : " << vrp.edge_weight << "\n";
            out << "CAPACITY : ";
            write_number(out, vrp.capacity);
            out << "\n";
            if (vrp.has_distance)
            {
                out << "DISTANCE : ";
                write_number(out, vrp.distance_limit);
                out << "\n";
            }
            out << "NODE_COORD_SECTION\n";
            for (size_t i = 0; i < vrp.nodes.size(); ++i)
            {
                out << " " << (i + 1) << " "
                    << static_cast<long long>(std::llround(vrp.nodes[i].x)) << " "
                    << static_cast<long long>(std::llround(vrp.nodes[i].y)) << "\n";
            }
            out << "DEMAND_SECTION\n";
            for (size_t i = 0; i < vrp.nodes.size(); ++i)
            {
                out << (i + 1) << " "
                    << static_cast<long long>(std::llround(vrp.nodes[i].demand)) << "\n";
            }
            out << "DEPOT_SECTION\n 1\n -1\nEOF\n";
        }
    }

    void run_dynamic(const DynamicRequest& request)
    {
        const auto started = std::chrono::steady_clock::now();

        const VrpInstance original = read_vrp(request.vrp_in);
        const int original_count = static_cast<int>(original.nodes.size());
        const std::vector<std::vector<node_t>> original_routes = read_solution(request.sol_in, original_count);

        std::vector<int> deleted;
        std::vector<AddEvent> added;
        read_events(request.events_in, original_count, deleted, added);

        const auto events_started = std::chrono::steady_clock::now();
        VrpInstance updated;
        std::vector<std::vector<node_t>> routes;
        std::vector<char> route_lost_customer;
        std::vector<int> new_customers;
        apply_events(original, original_routes, deleted, added, updated, routes, route_lost_customer, new_customers);
        const double event_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - events_started).count();

        const int updated_count = static_cast<int>(updated.nodes.size());
        std::vector<char> rebuild(routes.size(), 0);
        for (size_t r = 0; r < routes.size(); ++r)
        {
            rebuild[r] = route_lost_customer[r];
        }

        std::vector<int> route_of(updated_count, -1);
        for (size_t r = 0; r < routes.size(); ++r)
        {
            for (node_t customer : routes[r])
            {
                route_of[customer] = static_cast<int>(r);
            }
        }

        std::vector<int> survivors;
        survivors.reserve(updated_count);
        std::vector<char> is_new(updated_count, 0);
        for (int id : new_customers)
        {
            is_new[id] = 1;
        }
        for (int id = 1; id < updated_count; ++id)
        {
            if (!is_new[id])
            {
                survivors.push_back(id);
            }
        }

        std::vector<double> survivor_x;
        std::vector<double> survivor_y;
        std::vector<double> query_x;
        std::vector<double> query_y;
        survivor_x.reserve(survivors.size());
        survivor_y.reserve(survivors.size());
        query_x.reserve(new_customers.size());
        query_y.reserve(new_customers.size());
        for (int id : survivors)
        {
            survivor_x.push_back(updated.nodes[id].x);
            survivor_y.push_back(updated.nodes[id].y);
        }
        for (int id : new_customers)
        {
            query_x.push_back(updated.nodes[id].x);
            query_y.push_back(updated.nodes[id].y);
        }

        const auto neighbours_started = std::chrono::steady_clock::now();
        std::vector<int> neighbour_ids;
        Gpu::nearest_surviving_customers(
            survivor_x, survivor_y, survivors, query_x, query_y, kPoolNeighbours, neighbour_ids);
        const double neighbour_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - neighbours_started).count();

        for (size_t q = 0; q < new_customers.size(); ++q)
        {
            for (int t = 0; t < kPoolNeighbours; ++t)
            {
                const int neighbour = neighbour_ids[q * kPoolNeighbours + t];
                if (neighbour < 0)
                {
                    continue;
                }
                const int route = route_of[neighbour];
                if (route >= 0)
                {
                    rebuild[route] = 1;
                }
            }
        }

        std::vector<char> in_pool(updated_count, 0);
        in_pool[0] = 1;
        for (int id : new_customers)
        {
            in_pool[id] = 1;
        }

        std::vector<std::vector<node_t>> kept_routes;
        int rebuilt_route_count = 0;
        for (size_t r = 0; r < routes.size(); ++r)
        {
            if (!rebuild[r])
            {
                kept_routes.push_back(routes[r]);
                continue;
            }
            ++rebuilt_route_count;
            for (node_t customer : routes[r])
            {
                in_pool[customer] = 1;
            }
        }

        std::vector<int> local_to_real;
        std::vector<Point> pool_nodes;
        local_to_real.push_back(0);
        pool_nodes.push_back(updated.nodes[0]);
        for (int id = 1; id < updated_count; ++id)
        {
            if (!in_pool[id])
            {
                continue;
            }
            local_to_real.push_back(id);
            pool_nodes.push_back(updated.nodes[id]);
        }

        std::vector<std::vector<node_t>> final_routes = std::move(kept_routes);
        double solve_seconds = 0.0;
        const int pool_customers = static_cast<int>(pool_nodes.size()) - 1;
        if (pool_customers > 0)
        {
            const CVRP pool(updated.capacity, pool_nodes);
            const Solver solver(request.alpha, request.rho);
            const Solution pool_solution = solver.solve(pool);
            solve_seconds = pool_solution.solving_time();

            for (const std::vector<node_t>& local_route : pool_solution.get_routes())
            {
                std::vector<node_t> real_route;
                real_route.reserve(local_route.size());
                for (node_t local : local_route)
                {
                    real_route.push_back(local_to_real[local]);
                }
                final_routes.push_back(std::move(real_route));
            }
        }

        const CVRP full(updated.capacity, updated.nodes);
        distance_t exact_cost = 0.0;
        for (const std::vector<node_t>& route : final_routes)
        {
            exact_cost += route_cost(full, route);
        }

        Solution stitched(solve_seconds, 0.0, exact_cost, final_routes);
        if (!stitched.verify(full))
        {
            HANDLE_ERROR("Stitched solution failed verification", true);
        }

        const long long reported_cost = rounded_solution_cost(full, final_routes);
        write_solution(request.sol_out, final_routes, reported_cost);
        write_vrp(request.vrp_out, updated, static_cast<int>(added.size()), static_cast<int>(deleted.size()));

        const int routes_kept = static_cast<int>(routes.size()) - rebuilt_route_count;
        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cerr << std::fixed;
        std::cerr.precision(4);
        std::cerr << "Customers in:     " << (original_count - 1) << "\n";
        std::cerr << "Deleted:          " << deleted.size() << "\n";
        std::cerr << "Added:            " << added.size() << "\n";
        std::cerr << "Customers out:    " << (updated_count - 1) << "\n";
        std::cerr << "Routes kept:      " << routes_kept << "\n";
        std::cerr << "Routes rebuilt:   " << rebuilt_route_count << "\n";
        std::cerr << "Routes out:       " << final_routes.size() << "\n";
        std::cerr << "Pool customers:   " << pool_customers << "\n";
        std::cerr << "Event time (s):   " << event_seconds << "\n";
        std::cerr << "Neighbour time (s): " << neighbour_seconds << "\n";
        std::cerr << "BPMDS time (s):   " << solve_seconds << "\n";
        std::cerr << "Total time (s):   " << total_seconds << "\n";
        std::cerr << "Exact cost:       " << exact_cost << "\n";
        std::cerr << "Rounded cost:     " << reported_cost << "\n";
    }
}
