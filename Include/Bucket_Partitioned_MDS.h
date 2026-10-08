#ifndef BUCKET_PARTITIONED_MDS_H
#define BUCKET_PARTITIONED_MDS_H

#include "Utils.h"
#include <vector>
#include <string>
#include <cfloat>
#include <iostream>

namespace Bucket_Partitioned_MDS
{ 
    class CVRP
    {
        /*
        * CVRP: Class for maintaining CVRP 
        */
        
    private: 
        capacity_t _capacity;
        int _size;
        std::vector <Point> node;
        std::string type;
        const node_t _depot = 0; // depot is always 0

    public:
        CVRP(
            std::istream&);
        distance_t distance(
            const node_t, 
            const node_t) const;
        const Point& operator[](
            const node_t) const;
        void print(
            std::ostream&) const;
        // Getters
        capacity_t capacity() const;
        int size() const;
        node_t depot() const;
    };

    class Solution 
    {
    private:
        double time_for_solving;
        double maxMB_difference;
        double cost;
        std::vector <std::vector <node_t>> routes;

        distance_t get_total_cost_of_routes(
            const CVRP& cvrp);
    public:
        Solution(
            const double, 
            const double, 
            const double,
            const std::vector <std::vector <node_t>>&);
        bool verify(
            const CVRP&) const;
        void print(
            std::ostream&) const;
    };

    class Solver 
    {
        /*
        * Solver: Bucket Partitioned MDS solver for solving CVRP
        */

    private:
        const double alpha;
        const int rho;

        distance_t get_route_distance(
            const CVRP&, 
            const std::vector <node_t>&) const;

    public:
        Solver(
            const double, 
            const int);

        Solution solve(
            const CVRP&) const;
    };
}

#endif