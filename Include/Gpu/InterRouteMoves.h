#pragma once

#include "Utils.h"

#include <vector>

namespace Gpu
{
    struct InterRouteStats
    {
        double seconds = 0.0;
        int move1 = 0;
        int move8 = 0;
        int move9 = 0;
        int passes = 0;
    };

    // One pass of inter-route relocate (move 1) and 2-opt* (move 8)
    // on the routes BPMDS just built. Kept routes are not included.
    void improve_pool_routes(
        std::vector<std::vector<int>>& routes,
        const std::vector<Point>& nodes,
        double capacity,
        InterRouteStats& stats);
}
