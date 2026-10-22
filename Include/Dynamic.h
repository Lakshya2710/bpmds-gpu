#ifndef DYNAMIC_BPMDS_H
#define DYNAMIC_BPMDS_H

#include <string>

namespace Bucket_Partitioned_MDS
{
    struct DynamicRequest
    {
        double alpha;
        int rho;
        std::string vrp_in;
        std::string sol_in;
        std::string sol_out;
        std::string vrp_out;
        std::string events_in;
    };

    void run_dynamic(const DynamicRequest& request);
}

#endif
