#include "Dynamic.h"

#include <iostream>
#include <string>
#include <vector>

namespace
{
    void print_usage()
    {
        std::cerr
            << "Usage:\n"
            << "  dynamic-bpmds --alpha=<deg> --rho=<n> <in.vrp> <in.sol> <out.sol> <out.vrp> <events.txt>\n";
    }
}

int main(int argc, char* argv[])
{
    double alpha = 0.0;
    int rho = 0;
    bool have_alpha = false;
    bool have_rho = false;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg.rfind("--alpha=", 0) == 0)
        {
            alpha = std::stod(arg.substr(8));
            have_alpha = true;
        }
        else if (arg.rfind("--rho=", 0) == 0)
        {
            rho = std::stoi(arg.substr(6));
            have_rho = true;
        }
        else
        {
            files.push_back(arg);
        }
    }

    if (!have_alpha || !have_rho || files.size() != 5 || alpha <= 0.0 || alpha > 360.0 || rho <= 0)
    {
        print_usage();
        return 1;
    }

    Bucket_Partitioned_MDS::DynamicRequest request;
    request.alpha = alpha;
    request.rho = rho;
    request.vrp_in = files[0];
    request.sol_in = files[1];
    request.sol_out = files[2];
    request.vrp_out = files[3];
    request.events_in = files[4];

    Bucket_Partitioned_MDS::run_dynamic(request);
    return 0;
}
