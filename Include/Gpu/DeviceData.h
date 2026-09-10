#pragma once

#include "CudaCommon.h"
#include "Bucket_Partitioned_MDS.h"
#include <vector>

namespace Gpu
{
    struct BucketLayout
    {
        std::vector<node_t> nodes;
        std::vector<int>    offsets;
        int                 num_buckets;
    };

    /*
     * DeviceCVRP: O(N) GPU resident problem data (coordinates + demands).
     * Separating vectors are O(num_buckets) and also live on device.
     */
    class DeviceCVRP
    {
    public:
        DeviceCVRP(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha);
        ~DeviceCVRP();

        DeviceCVRP(const DeviceCVRP&)            = delete;
        DeviceCVRP& operator=(const DeviceCVRP&) = delete;

        int  size() const { return N_; }
        int  num_buckets() const { return num_buckets_; }
        int  depot() const { return 0; }

        const double* device_x() const { return d_x_; }
        const double* device_y() const { return d_y_; }
        const double* device_demand() const { return d_demand_; }
        const double* device_sep_x() const { return d_sep_x_; }
        const double* device_sep_y() const { return d_sep_y_; }
        int*          device_bucket_id() { return d_bucket_id_; }

    private:
        int     N_;
        int     num_buckets_;
        double  alpha_;
        double* d_x_          = nullptr;
        double* d_y_          = nullptr;
        double* d_demand_     = nullptr;
        double* d_sep_x_      = nullptr;
        double* d_sep_y_      = nullptr;
        int*    d_bucket_id_  = nullptr;
    };

    BucketLayout assign_and_compact_buckets(DeviceCVRP& device);
}
