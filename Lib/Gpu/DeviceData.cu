#include "Gpu/DeviceData.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace Gpu
{
    namespace
    {
        void compute_separating_vectors(
            double              alpha,
            int                 num_buckets,
            std::vector<double>& sep_x,
            std::vector<double>& sep_y)
        {
            sep_x.resize(num_buckets + 1);
            sep_y.resize(num_buckets + 1);

            // x-axis unit vector; bucket boundaries at i * alpha degrees CCW
            sep_x[0] = 1.0;
            sep_y[0] = 0.0;
            sep_x[num_buckets] = 1.0;
            sep_y[num_buckets] = 0.0;

            for (int i = 1; i < num_buckets; ++i)
            {
                const double theta = i * alpha * PI / 180.0;
                sep_x[i] = std::cos(theta);
                sep_y[i] = std::sin(theta);
            }
        }
    }

    DeviceCVRP::DeviceCVRP(
        const Bucket_Partitioned_MDS::CVRP& cvrp,
        double                              alpha)
        : N_(cvrp.size())
        , num_buckets_(static_cast<int>(std::ceil(360.0 / alpha)))
        , alpha_(alpha)
    {
        std::vector<double> h_x(N_);
        std::vector<double> h_y(N_);
        std::vector<double> h_demand(N_);

        for (int i = 0; i < N_; ++i)
        {
            h_x[i]      = cvrp[i].x;
            h_y[i]      = cvrp[i].y;
            h_demand[i] = cvrp[i].demand;
        }

        std::vector<double> h_sep_x;
        std::vector<double> h_sep_y;
        compute_separating_vectors(alpha_, num_buckets_, h_sep_x, h_sep_y);

        CUDA_CHECK(cudaMalloc(&d_x_,      N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_y_,      N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_demand_, N_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sep_x_,  (num_buckets_ + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sep_y_,  (num_buckets_ + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_bucket_id_, N_ * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_x_,      h_x.data(),      N_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_y_,      h_y.data(),      N_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_demand_, h_demand.data(), N_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sep_x_,  h_sep_x.data(),  (num_buckets_ + 1) * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sep_y_,  h_sep_y.data(),  (num_buckets_ + 1) * sizeof(double), cudaMemcpyHostToDevice));
    }

    void DeviceCVRP::upload_bucket_layout(const BucketLayout& layout)
    {
        cudaFree(d_bucket_nodes_);
        cudaFree(d_bucket_offsets_);
        d_bucket_nodes_   = nullptr;
        d_bucket_offsets_ = nullptr;

        bucket_layout_size_ = static_cast<int>(layout.nodes.size());
        max_bucket_size_    = 0;

        for (int b = 0; b < layout.num_buckets; ++b)
        {
            const int k = layout.offsets[b + 1] - layout.offsets[b];
            max_bucket_size_ = std::max(max_bucket_size_, k);
        }

        if (bucket_layout_size_ == 0)
        {
            return;
        }

        CUDA_CHECK(cudaMalloc(&d_bucket_nodes_,   bucket_layout_size_ * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_bucket_offsets_, (layout.num_buckets + 1) * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(
            d_bucket_nodes_,
            layout.nodes.data(),
            bucket_layout_size_ * sizeof(int),
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_bucket_offsets_,
            layout.offsets.data(),
            (layout.num_buckets + 1) * sizeof(int),
            cudaMemcpyHostToDevice));
    }

    DeviceCVRP::~DeviceCVRP()
    {
        cudaFree(d_x_);
        cudaFree(d_y_);
        cudaFree(d_demand_);
        cudaFree(d_sep_x_);
        cudaFree(d_sep_y_);
        cudaFree(d_bucket_id_);
        cudaFree(d_bucket_nodes_);
        cudaFree(d_bucket_offsets_);
    }
}
