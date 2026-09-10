#include "Gpu/SolverContext.h"
#include "Gpu/DeviceData.h"
#include "Gpu/MstKernels.h"

#include <algorithm>

namespace Gpu
{
    struct SolverContext::Impl
    {
        DeviceCVRP   device;
        BucketLayout layout;
        int          max_bucket_size = 0;

        int*    d_parent            = nullptr;
        double* d_cheapest_w        = nullptr;
        int*    d_cheapest_u        = nullptr;
        int*    d_cheapest_v        = nullptr;
        int*    d_mst_u             = nullptr;
        int*    d_mst_v             = nullptr;
        int*    d_mst_edge_count    = nullptr;
        int*    d_component_count   = nullptr;

        explicit Impl(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha)
            : device(cvrp, alpha)
        {
        }

        void allocate_mst_scratch(int max_k)
        {
            if (max_k <= 1)
            {
                return;
            }

            CUDA_CHECK(cudaMalloc(&d_parent,          max_k * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cheapest_w,      max_k * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_cheapest_u,      max_k * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_cheapest_v,      max_k * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_mst_u,           (max_k - 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_mst_v,           (max_k - 1) * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_mst_edge_count,  sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_component_count, sizeof(int)));
        }

        void free_mst_scratch()
        {
            cudaFree(d_parent);
            cudaFree(d_cheapest_w);
            cudaFree(d_cheapest_u);
            cudaFree(d_cheapest_v);
            cudaFree(d_mst_u);
            cudaFree(d_mst_v);
            cudaFree(d_mst_edge_count);
            cudaFree(d_component_count);

            d_parent          = nullptr;
            d_cheapest_w      = nullptr;
            d_cheapest_u      = nullptr;
            d_cheapest_v      = nullptr;
            d_mst_u           = nullptr;
            d_mst_v           = nullptr;
            d_mst_edge_count  = nullptr;
            d_component_count = nullptr;
        }

        ~Impl()
        {
            free_mst_scratch();
        }
    };

    SolverContext::SolverContext(
        const Bucket_Partitioned_MDS::CVRP& cvrp,
        double                              alpha)
        : impl_(std::make_unique<Impl>(cvrp, alpha))
    {
    }

    SolverContext::~SolverContext() = default;

    int SolverContext::num_buckets() const
    {
        return impl_->device.num_buckets();
    }

    void SolverContext::create_buckets(std::vector<std::vector<node_t>>& buckets)
    {
        const int num_buckets = static_cast<int>(buckets.size());

        for (auto& bucket : buckets)
        {
            bucket.clear();
        }

        impl_->layout = assign_and_compact_buckets(impl_->device);
        impl_->device.upload_bucket_layout(impl_->layout);
        impl_->max_bucket_size = impl_->device.max_bucket_size();

        impl_->free_mst_scratch();
        impl_->allocate_mst_scratch(impl_->max_bucket_size);

        for (int b = 0; b < num_buckets; ++b)
        {
            const int start = impl_->layout.offsets[b];
            const int end   = impl_->layout.offsets[b + 1];
            buckets[b].reserve(end - start);
            for (int i = start; i < end; ++i)
            {
                buckets[b].push_back(impl_->layout.nodes[i]);
            }
        }
    }

    void SolverContext::construct_mst(
        int                               bucket_id,
        std::vector<std::vector<node_t>>& mst_adj)
    {
        construct_mst_boruvka(
            impl_->device,
            impl_->device.device_bucket_nodes(),
            impl_->device.device_bucket_offsets(),
            bucket_id,
            impl_->max_bucket_size,
            impl_->d_parent,
            impl_->d_cheapest_w,
            impl_->d_cheapest_u,
            impl_->d_cheapest_v,
            impl_->d_mst_u,
            impl_->d_mst_v,
            impl_->d_mst_edge_count,
            impl_->d_component_count,
            mst_adj);
    }
}
