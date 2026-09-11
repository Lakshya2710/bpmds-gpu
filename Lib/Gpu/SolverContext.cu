#include "Gpu/SolverContext.h"
#include "Gpu/DeviceData.h"
#include "Gpu/MstKernels.h"
#include "Gpu/RouteKernels.h"

#include <algorithm>

namespace Gpu
{
    struct SolverContext::Impl
    {
        DeviceCVRP          device;
        BucketLayout        layout;
        int                 max_bucket_size = 0;
        int                 num_buckets     = 0;
        double              capacity      = 0.0;
        std::vector<int>    h_bucket_k;

        MstDeviceStorage*        mst_storage         = nullptr;
        MstBucketScratch*        per_bucket_scratch  = nullptr;
        cudaStream_t*            streams             = nullptr;

        RouteTrialStorage route_storage;
        int               route_rho = 0;

        explicit Impl(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha)
            : device(cvrp, alpha)
            , capacity(cvrp.capacity())
        {
        }

        void allocate_streams()
        {
            streams = new cudaStream_t[num_buckets];
            for (int b = 0; b < num_buckets; ++b)
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[b], cudaStreamNonBlocking));
            }
        }

        void free_streams()
        {
            if (streams == nullptr)
            {
                return;
            }
            for (int b = 0; b < num_buckets; ++b)
            {
                cudaStreamDestroy(streams[b]);
            }
            delete[] streams;
            streams = nullptr;
        }

        void allocate_mst_resources()
        {
            mst_storage = new MstDeviceStorage;
            allocate_mst_device_storage(*mst_storage, num_buckets, max_bucket_size);

            per_bucket_scratch = new MstBucketScratch[num_buckets];
            for (int b = 0; b < num_buckets; ++b)
            {
                allocate_mst_bucket_scratch(per_bucket_scratch[b], max_bucket_size);
            }
        }

        void free_mst_resources()
        {
            if (per_bucket_scratch != nullptr)
            {
                for (int b = 0; b < num_buckets; ++b)
                {
                    free_mst_bucket_scratch(per_bucket_scratch[b]);
                }
                delete[] per_bucket_scratch;
                per_bucket_scratch = nullptr;
            }
            if (mst_storage != nullptr)
            {
                free_mst_device_storage(*mst_storage);
                delete mst_storage;
                mst_storage = nullptr;
            }
        }

        ~Impl()
        {
            free_route_trial_storage(route_storage);
            free_mst_resources();
            free_streams();
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
        return impl_->num_buckets;
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
        impl_->num_buckets     = num_buckets;

        impl_->h_bucket_k.resize(num_buckets);
        for (int b = 0; b < num_buckets; ++b)
        {
            impl_->h_bucket_k[b] = impl_->layout.offsets[b + 1] - impl_->layout.offsets[b];
        }

        impl_->free_mst_resources();
        impl_->free_streams();
        impl_->allocate_streams();
        impl_->allocate_mst_resources();

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

    void SolverContext::build_all_msts_streamed()
    {
        build_all_msts_on_streams(
            impl_->device,
            impl_->device.device_bucket_offsets(),
            impl_->h_bucket_k,
            impl_->max_bucket_size,
            impl_->per_bucket_scratch,
            *impl_->mst_storage,
            impl_->streams);
    }

    void SolverContext::run_all_route_trials_streamed(int rho)
    {
        if (rho <= 0)
        {
            return;
        }

        if (impl_->route_rho != rho)
        {
            free_route_trial_storage(impl_->route_storage);
            allocate_route_trial_storage(
                impl_->route_storage,
                impl_->num_buckets,
                rho,
                impl_->max_bucket_size);
            impl_->route_rho = rho;
        }

        run_all_route_trials_on_streams(
            impl_->device,
            impl_->device.device_bucket_offsets(),
            impl_->h_bucket_k,
            *impl_->mst_storage,
            rho,
            impl_->max_bucket_size,
            impl_->capacity,
            impl_->route_storage,
            impl_->streams);
    }

    void SolverContext::fetch_best_routes_for_bucket(
        int                               bucket_id,
        int                               rho,
        std::vector<std::vector<node_t>>& routes,
        double&                           cost)
    {
        fetch_best_routes_from_device(
            bucket_id,
            rho,
            impl_->max_bucket_size,
            impl_->route_storage,
            routes,
            cost);
    }
}
