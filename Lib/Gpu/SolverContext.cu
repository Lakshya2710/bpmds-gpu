#include "Gpu/SolverContext.h"
#include "Gpu/DeviceData.h"
#include "Gpu/MstKernels.h"
#include "Gpu/RouteKernels.h"

#include <vector>

namespace Gpu
{
    struct SolverContext::Impl
    {
        DeviceCVRP          device;
        int                 max_bucket_size = 0;
        int                 num_buckets     = 0;
        double              capacity        = 0.0;
        std::vector<int>    h_offsets;

        MstDeviceStorage             mst_storage;
        std::vector<MstBucketScratch> per_bucket_scratch;
        std::vector<cudaStream_t>     streams;

        RouteTrialStorage route_storage;
        int               route_rho = 0;

        explicit Impl(const Bucket_Partitioned_MDS::CVRP& cvrp, double alpha)
            : device(cvrp, alpha)
            , capacity(cvrp.capacity())
        {
        }

        void allocate_streams()
        {
            streams.resize(num_buckets);
            for (int b = 0; b < num_buckets; ++b)
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&streams[b], cudaStreamNonBlocking));
            }
        }

        void free_streams()
        {
            for (cudaStream_t stream : streams)
            {
                cudaStreamDestroy(stream);
            }
            streams.clear();
        }

        void allocate_mst_resources()
        {
            allocate_mst_device_storage(mst_storage, num_buckets, max_bucket_size);

            per_bucket_scratch.assign(num_buckets, MstBucketScratch{});
            for (int b = 0; b < num_buckets; ++b)
            {
                allocate_mst_bucket_scratch(per_bucket_scratch[b], max_bucket_size);
            }
        }

        void free_mst_resources()
        {
            for (MstBucketScratch& scratch : per_bucket_scratch)
            {
                free_mst_bucket_scratch(scratch);
            }
            per_bucket_scratch.clear();
            free_mst_device_storage(mst_storage);
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

    void SolverContext::partition()
    {
        impl_->h_offsets       = compact_buckets(impl_->device);
        impl_->max_bucket_size = impl_->device.max_bucket_size();
        impl_->num_buckets     = impl_->device.num_buckets();

        impl_->free_mst_resources();
        impl_->free_streams();
        impl_->allocate_streams();
        impl_->allocate_mst_resources();
    }

    void SolverContext::build_all_msts_streamed()
    {
        build_all_msts_on_streams(
            impl_->device,
            impl_->h_offsets,
            impl_->max_bucket_size,
            impl_->per_bucket_scratch.data(),
            impl_->mst_storage,
            impl_->streams.data());
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
            impl_->h_offsets,
            impl_->mst_storage,
            rho,
            impl_->max_bucket_size,
            impl_->capacity,
            impl_->route_storage,
            impl_->streams.data());
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
