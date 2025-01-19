// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <mutex>
#include <memory>
#include <vector>
#include <sys/sysinfo.h>

#include "common.cuh"
#include "nttparameters.cuh"
#include "defines.h"

#include <thrust/host_vector.h>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <rmm/mr/host/pinned_memory_resource.hpp>
#include <rmm/mr/host/host_memory_resource.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/mr/device/statistics_resource_adaptor.hpp>
#include <rmm/mr/device/detail/stream_ordered_memory_resource.hpp>
#include <rmm/mr/device/logging_resource_adaptor.hpp>

// --------------------- //
// Author: Alisah Ozcan
// --------------------- //

class MemoryPool
{
    using DeviceResource = rmm::mr::cuda_memory_resource;
    using DevicePoolResource = rmm::mr::pool_memory_resource<DeviceResource>;
    using DeviceStatsAdaptor =
        rmm::mr::statistics_resource_adaptor<rmm::mr::device_memory_resource>;

    using HostResource = rmm::mr::pinned_memory_resource;
    using HostPoolResource = rmm::mr::pool_memory_resource<HostResource>;
    using HostStatsAdaptor =
        rmm::mr::statistics_resource_adaptor<HostPoolResource>;

  public:
    static MemoryPool& instance();

    void initialize();
    // for device
    void use_memory_pool(bool use);

    // for device
    void* allocate(size_t size, cudaStream_t stream = cudaStreamDefault);
    void deallocate(void* ptr, size_t size,
                    cudaStream_t stream = cudaStreamDefault);

    rmm::mr::device_memory_resource* get_device_resource() const;
    HostStatsAdaptor* get_host_resource() const;

    void print_memory_pool_status() const;
    size_t get_current_device_pool_memory_usage() const;
    size_t get_free_device_pool_memory() const;

    size_t get_current_host_pool_memory_usage() const;
    size_t get_free_host_pool_memory() const;

    ~MemoryPool();

  private:
    MemoryPool();
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void clean_pool();
    size_t get_host_avaliable_memory() const;
    size_t get_decive_avaliable_memory() const;
    size_t roundup_256(size_t size) const;

    static std::shared_ptr<HostResource> host_base_;
    static std::shared_ptr<HostPoolResource> host_pool_;
    static std::shared_ptr<HostStatsAdaptor> host_stats_adaptor_;

    static std::shared_ptr<DeviceResource> device_base_;
    static std::shared_ptr<DevicePoolResource> device_pool_;
    static std::shared_ptr<DeviceStatsAdaptor> device_stats_adaptor_;
    static bool initialized_;
    static std::mutex mutex_;
};

template <typename T> struct rmm_pinned_allocator
{
    using value_type = T;
    using HostResource = rmm::mr::pinned_memory_resource;
    using HostPoolResource = rmm::mr::pool_memory_resource<HostResource>;
    using HostStatsAdaptor =
        rmm::mr::statistics_resource_adaptor<HostPoolResource>;

    rmm_pinned_allocator() : pool_(MemoryPool::instance().get_host_resource())
    {
    }

    T* allocate(std::size_t n)
    {
        return static_cast<T*>(pool_->allocate(n * sizeof(T)));
    }

    void deallocate(T* p, std::size_t n)
    {
        pool_->deallocate(p, n * sizeof(T));
    }

    bool operator==(const rmm_pinned_allocator& other) const
    {
        return pool_ == other.pool_;
    }
    bool operator!=(const rmm_pinned_allocator& other) const
    {
        return !(*this == other);
    }

  private:
    HostStatsAdaptor* pool_;
};

#endif // MEMORY_POOL_H