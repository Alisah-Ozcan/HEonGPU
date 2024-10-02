// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "memorypool.cuh"
#include <iostream>
#include <stdexcept>

std::shared_ptr<MemoryPool::HostResource> MemoryPool::host_base_ = nullptr;
std::shared_ptr<MemoryPool::HostPoolResource> MemoryPool::host_pool_ = nullptr;
std::shared_ptr<MemoryPool::HostStatsAdaptor> MemoryPool::host_stats_adaptor_ =
    nullptr;

std::shared_ptr<MemoryPool::DeviceResource> MemoryPool::device_base_ = nullptr;
std::shared_ptr<MemoryPool::DevicePoolResource> MemoryPool::device_pool_ =
    nullptr;
std::shared_ptr<MemoryPool::DeviceStatsAdaptor>
    MemoryPool::device_stats_adaptor_ = nullptr;
bool MemoryPool::initialized_ = false;
std::mutex MemoryPool::mutex_;

MemoryPool& MemoryPool::instance()
{
    static MemoryPool instance;
    return instance;
}

size_t MemoryPool::get_host_avaliable_memory() const
{
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    size_t free_memory = memInfo.freeram * memInfo.mem_unit;
    return free_memory;
}

size_t MemoryPool::get_decive_avaliable_memory() const
{
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return free_mem; // total_mem
}

size_t MemoryPool::roundup_256(size_t size) const
{
    return ((size + 255) / 256) * 256;
}

void MemoryPool::initialize()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (!initialized_)
    {
        size_t total_host_memory = get_host_avaliable_memory();
        // size_t initial_host_pool_size =
        // roundup_256(static_cast<size_t>(total_host_memory *
        // initial_host_memorypool_size));
        size_t initial_host_pool_size =
            roundup_256(static_cast<size_t>(104857600));
        size_t max_host_pool_size = roundup_256(
            static_cast<size_t>(total_host_memory * max_host_memorypool_size));

        host_base_ = std::make_shared<HostResource>();
        host_pool_ = std::make_shared<HostPoolResource>(
            host_base_.get(), initial_host_pool_size, max_host_pool_size);
        host_stats_adaptor_ =
            std::make_shared<HostStatsAdaptor>(host_pool_.get());

        size_t total_device_memory = get_decive_avaliable_memory();
        size_t initial_device_pool_size = roundup_256(static_cast<size_t>(
            total_device_memory * initial_device_memorypool_size));
        size_t max_device_pool_size = roundup_256(static_cast<size_t>(
            total_device_memory * max_device_memorypool_size));

        device_base_ = std::make_shared<DeviceResource>();
        device_pool_ = std::make_shared<DevicePoolResource>(
            device_base_.get(), initial_device_pool_size, max_device_pool_size);
        device_stats_adaptor_ =
            std::make_shared<DeviceStatsAdaptor>(device_pool_.get());

        initialized_ = true;
    }
}

void MemoryPool::use_memory_pool(bool use)
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (use)
    {
        rmm::mr::set_current_device_resource(device_stats_adaptor_.get());
    }
    else
    {
        rmm::mr::set_current_device_resource(device_base_.get());
    }
}

void* MemoryPool::allocate(size_t size, cudaStream_t stream)
{
    std::lock_guard<std::mutex> guard(mutex_);
    return rmm::mr::get_current_device_resource()->allocate(size, stream);
}

void MemoryPool::deallocate(void* ptr, size_t size, cudaStream_t stream)
{
    std::lock_guard<std::mutex> guard(mutex_);
    rmm::mr::get_current_device_resource()->deallocate(ptr, size, stream);
}

rmm::mr::device_memory_resource* MemoryPool::get_device_resource() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    return device_stats_adaptor_.get();
}

rmm::mr::statistics_resource_adaptor<
    rmm::mr::pool_memory_resource<rmm::mr::pinned_memory_resource>>*
MemoryPool::get_host_resource() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    return host_stats_adaptor_.get();
}

void MemoryPool::print_memory_pool_status() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (device_stats_adaptor_ && device_pool_ && host_stats_adaptor_ &&
        host_pool_)
    {
        auto device_status = device_stats_adaptor_->get_bytes_counter();
        std::cout << "Device Memory Pool Statistics:" << std::endl;
        std::cout << "Total device pool size: " << device_pool_->pool_size()
                  << " bytes" << std::endl;
        std::cout << "Current device pool usage: " << device_status.value
                  << " bytes" << std::endl;
        std::cout << "Available device pool size: "
                  << device_pool_->pool_size() - device_status.value << " bytes"
                  << std::endl
                  << std::endl;

        auto host_status = host_stats_adaptor_->get_bytes_counter();
        std::cout << "host Memory Pool Statistics:" << std::endl;
        std::cout << "Total host pool size: " << host_pool_->pool_size()
                  << " bytes" << std::endl;
        std::cout << "Current host pool usage: " << host_status.value
                  << " bytes" << std::endl;
        std::cout << "Available host pool size: "
                  << host_pool_->pool_size() - host_status.value << " bytes"
                  << std::endl;
    }
    else
    {
        std::cout
            << "Unable to retrieve statistics from statistics_resource_adaptor"
            << std::endl;
    }
}

size_t MemoryPool::get_current_device_pool_memory_usage() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto device_status = device_stats_adaptor_->get_bytes_counter();
    return device_status.value;
}

size_t MemoryPool::get_free_device_pool_memory() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto device_status = device_stats_adaptor_->get_bytes_counter();
    return device_pool_->pool_size() - device_status.value;
}

size_t MemoryPool::get_current_host_pool_memory_usage() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto host_status = host_stats_adaptor_->get_bytes_counter();
    return host_status.value;
}

size_t MemoryPool::get_free_host_pool_memory() const
{
    std::lock_guard<std::mutex> guard(mutex_);
    auto host_status = host_stats_adaptor_->get_bytes_counter();
    return host_pool_->pool_size() - host_status.value;
}

MemoryPool::~MemoryPool()
{
    clean_pool();
}

MemoryPool::MemoryPool() = default;

void MemoryPool::clean_pool()
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (initialized_)
    {
        rmm::mr::set_current_device_resource(nullptr);
        host_stats_adaptor_.reset();
        host_pool_.reset();
        host_base_.reset();
        device_stats_adaptor_.reset();
        device_pool_.reset();
        device_base_.reset();
        initialized_ = false;
    }
}
