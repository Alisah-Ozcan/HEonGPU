// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "publickey.cuh"

namespace heongpu
{
    __host__ Publickey::Publickey(Parameters& context)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n
        in_ntt_domain_ = false;
    }

    Data64* Publickey::data()
    {
        return locations_.data();
    }

    void Publickey::device_to_host(std::vector<Data64>& public_key,
                                   cudaStream_t stream)
    {
        if (public_key.size() < (coeff_modulus_count_ * ring_size_))
        {
            public_key.resize(coeff_modulus_count_ * ring_size_);
        }

        cudaMemcpyAsync(public_key.data(), locations_.data(),
                        coeff_modulus_count_ * ring_size_ * sizeof(Data64),
                        cudaMemcpyDeviceToHost, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Publickey::host_to_device(std::vector<Data64>& public_key,
                                   cudaStream_t stream)
    {
        if (!(public_key.size() == (coeff_modulus_count_ * ring_size_)))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        cudaMemcpyAsync(locations_.data(), public_key.data(),
                        coeff_modulus_count_ * ring_size_ * sizeof(Data64),
                        cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Publickey::device_to_host(HostVector<Data64>& public_key,
                                   cudaStream_t stream)
    {
        if (public_key.size() < (coeff_modulus_count_ * ring_size_))
        {
            public_key.resize(coeff_modulus_count_ * ring_size_);
        }

        cudaMemcpyAsync(public_key.data(), locations_.data(),
                        coeff_modulus_count_ * ring_size_ * sizeof(Data64),
                        cudaMemcpyDeviceToHost, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Publickey::host_to_device(HostVector<Data64>& public_key,
                                   cudaStream_t stream)
    {
        if (!(public_key.size() == coeff_modulus_count_ * ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        cudaMemcpyAsync(locations_.data(), public_key.data(),
                        coeff_modulus_count_ * ring_size_ * sizeof(Data64),
                        cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ MultipartyPublickey::MultipartyPublickey(Parameters& context,
                                                      int seed)
        : Publickey(context), seed_(seed)
    {
    }

} // namespace heongpu