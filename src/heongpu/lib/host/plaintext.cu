// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "plaintext.cuh"

namespace heongpu
{
    __host__ Plaintext::Plaintext() {}

    __host__ Plaintext::Plaintext(Parameters& context, cudaStream_t stream)
    {
        scheme_ = context.scheme_;
        switch (static_cast<int>(context.scheme_))
        {
            case 1: // BFV
                plain_size_ = context.n;
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = false;
                break;
            case 2: // CKKS
                plain_size_ = context.n * context.Q_size; // n
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = true;
                break;
            default:
                break;
        }

        locations_ = DeviceVector<Data>(plain_size_, stream);
    }

    __host__ Plaintext::Plaintext(const std::vector<Data>& plain,
                                  Parameters& context, cudaStream_t stream)
    {
        scheme_ = context.scheme_;
        switch (static_cast<int>(context.scheme_))
        {
            case 1: // BFV
                plain_size_ = context.n;
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = false;

                if (!(plain.size() == plain_size_))
                {
                    throw std::invalid_argument(
                        "Plaintext size should be valid!");
                }

                break;
            case 2: // CKKS
                plain_size_ = context.n * context.Q_size; // n
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = true;

                if (!(plain.size() == plain_size_))
                {
                    throw std::invalid_argument(
                        "Plaintext size should be valid!");
                }

                break;
            default:
                break;
        }

        locations_ = DeviceVector<Data>(plain, stream);
    }

    __host__ Plaintext::Plaintext(const HostVector<Data>& plain,
                                  Parameters& context, cudaStream_t stream)
    {
        scheme_ = context.scheme_;
        switch (static_cast<int>(context.scheme_))
        {
            case 1: // BFV
                plain_size_ = context.n;
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = false;

                if (!(plain.size() == plain_size_))
                {
                    throw std::invalid_argument(
                        "Plaintext size should be valid!");
                }

                break;
            case 2: // CKKS
                plain_size_ = context.n * context.Q_size; // n
                depth_ = 0;
                scale_ = 0;
                in_ntt_domain_ = true;

                if (!(plain.size() == plain_size_))
                {
                    throw std::invalid_argument(
                        "Plaintext size should be valid!");
                }

                break;
            default:
                break;
        }

        locations_ = DeviceVector<Data>(plain, stream);
    }

    Data* Plaintext::data()
    {
        return locations_.data();
    }

    void Plaintext::device_to_host(std::vector<Data>& plain,
                                   cudaStream_t stream)
    {
        if (plain.size() < plain_size_)
        {
            plain.resize(plain_size_);
        }

        cudaMemcpyAsync(plain.data(), locations_.data(),
                        plain_size_ * sizeof(Data), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Plaintext::host_to_device(std::vector<Data>& plain,
                                   cudaStream_t stream)
    {
        if (!(plain.size() == plain_size_))
        {
            throw std::invalid_argument("Plaintext size should be valid!");
        }

        cudaMemcpyAsync(locations_.data(), plain.data(),
                        plain_size_ * sizeof(Data), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Plaintext::device_to_host(HostVector<Data>& plain, cudaStream_t stream)
    {
        if (plain.size() < plain_size_)
        {
            plain.resize(plain_size_);
        }

        cudaMemcpyAsync(plain.data(), locations_.data(),
                        plain_size_ * sizeof(Data), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Plaintext::host_to_device(HostVector<Data>& plain, cudaStream_t stream)
    {
        if (!(plain.size() == plain_size_))
        {
            throw std::invalid_argument("Plaintext size should be valid!");
        }

        cudaMemcpyAsync(locations_.data(), plain.data(),
                        plain_size_ * sizeof(Data), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu