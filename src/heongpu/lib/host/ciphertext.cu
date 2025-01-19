// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ciphertext.cuh"

namespace heongpu
{
    __host__ Ciphertext::Ciphertext() {}

    __host__ Ciphertext::Ciphertext(Parameters& context, cudaStream_t stream)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2; // default
        ring_size_ = context.n; // n
        depth_ = 0;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        locations_ = DeviceVector<Data>(
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_,
            stream);
    }

    __host__ Ciphertext::Ciphertext(const std::vector<Data>& cipher,
                                    Parameters& context, cudaStream_t stream)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2; // default
        ring_size_ = context.n; // n
        depth_ = 0;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        if (!(cipher.size() ==
              (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        locations_ = DeviceVector<Data>(
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_,
            stream);

        cudaMemcpyAsync(
            locations_.data(), cipher.data(),
            (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_) *
                sizeof(Data),
            cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ Ciphertext::Ciphertext(const HostVector<Data>& cipher,
                                    Parameters& context, cudaStream_t stream)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2; // default
        ring_size_ = context.n; // n
        depth_ = 0;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        if (!(cipher.size() ==
              (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        locations_ = DeviceVector<Data>(
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_,
            stream);

        cudaMemcpyAsync(
            locations_.data(), cipher.data(),
            (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_) *
                sizeof(Data),
            cudaMemcpyHostToDevice, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    Data* Ciphertext::data()
    {
        return locations_.data();
    }

    void Ciphertext::device_to_host(std::vector<Data>& cipher,
                                    cudaStream_t stream)
    {
        if (!(cipher.size() ==
              (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        cudaMemcpyAsync(
            cipher.data(), locations_.data(),
            (cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_) *
                sizeof(Data),
            cudaMemcpyDeviceToHost, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu