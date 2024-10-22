// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ciphertext.cuh"

namespace heongpu
{
    __host__ Ciphertext::Ciphertext() {}

    __host__ Ciphertext::Ciphertext(Parameters& context, double scale,
                                    int depth)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 3; // default
        ring_size_ = context.n; // n
        depth_ = depth;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = scale;

        locations_ = DeviceVector<Data>(3 * (coeff_modulus_count_ - depth_) *
                                        ring_size_);
    }

    __host__ Ciphertext::Ciphertext(Parameters& context, HEStream& stream,
                                    double scale, int depth)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 3; // default
        ring_size_ = context.n; // n
        depth_ = depth;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = scale;

        locations_ = DeviceVector<Data>(
            3 * (coeff_modulus_count_ - depth_) * ring_size_, stream.stream);
    }

    //

    __host__ Ciphertext::Ciphertext(const std::vector<Data>& cipher,
                                    Parameters& context, double scale,
                                    int depth)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 3; // default
        ring_size_ = context.n; // n
        depth_ = depth;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = scale;

        if (!(cipher.size() ==
              (2 * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        locations_ = DeviceVector<Data>(3 * (coeff_modulus_count_ - depth_) *
                                        ring_size_);

        cudaMemcpy(locations_.data(), cipher.data(),
                   (2 * (coeff_modulus_count_ - depth_) * ring_size_) *
                       sizeof(Data),
                   cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ Ciphertext::Ciphertext(const std::vector<Data>& cipher,
                                    Parameters& context, HEStream& stream,
                                    double scale, int depth)
    {
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 3; // default
        ring_size_ = context.n; // n
        depth_ = depth;
        scheme_ = context.scheme_;
        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = scale;

        if (!(cipher.size() ==
              (2 * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        locations_ = DeviceVector<Data>(
            3 * (coeff_modulus_count_ - depth_) * ring_size_, stream.stream);

        cudaMemcpyAsync(locations_.data(), cipher.data(),
                        (2 * (coeff_modulus_count_ - depth_) * ring_size_) *
                            sizeof(Data),
                        cudaMemcpyHostToDevice, stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    Data* Ciphertext::data()
    {
        return locations_.data();
    }

    void Ciphertext::device_to_host(std::vector<Data>& cipher)
    {
        if (!(cipher.size() ==
              (2 * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        cudaMemcpy(cipher.data(), locations_.data(),
                   (2 * (coeff_modulus_count_ - depth_) * ring_size_) *
                       sizeof(Data),
                   cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Ciphertext::device_to_host(std::vector<Data>& cipher, HEStream& stream)
    {
        if (!(cipher.size() ==
              (2 * (coeff_modulus_count_ - depth_) * ring_size_)))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        cudaMemcpyAsync(cipher.data(), locations_.data(),
                        (2 * (coeff_modulus_count_ - depth_) * ring_size_) *
                            sizeof(Data),
                        cudaMemcpyDeviceToHost, stream.stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Ciphertext::switch_stream(cudaStream_t stream)
    {
        locations_.set_stream(stream);
    }

} // namespace heongpu