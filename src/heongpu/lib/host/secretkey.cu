// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "secretkey.cuh"

namespace heongpu
{
    __host__ Secretkey::Secretkey(Parameters& context)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n
        n_power_ = context.n_power;
        modulus_ = context.modulus_;
        ntt_table_ = context.ntt_table_;

        hamming_weight_ = ring_size_ >> 1; // default
        in_ntt_domain_ = false;
    }

    __host__ Secretkey::Secretkey(Parameters& context, int hamming_weight)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n

        hamming_weight_ = hamming_weight;
        if ((hamming_weight_ <= 0) || (hamming_weight_ > ring_size_))
        {
            throw std::invalid_argument(
                "hamming weight has to be in range 0 to ring size.");
        }

        in_ntt_domain_ = false;
    }

    Data64* Secretkey::data()
    {
        return location_.data();
    }

    void Secretkey::device_to_host(std::vector<int>& secret_key,
                                   cudaStream_t stream)
    {
        if (secret_key.size() < ring_size_)
        {
            secret_key.resize(ring_size_);
        }

        cudaMemcpyAsync(secret_key.data(), secretkey_.data(),
                        ring_size_ * sizeof(int), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Secretkey::host_to_device(std::vector<int>& secret_key,
                                   cudaStream_t stream)
    {
        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        cudaMemcpyAsync(secretkey_.data(), secret_key.data(),
                        ring_size_ * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (location_.size() < coeff_modulus_count_ * ring_size_)
        {
            location_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secretkey_.data(), location_.data(), modulus_->data(), n_power_,
            coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(location_.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, coeff_modulus_count_,
                                coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        in_ntt_domain_ = true;
    }

    void Secretkey::device_to_host(HostVector<int>& secret_key,
                                   cudaStream_t stream)
    {
        if (secret_key.size() < ring_size_)
        {
            secret_key.resize(ring_size_);
        }

        cudaMemcpyAsync(secret_key.data(), secretkey_.data(),
                        ring_size_ * sizeof(int), cudaMemcpyDeviceToHost,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    void Secretkey::host_to_device(HostVector<int>& secret_key,
                                   cudaStream_t stream)
    {
        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        cudaMemcpyAsync(secretkey_.data(), secret_key.data(),
                        ring_size_ * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (location_.size() < coeff_modulus_count_ * ring_size_)
        {
            location_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secretkey_.data(), location_.data(), modulus_->data(), n_power_,
            coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(location_.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, coeff_modulus_count_,
                                coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        in_ntt_domain_ = true;
    }

} // namespace heongpu