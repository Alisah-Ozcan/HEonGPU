// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/encryptor.cuh>

namespace heongpu
{
    __host__
    HEEncryptor<Scheme::TFHE>::HEEncryptor(HEContext<Scheme::TFHE>& context,
                                           Secretkey<Scheme::TFHE>& secret_key)
    {
        if (!secret_key.secret_key_generated_)
        {
            throw std::runtime_error("Secretkey was not generated!");
        }

        std::random_device rd;
        rng = std::mt19937(rd());

        n_ = context.n_;
        alpha_min_ = secret_key.lwe_alpha_min;
        alpha_max_ = secret_key.lwe_alpha_max;

        if (secret_key.storage_type_ == storage_type::DEVICE)
        {
            lwe_key_device_location_ = secret_key.lwe_key_device_location_;
        }
        else
        {
            lwe_key_device_location_ =
                DeviceVector<int32_t>(secret_key.lwe_key_host_location_);
        }

        std::uniform_int_distribution<uint64_t> dist64(0, UINT64_MAX);
        cuda_seed = dist64(rng);
        cudaMalloc(&cuda_rng, n_ * sizeof(curandState));

        total_state = 512 * 32;
        initialize_random_states_kernel<<<((total_state + 511) >> 9), 512>>>(
            cuda_rng, cuda_seed, total_state);
    }

    __host__ void HEEncryptor<Scheme::TFHE>::encrypt_lwe_symmetric(
        Ciphertext<Scheme::TFHE>& ciphertext, std::vector<int32_t>& messages,
        const cudaStream_t stream)
    {
        std::normal_distribution<double> dist(0.0, alpha_min_);

        std::vector<int32_t> noise;
        noise.reserve(ciphertext.shape_);
        for (size_t i = 0; i < ciphertext.shape_; i++)
        {
            noise.push_back(double_to_torus32(dist(rng)));
        }

        std::vector<int32_t> b;
        b.reserve(ciphertext.shape_);

        for (size_t i = 0; i < ciphertext.shape_; i++)
        {
            uint32_t sum = uint32_t(messages[i]) + uint32_t(noise[i]);
            b.push_back(static_cast<int32_t>(sum));
        }

        const int THREADS = 512;
        int block_count = (ciphertext.shape_ >= 32)
                              ? 32
                              : ciphertext.shape_; // 32 come from rng states!
        size_t smem = (THREADS / 32 + 1) * sizeof(uint32_t);

        ciphertext.a_device_location_.resize(n_ * ciphertext.shape_);
        ciphertext.b_device_location_ = DeviceVector<int32_t>(b);
        encrypt_lwe_kernel<<<block_count, THREADS, smem>>>(
            cuda_rng, lwe_key_device_location_.data(),
            ciphertext.a_device_location_.data(),
            ciphertext.b_device_location_.data(), n_, ciphertext.shape_,
            total_state);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ciphertext.variances_ =
            std::vector<double>(ciphertext.shape_, alpha_min_ * alpha_min_);
        ciphertext.storage_type_ = storage_type::DEVICE;
    }

    __host__ int32_t
    HEEncryptor<Scheme::TFHE>::encode_to_torus32(uint32_t mu, uint32_t m_size)
    {
        uint64_t interval = ((1ULL << 63) / m_size) * 2;
        uint64_t phase64 = mu * interval;
        return static_cast<int32_t>(phase64 >> 32);
    }

    __host__ inline int32_t
    HEEncryptor<Scheme::TFHE>::double_to_torus32(double input)
    {
        double frac = input - std::trunc(input);
        uint32_t result = static_cast<uint32_t>(
            std::llround(frac * static_cast<double>(1ULL << 32)));
        return static_cast<int32_t>(result);
    }

    __host__ std::vector<int32_t> HEEncryptor<Scheme::TFHE>::double_to_torus32(
        const std::vector<double>& input)
    {
        std::vector<int32_t> result;
        result.reserve(input.size());
        for (double v : input)
        {
            result.push_back(double_to_torus32(v));
        }
        return result;
    }

    __host__ HEEncryptor<Scheme::TFHE>::~HEEncryptor()
    {
        cudaFree(cuda_rng);
    }

} // namespace heongpu