// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "tfhe/keygenerator.cuh"

namespace heongpu
{
    __host__ HEKeyGenerator<Scheme::TFHE>::HEKeyGenerator(
        HEContext<Scheme::TFHE>& context)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        rng_seed_ = gen();
        rng_offset_ = gen();

        prime_ = context.prime_;
        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;
        n_inverse_ = context.n_inverse_;

        ks_base_bit_ = context.ks_base_bit_;
        ks_length_ = context.ks_length_;

        ks_stdev_ = context.ks_stdev_;
        bk_stdev_ = context.bk_stdev_;
        max_stdev_ = context.max_stdev_;

        n_ = context.n_;

        N_ = context.N_;
        k_ = context.k_;

        bk_l_ = context.bk_l_;
        bk_bg_bit_ = context.bk_bg_bit_;
        bg_ = context.bg_;
        half_bg_ = context.half_bg_;
        mask_mod_ = context.mask_mod_;
        kpl_ = context.kpl_;
        h_ = context.h_;
        offset_ = context.offset_;
    }

    __host__ void HEKeyGenerator<Scheme::TFHE>::generate_secret_key(
        Secretkey<Scheme::TFHE>& sk, const ExecutionOptions& options)
    {
        if (sk.secret_key_generated_)
        {
            throw std::runtime_error("Secretkey is already generated!");
        }

        output_storage_manager(
            sk,
            [&](Secretkey<Scheme::TFHE>& sk_)
            {
                sk.lwe_key_device_location_.resize(n_);

                tfhe_secretkey_gen_kernel<<<1, 512, 0, options.stream_>>>(
                    sk.lwe_key_device_location_.data(), n_, rng_seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                sk.tlwe_key_device_location_.resize(k_ * N_);

                tfhe_secretkey_gen_kernel<<<(k_ * (N_ >> 9)), 512, 0,
                                            options.stream_>>>(
                    sk.tlwe_key_device_location_.data(), n_, rng_seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                sk.storage_type_ = storage_type::DEVICE;
            },
            options);

        sk.secret_key_generated_ = true;
    }

    __host__ void HEKeyGenerator<Scheme::TFHE>::generate_bootstrapping_key(
        Bootstrappingkey<Scheme::TFHE>& bk, Secretkey<Scheme::TFHE>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (bk.boot_key_generated_)
        {
            throw std::logic_error("Bootkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::TFHE>& sk_)
            {
                output_storage_manager(
                    bk,
                    [&](Bootstrappingkey<Scheme::TFHE>& bk_)
                    {
                        // Boot Key
                        int bk_n = n_;
                        int bk_N = N_;
                        int bk_k = bk.bk_k_;
                        int bk_base_bit_size = bk.bk_base_bit_;
                        int bk_length = bk.bk_length_;
                        double bk_stddev = bk.bk_stdev_;

                        Data64 total_bootkey_size =
                            (Data64) bk_n * (Data64) (bk_k + 1) *
                            (Data64) bk_length * (Data64) (bk_k + 1) *
                            (Data64) bk_N;
                        DeviceVector<int32_t> temp_boot_key(total_bootkey_size,
                                                            options.stream_);

                        tfhe_generate_bootkey_random_numbers_kernel<<<
                            bk_n, 512, 0, options.stream_>>>(
                            temp_boot_key.data(), bk_N, bk_k, bk_length, 0,
                            bk_stddev);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> tlwe_key_ntt(bk_N * bk_k,
                                                          options.stream_);
                        tfhe_convert_rlwekey_ntt_domain_kernel<<<
                            bk_k, 512, sizeof(Data64) * bk_N,
                            options.stream_>>>(
                            tlwe_key_ntt.data(),
                            sk.tlwe_key_device_location_.data(),
                            ntt_table_->data(), prime_, bk_N);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        tfhe_generate_bootkey_kernel<<<bk_n, 512,
                                                       sizeof(Data64) * bk_N,
                                                       options.stream_>>>(
                            tlwe_key_ntt.data(),
                            sk.lwe_key_device_location_.data(),
                            temp_boot_key.data(), ntt_table_->data(),
                            intt_table_->data(), n_inverse_, prime_, bk_N, bk_k,
                            bk_base_bit_size, bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.boot_key_device_location_.resize(total_bootkey_size,
                                                            options.stream_);
                        tfhe_convert_bootkey_ntt_domain_kernel<<<
                            bk_n, 512, sizeof(Data64) * bk_N,
                            options.stream_>>>(
                            bk.boot_key_device_location_.data(),
                            temp_boot_key.data(), ntt_table_->data(), prime_,
                            bk_N, bk_k, bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.switch_key_variances_ =
                            std::vector<double>(bk_n * (bk_k + 1) * bk_length,
                                                bk_stdev_ * bk_stdev_);

                        // Switch Key
                        int ks_n = n_;
                        int ks_N = k_ * N_;
                        int ks_base_bit_size = bk.ks_base_bit_;
                        int ks_length = bk.ks_length_;
                        int ks_base = 1 << ks_base_bit_size;

                        Data64 total_noise_size = (Data64) ks_N *
                                                  (Data64) ks_length *
                                                  (Data64) (ks_base - 1);
                        DeviceVector<double> noise(total_noise_size,
                                                   options.stream_);

                        tfhe_generate_noise_kernel<<<
                            ((total_noise_size + 511) >> 9), 512, 0,
                            options.stream_>>>(noise.data(), 0,
                                               total_noise_size, ks_stdev_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        Data64 total_random_number_size =
                            total_noise_size * ks_n;
                        bk.switch_key_device_location_a_.resize(
                            total_random_number_size, options.stream_);
                        tfhe_generate_uniform_random_number_kernel<<<
                            ((total_random_number_size + 511) >> 9), 512, 0,
                            options.stream_>>>(
                            bk.switch_key_device_location_a_.data(), 0,
                            total_random_number_size);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        size_t smem = (512 / 32 + 1) * sizeof(uint32_t);

                        bk.switch_key_device_location_b_.resize(
                            total_noise_size, options.stream_);
                        tfhe_generate_switchkey_kernel<<<ks_N, 512, smem,
                                                         options.stream_>>>(
                            sk.tlwe_key_device_location_.data(),
                            sk.lwe_key_device_location_.data(), noise.data(),
                            bk.switch_key_device_location_a_.data(),
                            bk.switch_key_device_location_b_.data(), ks_n,
                            ks_base_bit_size, ks_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.switch_key_variances_ = std::vector<double>(
                            total_noise_size, ks_stdev_ * ks_stdev_);
                        bk.storage_type_ = storage_type::DEVICE;
                    },
                    options);
            },
            options, false);

        bk.boot_key_generated_ = true;
    }

} // namespace heongpu