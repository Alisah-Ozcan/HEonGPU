// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/keygenerator.cuh>

namespace heongpu
{
    __host__ HEKeyGenerator<Scheme::TFHE>::HEKeyGenerator(
        HEContext<Scheme::TFHE> context)
    {
        if (!context)
        {
            throw std::invalid_argument("HEContext is not set!");
        }

        context_ = std::move(context);

        std::random_device rd;
        std::mt19937 gen(rd());

        rng_seed_ = gen();
        rng_offset_ = gen();
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
                sk.lwe_key_device_location_.resize(context_->n_);

                tfhe_secretkey_gen_kernel<<<1, 512, 0, options.stream_>>>(
                    sk.lwe_key_device_location_.data(), context_->n_,
                    rng_seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                sk.tlwe_key_device_location_.resize(context_->k_ *
                                                    context_->N_);

                tfhe_secretkey_gen_kernel<<<
                    (context_->k_ * (context_->N_ >> 9)), 512, 0,
                    options.stream_>>>(sk.tlwe_key_device_location_.data(),
                                       context_->n_, rng_seed_);
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
                        int bk_n = context_->n_;
                        int bk_N = context_->N_;
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
                            context_->ntt_table_->data(), context_->prime_,
                            bk_N);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        tfhe_generate_bootkey_kernel<<<bk_n, 512,
                                                       sizeof(Data64) * bk_N,
                                                       options.stream_>>>(
                            tlwe_key_ntt.data(),
                            sk.lwe_key_device_location_.data(),
                            temp_boot_key.data(), context_->ntt_table_->data(),
                            context_->intt_table_->data(), context_->n_inverse_,
                            context_->prime_, bk_N, bk_k, bk_base_bit_size,
                            bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.boot_key_device_location_.resize(total_bootkey_size,
                                                            options.stream_);
                        tfhe_convert_bootkey_ntt_domain_kernel<<<
                            bk_n, 512, sizeof(Data64) * bk_N,
                            options.stream_>>>(
                            bk.boot_key_device_location_.data(),
                            temp_boot_key.data(), context_->ntt_table_->data(),
                            context_->prime_, bk_N, bk_k, bk_length);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        bk.switch_key_variances_ = std::vector<double>(
                            bk_n * (bk_k + 1) * bk_length,
                            context_->bk_stdev_ * context_->bk_stdev_);

                        // Switch Key
                        int ks_n = context_->n_;
                        int ks_N = context_->k_ * context_->N_;
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
                                               total_noise_size,
                                               context_->ks_stdev_);
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
                            total_noise_size,
                            context_->ks_stdev_ * context_->ks_stdev_);
                        bk.storage_type_ = storage_type::DEVICE;
                    },
                    options);
            },
            options, false);

        bk.boot_key_generated_ = true;
    }

} // namespace heongpu
