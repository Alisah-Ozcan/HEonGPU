// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ckks/decryptor.cuh"

namespace heongpu
{

    __host__
    HEDecryptor<Scheme::CKKS>::HEDecryptor(HEContext<Scheme::CKKS>& context,
                                           Secretkey<Scheme::CKKS>& secret_key)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        if (secret_key.storage_type_ == storage_type::DEVICE)
        {
            secret_key_ = secret_key.device_locations_;
        }
        else
        {
            secret_key.store_in_device();
            secret_key_ = secret_key.device_locations_;
        }

        n = context.n;
        n_power = context.n_power;

        Q_size_ = context.Q_size;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;
    }

    __host__ void HEDecryptor<Scheme::CKKS>::decrypt_ckks(
        Plaintext<Scheme::CKKS>& plaintext,
        Ciphertext<Scheme::CKKS>& ciphertext, const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - ciphertext.depth_;
        DeviceVector<Data64> output_memory(n * current_decomp_count, stream);

        sk_multiplication_ckks<<<dim3((n >> 8), current_decomp_count, 1), 256,
                                 0, stream>>>(
            ciphertext.data(), output_memory.data(), secret_key_.data(),
            modulus_->data(), n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void HEDecryptor<Scheme::CKKS>::partial_decrypt_ckks(
        Ciphertext<Scheme::CKKS>& ciphertext, Secretkey<Scheme::CKKS>& sk,
        Ciphertext<Scheme::CKKS>& partial_ciphertext, const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - ciphertext.depth_;

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (current_decomp_count << n_power);

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        sk_multiplication<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                            stream>>>(ct1, sk.data(),
                                      output_memory.data() +
                                          (current_decomp_count << n_power),
                                      modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Data64> error_poly(current_decomp_count * n, stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly.data(), modulus_->data(), n_power,
                current_decomp_count, 1, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, current_decomp_count,
                                current_decomp_count);

        // TODO: Optimize it!
        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            output_memory.data() + (current_decomp_count * n),
            error_poly.data(),
            output_memory.data() + (current_decomp_count * n), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<dim3((n >> 8), current_decomp_count, 1),
                                       256, 0, stream>>>(
            ct0, output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        partial_ciphertext.memory_set(std::move(output_memory));
    }

    __host__ void HEDecryptor<Scheme::CKKS>::decrypt_fusion_ckks(
        std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
        Plaintext<Scheme::CKKS>& plaintext, const cudaStream_t stream)
    {
        int cipher_count = ciphertexts.size();
        int current_detph = ciphertexts[0].depth_;
        int current_decomp_count = Q_size_ - current_detph;

        DeviceVector<Data64> output_memory(n * current_decomp_count, stream);

        Data64* ct0 = ciphertexts[0].data();
        Data64* ct1 = ciphertexts[0].data() + (current_decomp_count << n_power);
        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            ct0, ct1, output_memory.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            Data64* ct1_i =
                ciphertexts[i].data() + (current_decomp_count << n_power);

            addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                       stream>>>(ct1_i, output_memory.data(),
                                 output_memory.data(), modulus_->data(),
                                 n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        plaintext.memory_set(std::move(output_memory));
    }

} // namespace heongpu