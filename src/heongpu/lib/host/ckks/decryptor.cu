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

} // namespace heongpu