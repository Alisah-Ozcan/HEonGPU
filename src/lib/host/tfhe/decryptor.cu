// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/decryptor.cuh>

namespace heongpu
{
    __host__
    HEDecryptor<Scheme::TFHE>::HEDecryptor(HEContext<Scheme::TFHE>& context,
                                           Secretkey<Scheme::TFHE>& secret_key)
    {
        n_ = context.n_;

        lwe_key_device_location_ = secret_key.lwe_key_device_location_;
    }

    __host__ void
    HEDecryptor<Scheme::TFHE>::decrypt_lwe(std::vector<bool>& messages,
                                           Ciphertext<Scheme::TFHE>& ciphertext,
                                           const cudaStream_t stream)
    {
        const int THREADS = 512;
        int block_count = ciphertext.shape_;
        size_t smem = (THREADS / 32 + 1) * sizeof(uint32_t);

        DeviceVector<int32_t> messages_gpu(ciphertext.shape_, stream);

        decrypt_lwe_kernel<<<block_count, THREADS, smem>>>(
            lwe_key_device_location_.data(),
            ciphertext.a_device_location_.data(),
            ciphertext.b_device_location_.data(), messages_gpu.data(), n_,
            ciphertext.shape_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        std::vector<int32_t> messages_encoded(ciphertext.shape_);
        cudaMemcpy(messages_encoded.data(), messages_gpu.data(),
                   ciphertext.shape_ * sizeof(int32_t), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        messages.resize(ciphertext.shape_);
        for (int i = 0; i < ciphertext.shape_; i++)
        {
            messages[i] = (messages_encoded[i] > 0);
        }
    }

} // namespace heongpu