// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_ENCRYPTOR_H
#define HEONGPU_TFHE_ENCRYPTOR_H

#include "ntt.cuh"
#include "encryption.cuh"
#include "tfhe/context.cuh"
#include "tfhe/secretkey.cuh"
#include "tfhe/ciphertext.cuh"

namespace heongpu
{
    template <> class HEEncryptor<Scheme::TFHE>
    {
      public:
        /**
         * @brief Constructs a new HEEncryptor object for encrypting messages
         *        under the TFHE scheme.
         *
         * This constructor sets up the encryption context with the specified
         * TFHE parameters and secret key. It enables symmetric encryption of
         * boolean messages into ciphertexts that can later be decrypted by
         * the corresponding decryptor.
         *
         * @param context Reference to the HEContext object containing encryption
         *                parameters for the TFHE scheme.
         * @param secret_key Reference to the Secretkey object used for encryption.
         */
        __host__ HEEncryptor(HEContext<Scheme::TFHE>& context,
                             Secretkey<Scheme::TFHE>& secret_key);
        
         /**
         * @brief Encrypts a vector of boolean messages into a TFHE ciphertext.
         *
         * This function performs symmetric encryption by encoding the input
         * boolean messages into Torus format and encrypting them using the
         * secret key.
         *
         * @param ciphertext Output ciphertext where the encrypted result will be stored.
         * @param messages Vector of boolean values representing the plaintext to encrypt.
         */
        __host__ void
        encrypt(Ciphertext<Scheme::TFHE>& ciphertext,
                const std::vector<bool>& messages,
                const ExecutionOptions& options = ExecutionOptions())
        {
            int32_t one_over_8 = encode_to_torus32(1, 8);

            ciphertext.shape_ = messages.size();

            std::vector<int32_t> encoded_messages;
            encoded_messages.reserve(ciphertext.shape_);
            for (bool bit : messages)
            {
                encoded_messages.push_back(bit ? one_over_8 : -one_over_8);
            }

            output_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::TFHE>& ciphertext_)
                {
                    encrypt_lwe_symmetric(ciphertext, encoded_messages,
                                          options.stream_);

                    ciphertext.n_ = n_;
                    ciphertext.alpha_min_ = alpha_min_;
                    ciphertext.alpha_max_ = alpha_max_;
                    ciphertext.ciphertext_generated_ = true;
                },
                options);
        }

        __host__ ~HEEncryptor();

      private:
        __host__ void
        encrypt_lwe_symmetric(Ciphertext<Scheme::TFHE>& ciphertext,
                              std::vector<int32_t>& messages,
                              const cudaStream_t stream);

        __host__ int32_t encode_to_torus32(uint32_t mu, uint32_t m_size);

        __host__ inline int32_t double_to_torus32(double input);

        __host__ std::vector<int32_t>
        double_to_torus32(const std::vector<double>& input);

      private:
        const scheme_type scheme_ = scheme_type::tfhe;

        std::mt19937 rng;

        Data64 cuda_seed;
        int total_state;
        curandState* cuda_rng;

        int n_;
        double alpha_min_;
        double alpha_max_;

        DeviceVector<int32_t> lwe_key_device_location_;
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_ENCRYPTOR_H
