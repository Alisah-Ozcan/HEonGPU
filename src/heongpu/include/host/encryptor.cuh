// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef ENCRYPTOR_H
#define ENCRYPTOR_H

#include <curand_kernel.h>
#include <stdio.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "common.cuh"
#include "cuda_runtime.h"
#include "encryption.cuh"
#include "ntt.cuh"
#include "context.cuh"
#include "random.cuh"

#include "publickey.cuh"
#include "ciphertext.cuh"
#include "plaintext.cuh"

namespace heongpu
{
    /**
     * @brief HEEncryptor is responsible for encrypting plaintexts into
     * ciphertexts using public keys in homomorphic encryption schemes.
     *
     * The HEEncryptor class is initialized with encryption parameters and a
     * public key. It provides methods to encrypt plaintext data into
     * ciphertexts for BFV and CKKS schemes. The class supports both synchronous
     * and asynchronous encryption operations.
     */
    class HEEncryptor
    {
      public:
        /**
         * @brief Constructs a new HEEncryptor object with specified parameters
         * and public key.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param public_key Reference to the Publickey object used for
         * encryption.
         */
        __host__ HEEncryptor(Parameters& context, Publickey& public_key);

        /**
         * @brief Encrypts a plaintext into a ciphertext, automatically
         * determining the scheme type.
         *
         * @param ciphertext Ciphertext object where the result of the
         * encryption will be stored.
         * @param plaintext Plaintext object to be encrypted.
         */
        __host__ void
        encrypt(Ciphertext& ciphertext, Plaintext& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (plaintext.size() < n)
                    {
                        throw std::invalid_argument("Invalid plaintext size.");
                    }

                    if (plaintext.depth() != 0)
                    {
                        throw std::invalid_argument(
                            "Invalid plaintext depth must be zero.");
                    }

                    input_storage_manager(
                        plaintext,
                        [&](Plaintext& plaintext_)
                        {
                            output_storage_manager(
                                ciphertext,
                                [&](Ciphertext& ciphertext_)
                                {
                                    encrypt_bfv(ciphertext_, plaintext_,
                                                options.stream_);

                                    ciphertext.scheme_ = scheme_;
                                    ciphertext.ring_size_ = n;
                                    ciphertext.coeff_modulus_count_ = Q_size_;
                                    ciphertext.cipher_size_ = 2;
                                    ciphertext.depth_ = 0;
                                    ciphertext.in_ntt_domain_ = false;
                                    ciphertext.scale_ = 0;
                                    ciphertext.rescale_required_ = false;
                                    ciphertext.relinearization_required_ =
                                        false;
                                },
                                options);
                        },
                        options, false);

                    break;
                case 2: // CKKS
                    if (plaintext.size() < (n * Q_size_))
                    {
                        throw std::invalid_argument("Invalid plaintext size.");
                    }

                    if (plaintext.depth() != 0)
                    {
                        throw std::invalid_argument(
                            "Invalid plaintext depth must be zero.");
                    }

                    input_storage_manager(
                        plaintext,
                        [&](Plaintext& plaintext_)
                        {
                            output_storage_manager(
                                ciphertext,
                                [&](Ciphertext& ciphertext_)
                                {
                                    encrypt_ckks(ciphertext_, plaintext,
                                                 options.stream_);

                                    ciphertext.scheme_ = scheme_;
                                    ciphertext.ring_size_ = n;
                                    ciphertext.coeff_modulus_count_ = Q_size_;
                                    ciphertext.cipher_size_ = 2;
                                    ciphertext.depth_ = 0;
                                    ciphertext.in_ntt_domain_ = true;
                                    ciphertext.scale_ = plaintext.scale_;
                                    ciphertext.rescale_required_ = false;
                                    ciphertext.relinearization_required_ =
                                        false;
                                },
                                options);
                        },
                        options, false);

                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Returns the seed of the encryptor.
         *
         * @return int Seed of the encryptor.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the seed of the encryptor with new seed.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the encryptor(curand).
         *
         * @return int Offset of the encryptor.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the encryptor with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

        HEEncryptor() = default;
        HEEncryptor(const HEEncryptor& copy) = default;
        HEEncryptor(HEEncryptor&& source) = default;
        HEEncryptor& operator=(const HEEncryptor& assign) = default;
        HEEncryptor& operator=(HEEncryptor&& assign) = default;

      private:
        __host__ void encrypt_bfv(Ciphertext& ciphertext, Plaintext& plaintext,
                                  const cudaStream_t stream);

        __host__ void encrypt_ckks(Ciphertext& ciphertext, Plaintext& plaintext,
                                   const cudaStream_t stream);

      private:
        scheme_type scheme_;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        DeviceVector<Data64> public_key_;

        int n;

        int n_power;

        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;
        std::shared_ptr<DeviceVector<Data64>> half_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;

        Modulus64 plain_modulus_;

        // BFV
        Data64 Q_mod_t_;

        Data64 upper_threshold_;

        std::shared_ptr<DeviceVector<Data64>> coeeff_div_plainmod_;
    };

} // namespace heongpu
#endif // ENCRYPTOR_H
