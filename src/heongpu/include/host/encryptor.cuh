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
        __host__ void encrypt(Ciphertext& ciphertext, Plaintext& plaintext)
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

                    if (ciphertext.locations_.size() < (2 * n * Q_size_))
                    {
                        ciphertext.resize((2 * n * Q_size_));
                    }

                    encrypt_bfv(ciphertext, plaintext);

                    ciphertext.scheme_ = scheme_;
                    ciphertext.ring_size_ = n;
                    ciphertext.coeff_modulus_count_ = Q_size_;
                    ciphertext.cipher_size_ = 3; // default
                    ciphertext.depth_ = 0;
                    ciphertext.in_ntt_domain_ = false;
                    ciphertext.scale_ = 0;
                    ciphertext.rescale_required_ = false;
                    ciphertext.relinearization_required_ = false;
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

                    if (ciphertext.locations_.size() < (2 * n * Q_size_))
                    {
                        ciphertext.resize((2 * n * Q_size_));
                    }

                    encrypt_ckks(ciphertext, plaintext);

                    ciphertext.scheme_ = scheme_;
                    ciphertext.ring_size_ = n;
                    ciphertext.coeff_modulus_count_ = Q_size_;
                    ciphertext.cipher_size_ = 3; // default
                    ciphertext.depth_ = 0;
                    ciphertext.in_ntt_domain_ = true;
                    ciphertext.scale_ = plaintext.scale_;
                    ciphertext.rescale_required_ = false;
                    ciphertext.relinearization_required_ = false;
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encrypts a plaintext into a ciphertext asynchronously,
         * automatically determining the scheme type.
         *
         * @param ciphertext Ciphertext object where the result of the
         * encryption will be stored.
         * @param plaintext Plaintext object to be encrypted.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void encrypt(Ciphertext& ciphertext, Plaintext& plaintext,
                              HEStream& stream)
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

                    if (ciphertext.locations_.size() < (2 * n * Q_size_))
                    {
                        ciphertext.resize((2 * n * Q_size_), stream);
                    }

                    encrypt_bfv(ciphertext, plaintext, stream);

                    ciphertext.scheme_ = scheme_;
                    ciphertext.ring_size_ = n;
                    ciphertext.coeff_modulus_count_ = Q_size_;
                    ciphertext.cipher_size_ = 3; // default
                    ciphertext.depth_ = 0;
                    ciphertext.in_ntt_domain_ = false;
                    ciphertext.scale_ = 0;
                    ciphertext.rescale_required_ = false;
                    ciphertext.relinearization_required_ = false;
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

                    if (ciphertext.locations_.size() < (2 * n * Q_size_))
                    {
                        ciphertext.resize((2 * n * Q_size_), stream);
                    }

                    encrypt_ckks(ciphertext, plaintext, stream);

                    ciphertext.scheme_ = scheme_;
                    ciphertext.ring_size_ = n;
                    ciphertext.coeff_modulus_count_ = Q_size_;
                    ciphertext.cipher_size_ = 3; // default
                    ciphertext.depth_ = 0;
                    ciphertext.in_ntt_domain_ = true;
                    ciphertext.scale_ = plaintext.scale_;
                    ciphertext.rescale_required_ = false;
                    ciphertext.relinearization_required_ = false;
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

        HEEncryptor() = default;
        HEEncryptor(const HEEncryptor& copy) = default;
        HEEncryptor(HEEncryptor&& source) = default;
        HEEncryptor& operator=(const HEEncryptor& assign) = default;
        HEEncryptor& operator=(HEEncryptor&& assign) = default;

      private:
        __host__ void encrypt_bfv(Ciphertext& ciphertext, Plaintext& plaintext);

        __host__ void encrypt_bfv(Ciphertext& ciphertext, Plaintext& plaintext,
                                  HEStream& stream);

        __host__ void encrypt_ckks(Ciphertext& ciphertext,
                                   Plaintext& plaintext);

        __host__ void encrypt_ckks(Ciphertext& ciphertext, Plaintext& plaintext,
                                   HEStream& stream);

      private:
        scheme_type scheme_;
        int seed_;

        Data* public_key_;

        int n;

        int n_power;

        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus>> modulus_;
        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;
        std::shared_ptr<DeviceVector<Data>> last_q_modinv_;
        std::shared_ptr<DeviceVector<Data>> half_;
        std::shared_ptr<DeviceVector<Data>> half_mod_;

        Modulus plain_modulus_;

        // BFV
        Data Q_mod_t_;

        Data upper_threshold_;

        std::shared_ptr<DeviceVector<Data>> coeeff_div_plainmod_;

        // Temp !!! Check All Stream Need This
        DeviceVector<Data> temp_data;
        Data* temp1_enc;
        Data* temp2_enc;
    };

} // namespace heongpu
#endif // ENCRYPTOR_H
