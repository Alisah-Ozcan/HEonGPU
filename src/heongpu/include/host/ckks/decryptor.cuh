﻿// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_DECRYPTOR_H
#define HEONGPU_CKKS_DECRYPTOR_H

#include "ntt.cuh"
#include "addition.cuh"
#include "decryption.cuh"
#include "switchkey.cuh"
#include "ckks/context.cuh"
#include "ckks/secretkey.cuh"
#include "ckks/plaintext.cuh"
#include "ckks/ciphertext.cuh"

namespace heongpu
{

    /**
     * @brief HEDecryptor is responsible for decrypting ciphertexts back into
     * plaintexts using a secret key in homomorphic encryption schemes.
     *
     * The HEDecryptor class is initialized with encryption parameters and a
     * secret key. It provides methods to decrypt ciphertexts into plaintexts
     * for BFV and CKKS schemes. The class also supports noise budget
     * calculation, which is essential for understanding the remaining "noise"
     * tolerance in a given ciphertext.
     *
     * Additionally, the class includes methods for multiparty computation (MPC)
     * scenarios. These methods enable partial decryption by multiple
     * participants and the fusion of these partial decryptions into a fully
     * decrypted plaintext.
     */
    template <> class HEDecryptor<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEDecryptor object with specified parameters
         * and secret key.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param secret_key Reference to the Secretkey object used for
         * decryption.
         */
        __host__ HEDecryptor(HEContext<Scheme::CKKS>& context,
                             Secretkey<Scheme::CKKS>& secret_key);

        /**
         * @brief Decrypts a ciphertext into a plaintext, automatically
         * determining the scheme type.
         *
         * @param plaintext Plaintext object where the result of the decryption
         * will be stored.
         * @param ciphertext Ciphertext object to be decrypted.
         */
        __host__ void
        decrypt(Plaintext<Scheme::CKKS>& plaintext,
                Ciphertext<Scheme::CKKS>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::CKKS>& ciphertext_)
                {
                    output_storage_manager(
                        plaintext,
                        [&](Plaintext<Scheme::CKKS>& plaintext_)
                        {
                            decrypt_ckks(plaintext_, ciphertext,
                                         options.stream_);

                            plaintext.plain_size_ = n * Q_size_;
                            plaintext.scheme_ = scheme_;
                            plaintext.depth_ = ciphertext.depth_;
                            plaintext.scale_ = ciphertext.scale_;
                            plaintext.in_ntt_domain_ = true;
                        },
                        options);
                },
                options, false);
        }

        /**
         * @brief Returns the seed of the decryptor.
         *
         * @return int Seed of the decryptor.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the seed of the decryptor with new seed.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the decryptor(curand).
         *
         * @return int Offset of the decryptor.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the decryptor with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

        HEDecryptor() = default;
        HEDecryptor(const HEDecryptor& copy) = default;
        HEDecryptor(HEDecryptor&& source) = default;
        HEDecryptor& operator=(const HEDecryptor& assign) = default;
        HEDecryptor& operator=(HEDecryptor&& assign) = default;

      private:
        __host__ void decrypt_ckks(Plaintext<Scheme::CKKS>& plaintext,
                                   Ciphertext<Scheme::CKKS>& ciphertext,
                                   const cudaStream_t stream);

        __host__ void
        partial_decrypt_ckks(Ciphertext<Scheme::CKKS>& ciphertext,
                             Secretkey<Scheme::CKKS>& sk,
                             Ciphertext<Scheme::CKKS>& partial_ciphertext,
                             const cudaStream_t stream);

        __host__ void
        decrypt_fusion_ckks(std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
                            Plaintext<Scheme::CKKS>& plaintext,
                            const cudaStream_t stream);

      private:
        scheme_type scheme_;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        DeviceVector<Data64> secret_key_;

        int n;

        int n_power;

        int Q_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;

        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_DECRYPTOR_H
