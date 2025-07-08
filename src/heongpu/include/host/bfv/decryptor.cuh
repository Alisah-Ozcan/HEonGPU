﻿// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_DECRYPTOR_H
#define HEONGPU_BFV_DECRYPTOR_H

#include "ntt.cuh"
#include "addition.cuh"
#include "decryption.cuh"
#include "switchkey.cuh"
#include "bfv/context.cuh"
#include "bfv/secretkey.cuh"
#include "bfv/plaintext.cuh"
#include "bfv/ciphertext.cuh"

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
    template <> class HEDecryptor<Scheme::BFV>
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
        __host__ HEDecryptor(HEContext<Scheme::BFV>& context,
                             Secretkey<Scheme::BFV>& secret_key);

        /**
         * @brief Decrypts a ciphertext into a plaintext, automatically
         * determining the scheme type.
         *
         * @param plaintext Plaintext object where the result of the decryption
         * will be stored.
         * @param ciphertext Ciphertext object to be decrypted.
         */
        __host__ void
        decrypt(Plaintext<Scheme::BFV>& plaintext,
                Ciphertext<Scheme::BFV>& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                ciphertext,
                [&](Ciphertext<Scheme::BFV>& ciphertext_)
                {
                    output_storage_manager(
                        plaintext,
                        [&](Plaintext<Scheme::BFV>& plaintext_)
                        {
                            decrypt_bfv(plaintext_, ciphertext_,
                                        options.stream_);

                            plaintext.plain_size_ = n;
                            plaintext.scheme_ = scheme_;
                            plaintext.in_ntt_domain_ = false;
                        },
                        options);
                },
                options, false);
        }

        /**
         * @brief Calculates the remainder of the noise budget in a ciphertext.
         *
         * @param ciphertext Ciphertext object for which the remaining noise
         * budget is calculated.
         * @return int The remainder of the noise budget in the ciphertext.
         */
        __host__ int remainder_noise_budget(
            Ciphertext<Scheme::BFV>& ciphertext,
            const ExecutionOptions& options = ExecutionOptions())
        {
            return noise_budget_calculation(ciphertext, options);
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
        __host__ void decrypt_bfv(Plaintext<Scheme::BFV>& plaintext,
                                  Ciphertext<Scheme::BFV>& ciphertext,
                                  const cudaStream_t stream);

        __host__ void decryptx3_bfv(Plaintext<Scheme::BFV>& plaintext,
                                    Ciphertext<Scheme::BFV>& ciphertext,
                                    const cudaStream_t stream);

        __host__ int noise_budget_calculation(
            Ciphertext<Scheme::BFV>& ciphertext,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        partial_decrypt_bfv(Ciphertext<Scheme::BFV>& ciphertext,
                            Secretkey<Scheme::BFV>& sk,
                            Ciphertext<Scheme::BFV>& partial_ciphertext,
                            const cudaStream_t stream);

        __host__ void
        decrypt_fusion_bfv(std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
                           Plaintext<Scheme::BFV>& plaintext,
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

        // BFV
        Modulus64 plain_modulus_;

        Modulus64 gamma_;

        std::shared_ptr<DeviceVector<Data64>> Qi_t_;

        std::shared_ptr<DeviceVector<Data64>> Qi_gamma_;

        std::shared_ptr<DeviceVector<Data64>> Qi_inverse_;

        Data64 mulq_inv_t_;

        Data64 mulq_inv_gamma_;

        Data64 inv_gamma_;

        // Noise Budget Calculation

        std::shared_ptr<DeviceVector<Data64>> Mi_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data64>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data64>> decryption_modulus_;

        int total_bit_count_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_DECRYPTOR_H
