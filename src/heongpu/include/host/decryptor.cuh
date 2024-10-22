// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef DECRYPTOR_H
#define DECRYPTOR_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "decryption.cuh"
#include "ntt.cuh"
#include "context.cuh"
#include "secretkey.cuh"
#include "ciphertext.cuh"
#include "plaintext.cuh"

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
     */
    class HEDecryptor
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
        __host__ HEDecryptor(Parameters& context, Secretkey& secret_key);

        /**
         * @brief Decrypts a ciphertext into a plaintext, automatically
         * determining the scheme type.
         *
         * @param plaintext Plaintext object where the result of the decryption
         * will be stored.
         * @param ciphertext Ciphertext object to be decrypted.
         */
        __host__ void decrypt(Plaintext& plaintext, Ciphertext& ciphertext)
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (plaintext.size() != n)
                    {
                        plaintext.resize(n);
                    }

                    decrypt_bfv(plaintext, ciphertext);

                    plaintext.scheme_ = scheme_;
                    plaintext.depth_ = 0;
                    plaintext.scale_ = 0;
                    plaintext.in_ntt_domain_ = false;
                    break;
                case 2: // CKKS
                    if (plaintext.size() != (n * (Q_size_ - ciphertext.depth_)))
                    {
                        plaintext.resize((n * (Q_size_ - ciphertext.depth_)));
                    }

                    decrypt_ckks(plaintext, ciphertext);

                    plaintext.scheme_ = scheme_;
                    plaintext.depth_ = ciphertext.depth_;
                    plaintext.scale_ = ciphertext.scale_;
                    plaintext.in_ntt_domain_ = true;
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Calculates the remainder of the noise budget in a ciphertext.
         *
         * @param ciphertext Ciphertext object for which the remaining noise
         * budget is calculated.
         * @return int The remainder of the noise budget in the ciphertext.
         */
        __host__ int remainder_noise_budget(Ciphertext& ciphertext)
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    return noise_budget_calculation(ciphertext);
                case 2: // CKKS
                    throw std::invalid_argument(
                        "Can not be used for CKKS Scheme");
                case 3: // BGV
                    break;
                default:
                    break;
            }

            throw std::invalid_argument("Invalid Scheme Type");
        }

        HEDecryptor() = default;
        HEDecryptor(const HEDecryptor& copy) = default;
        HEDecryptor(HEDecryptor&& source) = default;
        HEDecryptor& operator=(const HEDecryptor& assign) = default;
        HEDecryptor& operator=(HEDecryptor&& assign) = default;

      private:
        __host__ void decrypt_bfv(Plaintext& plaintext, Ciphertext& ciphertext);

        __host__ void decryptx3_bfv(Plaintext& plaintext,
                                    Ciphertext& ciphertext);

        __host__ void decrypt_ckks(Plaintext& plaintext,
                                   Ciphertext& ciphertext);

        __host__ int noise_budget_calculation(Ciphertext& ciphertext);

      private:
        scheme_type scheme_;

        Data* secret_key_;

        int n;

        int n_power;

        int Q_size_;

        std::shared_ptr<DeviceVector<Modulus>> modulus_;

        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;

        // BFV
        Modulus plain_modulus_;

        Modulus gamma_;

        std::shared_ptr<DeviceVector<Data>> Qi_t_;

        std::shared_ptr<DeviceVector<Data>> Qi_gamma_;

        std::shared_ptr<DeviceVector<Data>> Qi_inverse_;

        Data mulq_inv_t_;

        Data mulq_inv_gamma_;

        Data inv_gamma_;

        // Noise Budget Calculation

        std::shared_ptr<DeviceVector<Data>> Mi_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data>> decryption_modulus_;

        int total_bit_count_;

        DeviceVector<Data> temp_memory_; // for noise budget calculation
        std::vector<Data> max_norm_memory_;

        DeviceVector<Data> temp_memory2_; // Decryption
    };

} // namespace heongpu
#endif // DECRYPTOR_H
