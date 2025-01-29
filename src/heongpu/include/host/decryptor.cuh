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
#include "random.cuh"
#include "addition.cuh"
#include "switchkey.cuh"
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
     *
     * Additionally, the class includes methods for multiparty computation (MPC)
     * scenarios. These methods enable partial decryption by multiple
     * participants and the fusion of these partial decryptions into a fully
     * decrypted plaintext.
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
        __host__ void
        decrypt(Plaintext& plaintext, Ciphertext& ciphertext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    input_storage_manager(
                        ciphertext,
                        [&](Ciphertext& ciphertext_)
                        {
                            output_storage_manager(
                                plaintext,
                                [&](Plaintext& plaintext_)
                                {
                                    decrypt_bfv(plaintext_, ciphertext_,
                                                options.stream_);

                                    plaintext.plain_size_ = n;
                                    plaintext.scheme_ = scheme_;
                                    plaintext.depth_ = 0;
                                    plaintext.scale_ = 0;
                                    plaintext.in_ntt_domain_ = false;
                                },
                                options);
                        },
                        options, false);
                    break;
                case 2: // CKKS

                    input_storage_manager(
                        ciphertext,
                        [&](Ciphertext& ciphertext_)
                        {
                            output_storage_manager(
                                plaintext,
                                [&](Plaintext& plaintext_)
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
        __host__ int remainder_noise_budget(
            Ciphertext& ciphertext,
            const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    return noise_budget_calculation(ciphertext, options);
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

        /**
         * @brief Performs a partial decryption of a ciphertext using a secret
         * key.
         *
         * This method is used in multiparty decryption scenarios where each
         * party partially decrypts the ciphertext with their own secret key.
         * The resulting partially decrypted ciphertext is stored for later
         * fusion.
         *
         * @param ciphertext The ciphertext to be partially decrypted.
         * @param sk The secret key of the party performing the partial
         * decryption.
         * @param partial_ciphertext The output ciphertext containing the
         * partially decrypted data.
         */
        __host__ void
        multi_party_decrypt_partial(Ciphertext& ciphertext, Secretkey& sk,
                                    Ciphertext& partial_ciphertext,
                                    cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV

                    partial_decrypt_bfv(ciphertext, sk, partial_ciphertext,
                                        stream);

                    partial_ciphertext.scheme_ = scheme_;
                    partial_ciphertext.ring_size_ = n;
                    partial_ciphertext.coeff_modulus_count_ = Q_size_;
                    partial_ciphertext.cipher_size_ = 2;
                    partial_ciphertext.depth_ = ciphertext.depth_;
                    partial_ciphertext.in_ntt_domain_ =
                        ciphertext.in_ntt_domain_;
                    partial_ciphertext.scale_ = ciphertext.scale_;
                    partial_ciphertext.rescale_required_ =
                        ciphertext.rescale_required_;
                    partial_ciphertext.relinearization_required_ =
                        ciphertext.relinearization_required_;

                    break;
                case 2: // CKKS

                    partial_decrypt_ckks(ciphertext, sk, partial_ciphertext,
                                         stream);

                    partial_ciphertext.scheme_ = scheme_;
                    partial_ciphertext.ring_size_ = n;
                    partial_ciphertext.coeff_modulus_count_ = Q_size_;
                    partial_ciphertext.cipher_size_ = 2;
                    partial_ciphertext.depth_ = ciphertext.depth_;
                    partial_ciphertext.in_ntt_domain_ =
                        ciphertext.in_ntt_domain_;
                    partial_ciphertext.scale_ = ciphertext.scale_;
                    partial_ciphertext.rescale_required_ =
                        ciphertext.rescale_required_;
                    partial_ciphertext.relinearization_required_ =
                        ciphertext.relinearization_required_;

                    break;
                case 3: // BGV
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Fuses partially decrypted ciphertexts into a fully decrypted
         * plaintext.
         *
         * In multiparty decryption, each participant generates a partial
         * decryption of the ciphertext. This method combines those partial
         * decryptions to produce the final plaintext output.
         *
         * @param ciphertexts A vector containing partially decrypted
         * ciphertexts from multiple parties.
         * @param plaintext The output plaintext resulting from the fusion of
         * all partial decryptions.
         */
        __host__ void multi_party_decrypt_fusion(
            std::vector<Ciphertext>& ciphertexts, Plaintext& plaintext,
            const ExecutionOptions& options = ExecutionOptions())
        {
            int cipher_count = ciphertexts.size();

            if (cipher_count == 0)
            {
                throw std::invalid_argument("No ciphertext to decrypt!");
            }

            scheme_type scheme_check = ciphertexts[0].scheme_;
            int depth_check = ciphertexts[0].depth_;
            double scale_check = ciphertexts[0].scale_;

            for (int i = 1; i < cipher_count; i++)
            {
                if (scheme_check != ciphertexts[i].scheme_)
                {
                    throw std::invalid_argument(
                        "Ciphertext schemes should be same!");
                }

                if (depth_check != ciphertexts[i].depth_)
                {
                    throw std::invalid_argument(
                        "Ciphertext levels should be same!");
                }

                if (scale_check != ciphertexts[i].scale_)
                {
                    throw std::invalid_argument(
                        "Ciphertext scales should be same!");
                }
            }

            input_vector_storage_manager(
                ciphertexts,
                [&](std::vector<Ciphertext>& ciphertexts_)
                {
                    switch (static_cast<int>(scheme_))
                    {
                        case 1: // BFV
                            output_storage_manager(
                                plaintext,
                                [&](Plaintext& plaintext_)
                                {
                                    decrypt_fusion_bfv(ciphertexts_, plaintext_,
                                                       options.stream_);

                                    plaintext.plain_size_ = n;
                                    plaintext.scheme_ = scheme_;
                                    plaintext.depth_ = 0;
                                    plaintext.scale_ = 0;
                                    plaintext.in_ntt_domain_ = false;
                                },
                                options);
                            break;
                        case 2: // CKKS
                            output_storage_manager(
                                plaintext,
                                [&](Plaintext& plaintext_)
                                {
                                    decrypt_fusion_ckks(ciphertexts_,
                                                        plaintext_,
                                                        options.stream_);

                                    plaintext.plain_size_ = n * Q_size_;
                                    plaintext.scheme_ = scheme_;
                                    plaintext.depth_ = depth_check;
                                    plaintext.scale_ = scale_check;
                                    plaintext.in_ntt_domain_ = true;
                                },
                                options);
                            break;
                        case 3: // BGV
                            break;
                        default:
                            throw std::invalid_argument("Invalid Scheme Type");
                            break;
                    }
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
        __host__ void decrypt_bfv(Plaintext& plaintext, Ciphertext& ciphertext,
                                  const cudaStream_t stream);

        __host__ void decryptx3_bfv(Plaintext& plaintext,
                                    Ciphertext& ciphertext,
                                    const cudaStream_t stream);

        __host__ void decrypt_ckks(Plaintext& plaintext, Ciphertext& ciphertext,
                                   const cudaStream_t stream);

        __host__ int noise_budget_calculation(
            Ciphertext& ciphertext,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void partial_decrypt_bfv(Ciphertext& ciphertext, Secretkey& sk,
                                          Ciphertext& partial_ciphertext,
                                          const cudaStream_t stream);

        __host__ void partial_decrypt_ckks(Ciphertext& ciphertext,
                                           Secretkey& sk,
                                           Ciphertext& partial_ciphertext,
                                           const cudaStream_t stream);

        __host__ void decrypt_fusion_bfv(std::vector<Ciphertext>& ciphertexts,
                                         Plaintext& plaintext,
                                         const cudaStream_t stream);

        __host__ void decrypt_fusion_ckks(std::vector<Ciphertext>& ciphertexts,
                                          Plaintext& plaintext,
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
#endif // DECRYPTOR_H
