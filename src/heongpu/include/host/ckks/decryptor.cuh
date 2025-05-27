// Copyright 2024-2025 Alişah Özcan
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
        __host__ void multi_party_decrypt_partial(
            Ciphertext<Scheme::CKKS>& ciphertext, Secretkey<Scheme::CKKS>& sk,
            Ciphertext<Scheme::CKKS>& partial_ciphertext,
            cudaStream_t stream = cudaStreamDefault)
        {
            partial_decrypt_ckks(ciphertext, sk, partial_ciphertext, stream);

            partial_ciphertext.scheme_ = scheme_;
            partial_ciphertext.ring_size_ = n;
            partial_ciphertext.coeff_modulus_count_ = Q_size_;
            partial_ciphertext.cipher_size_ = 2;
            partial_ciphertext.depth_ = ciphertext.depth_;
            partial_ciphertext.in_ntt_domain_ = ciphertext.in_ntt_domain_;
            partial_ciphertext.scale_ = ciphertext.scale_;
            partial_ciphertext.rescale_required_ = ciphertext.rescale_required_;
            partial_ciphertext.relinearization_required_ =
                ciphertext.relinearization_required_;
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
            std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts,
            Plaintext<Scheme::CKKS>& plaintext,
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
                [&](std::vector<Ciphertext<Scheme::CKKS>>& ciphertexts_)
                {
                    output_storage_manager(
                        plaintext,
                        [&](Plaintext<Scheme::CKKS>& plaintext_)
                        {
                            decrypt_fusion_ckks(ciphertexts_, plaintext_,
                                                options.stream_);

                            plaintext_.plain_size_ = n * Q_size_;
                            plaintext_.scheme_ = scheme_;
                            plaintext_.depth_ = depth_check;
                            plaintext_.scale_ = scale_check;
                            plaintext_.in_ntt_domain_ = true;
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
#endif // HEONGPU_CKKS_DECRYPTOR_H
