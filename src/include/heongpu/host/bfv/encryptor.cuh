// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_ENCRYPTOR_H
#define HEONGPU_BFV_ENCRYPTOR_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/kernel/encryption.cuh>
#include <heongpu/host/bfv/context.cuh>
#include <heongpu/host/bfv/publickey.cuh>
#include <heongpu/host/bfv/plaintext.cuh>
#include <heongpu/host/bfv/ciphertext.cuh>

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
    template <> class HEEncryptor<Scheme::BFV>
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
        __host__ HEEncryptor(HEContext<Scheme::BFV>& context,
                             Publickey<Scheme::BFV>& public_key);

        /**
         * @brief Encrypts a plaintext into a ciphertext, automatically
         * determining the scheme type.
         *
         * @param ciphertext Ciphertext object where the result of the
         * encryption will be stored.
         * @param plaintext Plaintext object to be encrypted.
         */
        __host__ void
        encrypt(Ciphertext<Scheme::BFV>& ciphertext,
                Plaintext<Scheme::BFV>& plaintext,
                const ExecutionOptions& options = ExecutionOptions())
        {
            if (plaintext.size() < n)
            {
                throw std::invalid_argument("Invalid plaintext size.");
            }

            input_storage_manager(
                plaintext,
                [&](Plaintext<Scheme::BFV>& plaintext_)
                {
                    output_storage_manager(
                        ciphertext,
                        [&](Ciphertext<Scheme::BFV>& ciphertext_)
                        {
                            encrypt_bfv(ciphertext_, plaintext_,
                                        options.stream_);

                            ciphertext.scheme_ = scheme_;
                            ciphertext.ring_size_ = n;
                            ciphertext.coeff_modulus_count_ = Q_size_;
                            ciphertext.cipher_size_ = 2;
                            ciphertext.in_ntt_domain_ = false;
                            ciphertext.relinearization_required_ = false;
                            ciphertext.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, false);
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
        __host__ void encrypt_bfv(Ciphertext<Scheme::BFV>& ciphertext,
                                  Plaintext<Scheme::BFV>& plaintext,
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
#endif // HEONGPU_BFV_ENCRYPTOR_H
