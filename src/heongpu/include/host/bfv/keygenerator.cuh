// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_KEYGENERATOR_H
#define HEONGPU_BFV_KEYGENERATOR_H

#include "ntt.cuh"
#include "keygeneration.cuh"
#include "switchkey.cuh"
#include "bfv/context.cuh"
#include "bfv/secretkey.cuh"
#include "bfv/publickey.cuh"
#include "bfv/evaluationkey.cuh"

namespace heongpu
{

    /**
     * @brief HEKeyGenerator is responsible for generating various keys used in
     * homomorphic encryption, such as secret keys, public keys, relinearization
     * keys, Galois keys, and switch keys.
     *
     * The HEKeyGenerator class is initialized with encryption parameters and
     * provides methods to generate the keys required for different homomorphic
     * encryption schemes. It helps facilitate the encryption, decryption, and
     * key switching operations.
     *
     * Additionally, the class supports multiparty computation (MPC) by
     * implementing key generation protocols for collaborative schemes. These
     * include the generation of multiparty public keys, relinearization keys,
     * and Galois keys. The implementation is based on the techniques described
     * in the paper "Multiparty Homomorphic Encryption from
     * Ring-Learning-With-Errors".
     */
    template <> class HEKeyGenerator<Scheme::BFV>
    {
      public:
        /**
         * @brief Constructs a new HEKeyGenerator object with specified
         * parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ HEKeyGenerator(HEContext<Scheme::BFV>& context);

        /**
         * @brief Generates a secret key.
         *
         * @param sk Reference to the Secretkey object where the generated
         * secret key will be stored.
         */
        __host__ void generate_secret_key(
            Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a public key using a secret key.
         *
         * @param pk Reference to the Publickey object where the generated
         * public key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * public key.
         */
        __host__ void generate_public_key(
            Publickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a relinearization key using a secret key.
         *
         * @param rk Reference to the Relinkey object where the generated
         * relinearization key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * relinearization key.
         */
        __host__ void
        generate_relin_key(Relinkey<Scheme::BFV>& rk,
                           Secretkey<Scheme::BFV>& sk,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(rk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_relin_key_method_I(rk, sk, options);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    generate_bfv_relin_key_method_II(rk, sk, options);
                    break;
                case 3: // KEYSWITCHING_METHOD_III
                    generate_bfv_relin_key_method_III(rk, sk, options);
                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Generates a Galois key using a secret key.
         *
         * @param gk Reference to the Galoiskey object where the generated
         * Galois key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * Galois key.
         */
        __host__ void generate_galois_key(
            Galoiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(gk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_galois_key_method_I(gk, sk, options);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    generate_bfv_galois_key_method_II(gk, sk, options);
                    break;
                case 3: // KEYSWITCHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITCHING_METHOD_III are not supported because of "
                        "high memory consumption for galois key generation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Generates a switch key for key switching between two secret
         * keys.
         *
         * @param swk Reference to the Switchkey object where the generated
         * switch key will be stored.
         * @param new_sk Reference to the new Secretkey object to switch to.
         * @param old_sk Reference to the old Secretkey object to switch from.
         */
        __host__ void generate_switch_key(
            Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
            Secretkey<Scheme::BFV>& old_sk,
            const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(swk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_switch_key_method_I(swk, new_sk, old_sk, options);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    generate_bfv_switch_key_method_II(swk, new_sk, old_sk,
                                                      options);
                    break;
                case 3: // KEYSWITCHING_METHOD_III
                    throw std::invalid_argument(
                        "KEYSWITCHING_METHOD_III are not supported because of "
                        "high memory consumption for galois key generation!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Returns the seed of the key generator.
         *
         * @return int Seed of the key generator.
         */
        inline int get_seed() const noexcept { return seed_; }

        /**
         * @brief Sets the seed of the key generator with new seed.
         */
        inline void set_seed(int new_seed) { seed_ = new_seed; }

        /**
         * @brief Returns the offset of the key generator(curand).
         *
         * @return int Offset of the key generator.
         */
        inline int get_offset() const noexcept { return offset_; }

        /**
         * @brief Sets the offset of the key generator with new offset(curand).
         */
        inline void set_offset(int new_offset) { offset_ = new_offset; }

        HEKeyGenerator() = delete;
        HEKeyGenerator(const HEKeyGenerator& copy) = delete;
        HEKeyGenerator(HEKeyGenerator&& source) = delete;
        HEKeyGenerator& operator=(const HEKeyGenerator& assign) = delete;
        HEKeyGenerator& operator=(HEKeyGenerator&& assign) = delete;

      private:
        __host__ void
        generate_relin_key_method_I(Relinkey<Scheme::BFV>& rk,
                                    Secretkey<Scheme::BFV>& sk,
                                    const ExecutionOptions& options);

        __host__ void
        generate_bfv_relin_key_method_II(Relinkey<Scheme::BFV>& rk,
                                         Secretkey<Scheme::BFV>& sk,
                                         const ExecutionOptions& options);

        __host__ void
        generate_bfv_relin_key_method_III(Relinkey<Scheme::BFV>& rk,
                                          Secretkey<Scheme::BFV>& sk,
                                          const ExecutionOptions& options);

        __host__ void
        generate_galois_key_method_I(Galoiskey<Scheme::BFV>& gk,
                                     Secretkey<Scheme::BFV>& sk,
                                     const ExecutionOptions& options);

        __host__ void
        generate_bfv_galois_key_method_II(Galoiskey<Scheme::BFV>& gk,
                                          Secretkey<Scheme::BFV>& sk,
                                          const ExecutionOptions& options);

        __host__ void generate_switch_key_method_I(
            Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
            Secretkey<Scheme::BFV>& old_sk, const ExecutionOptions& options);

        __host__ void generate_bfv_switch_key_method_II(
            Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
            Secretkey<Scheme::BFV>& old_sk, const ExecutionOptions& options);

      private:
        scheme_type scheme;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

        RNGSeed new_seed_;

        int n;

        int n_power;

        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> factor_;

        int d_;
        int d_tilda_;
        int r_prime_;

        std::shared_ptr<DeviceVector<Modulus64>> B_prime_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> B_prime_n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_to_D_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_KEYGENERATOR_H