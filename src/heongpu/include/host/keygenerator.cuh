// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef KEYGENERATOR_H
#define KEYGENERATOR_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "keygeneration.cuh"
#include "keyswitch.cuh"
#include "secretkey.cuh"
#include "publickey.cuh"

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
     */
    class HEKeyGenerator
    {
      public:
        /**
         * @brief Constructs a new HEKeyGenerator object with specified
         * parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ HEKeyGenerator(Parameters& context);

        /**
         * @brief Generates a secret key.
         *
         * @param sk Reference to the Secretkey object where the generated
         * secret key will be stored.
         */
        __host__ void generate_secret_key(Secretkey& sk);

        /**
         * @brief Generates conjugation of given secret key.
         *
         * @param conj_sk Reference to the Secretkey object where the conjugated
         * secret key will be stored. This object will be modified to contain
         * the conjugate of the original secret key.
         * @param sk Reference to the original Secretkey object that is used as
         * the source for the conjugation. This object remains unchanged.
         */
        __host__ void generate_conjugate_secret_key(Secretkey& conj_sk,
                                                    Secretkey& orginal_sk);

        /**
         * @brief Generates a public key using a secret key.
         *
         * @param pk Reference to the Publickey object where the generated
         * public key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * public key.
         */
        __host__ void generate_public_key(Publickey& pk, Secretkey& sk);

        /**
         * @brief Generates a relinearization key using a secret key.
         *
         * @param rk Reference to the Relinkey object where the generated
         * relinearization key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * relinearization key.
         */
        __host__ void generate_relin_key(Relinkey& rk, Secretkey& sk)
        {
            switch (static_cast<int>(rk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_relin_key_method_I(rk, sk);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (rk.scheme_ == scheme_type::bfv)
                    { // no leveled
                        generate_bfv_relin_key_method_II(rk, sk);
                    }
                    else if (rk.scheme_ == scheme_type::ckks)
                    { // leveled
                        generate_ckks_relin_key_method_II(rk, sk);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

                    break;
                case 3: // KEYSWITCHING_METHOD_III

                    if (rk.scheme_ == scheme_type::bfv)
                    { // no leveled
                        generate_bfv_relin_key_method_III(rk, sk);
                    }
                    else if (rk.scheme_ == scheme_type::ckks)
                    { // leveled
                        generate_ckks_relin_key_method_III(rk, sk);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

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
        __host__ void generate_galois_key(Galoiskey& gk, Secretkey& sk)
        {
            switch (static_cast<int>(gk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_galois_key_method_I(gk, sk);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (gk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_galois_key_method_II(gk, sk);
                    }
                    else if (gk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_galois_key_method_II(gk, sk);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

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
        __host__ void generate_switch_key(Switchkey& swk, Secretkey& new_sk,
                                          Secretkey& old_sk)
        {
            switch (static_cast<int>(swk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_switch_key_method_I(swk, new_sk, old_sk);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (swk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_switch_key_method_II(swk, new_sk, old_sk);
                    }
                    else if (swk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_switch_key_method_II(swk, new_sk, old_sk);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

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

        HEKeyGenerator() = delete;
        HEKeyGenerator(const HEKeyGenerator& copy) = delete;
        HEKeyGenerator(HEKeyGenerator&& source) = delete;
        HEKeyGenerator& operator=(const HEKeyGenerator& assign) = delete;
        HEKeyGenerator& operator=(HEKeyGenerator&& assign) = delete;

      private:
        __host__ void generate_relin_key_method_I(Relinkey& rk, Secretkey& sk);

        __host__ void generate_bfv_relin_key_method_II(Relinkey& rk,
                                                       Secretkey& sk);

        __host__ void generate_bfv_relin_key_method_III(Relinkey& rk,
                                                        Secretkey& sk);

        __host__ void generate_ckks_relin_key_method_II(Relinkey& rk,
                                                        Secretkey& sk);

        __host__ void generate_ckks_relin_key_method_III(Relinkey& rk,
                                                         Secretkey& sk);

        __host__ void generate_galois_key_method_I(Galoiskey& gk,
                                                   Secretkey& sk);

        __host__ void generate_bfv_galois_key_method_II(Galoiskey& gk,
                                                        Secretkey& sk);

        __host__ void generate_ckks_galois_key_method_II(Galoiskey& gk,
                                                         Secretkey& sk);

        __host__ void generate_switch_key_method_I(Switchkey& swk,
                                                   Secretkey& new_sk,
                                                   Secretkey& old_sk);

        __host__ void generate_bfv_switch_key_method_II(Switchkey& swk,
                                                        Secretkey& new_sk,
                                                        Secretkey& old_sk);

        __host__ void generate_ckks_switch_key_method_II(Switchkey& swk,
                                                         Secretkey& new_sk,
                                                         Secretkey& old_sk);

      private:
        scheme_type scheme;
        int seed_;

        int n;

        int n_power;

        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus>> modulus_;
        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;
        std::shared_ptr<DeviceVector<Data>> factor_;

        int d_;
        int d_tilda_;
        int r_prime_;

        std::shared_ptr<std::vector<int>> d_leveled_;
        std::shared_ptr<std::vector<int>> d_tilda_leveled_;
        int r_prime_leveled_;

        std::shared_ptr<DeviceVector<Modulus>> B_prime_;
        std::shared_ptr<DeviceVector<Root>> B_prime_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> B_prime_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse>> B_prime_n_inverse_;

        std::shared_ptr<DeviceVector<Data>> base_change_matrix_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_B_to_D_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_B_to_D_;
        std::shared_ptr<DeviceVector<Data>> prod_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> prod_B_to_D_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;

        std::shared_ptr<DeviceVector<Modulus>> B_prime_leveled_;
        std::shared_ptr<DeviceVector<Root>> B_prime_ntt_tables_leveled_;
        std::shared_ptr<DeviceVector<Root>> B_prime_intt_tables_leveled_;
        std::shared_ptr<DeviceVector<Ninverse>> B_prime_n_inverse_leveled_;

        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> Mi_inv_D_to_B_leveled_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_B_to_D_leveled_;

        std::shared_ptr<std::vector<DeviceVector<int>>> I_j_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> I_location_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> Sk_pair_leveled_;

        std::shared_ptr<DeviceVector<int>> prime_location_leveled_;
    };

} // namespace heongpu
#endif // KEYGENERATOR_H