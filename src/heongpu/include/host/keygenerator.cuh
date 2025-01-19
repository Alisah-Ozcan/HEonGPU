// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef KEYGENERATOR_H
#define KEYGENERATOR_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "random.cuh"
#include "switchkey.cuh"
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
     *
     * Additionally, the class supports multiparty computation (MPC) by
     * implementing key generation protocols for collaborative schemes. These
     * include the generation of multiparty public keys, relinearization keys,
     * and Galois keys. The implementation is based on the techniques described
     * in the paper "Multiparty Homomorphic Encryption from
     * Ring-Learning-With-Errors".
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
        __host__ void
        generate_secret_key(Secretkey& sk,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates a public key using a secret key.
         *
         * @param pk Reference to the Publickey object where the generated
         * public key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * public key.
         */
        __host__ void
        generate_public_key(Publickey& pk, Secretkey& sk,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates a partial public key for multiparty computation
         * (Each participant).
         *
         * Each participant generates a partial public key piece using their
         * secret key. These partial public keys will later be combined to form
         * the final public key.
         *
         * @param pk Reference to the MultipartyPublickey object where the
         * generated partial public key will be stored.
         * @param sk The Secretkey of the participant generating the partial
         * public key.
         */
        __host__ void generate_multi_party_public_key_piece(
            MultipartyPublickey& pk, Secretkey& sk,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Combines partial public keys from all participants into a
         * final public key (Collective).
         *
         * This function aggregates all partial public keys generated by the
         * participants into a single final public key for use in multiparty
         * computations.
         *
         * @param all_pk Vector containing the partial public keys from all
         * participants.
         * @param pk Reference to the Publickey object where the combined final
         * public key will be stored.
         */
        __host__ void generate_multi_party_public_key(
            std::vector<MultipartyPublickey>& all_pk, Publickey& pk,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates a relinearization key using a secret key.
         *
         * @param rk Reference to the Relinkey object where the generated
         * relinearization key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * relinearization key.
         */
        __host__ void
        generate_relin_key(Relinkey& rk, Secretkey& sk,
                           cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(rk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_relin_key_method_I(rk, sk, stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (rk.scheme_ == scheme_type::bfv)
                    { // no leveled
                        generate_bfv_relin_key_method_II(rk, sk, stream);
                    }
                    else if (rk.scheme_ == scheme_type::ckks)
                    { // leveled
                        generate_ckks_relin_key_method_II(rk, sk, stream);
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
                        generate_bfv_relin_key_method_III(rk, sk, stream);
                    }
                    else if (rk.scheme_ == scheme_type::ckks)
                    { // leveled
                        generate_ckks_relin_key_method_III(rk, sk, stream);
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
         * @brief Generates a partial relinearization key piece for multiparty
         * computation (Stage 1 - Each).
         *
         * Each party generates its own partial relinearization key piece using
         * their secret key. The process depends on the specified key switching
         * method and scheme type (BFV or CKKS).
         *
         * @param rk Reference to the MultipartyRelinkey object that will store
         * the generated key piece.
         * @param sk The Secretkey of the participant generating the partial
         * relinearization key.
         */
        __host__ void generate_multi_party_relin_key_piece(
            MultipartyRelinkey& rk, Secretkey& sk,
            cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(rk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_multi_party_relin_key_piece_method_I_stage_I(
                        rk, sk, stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    if (rk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_multi_party_relin_key_piece_method_II_stage_I(
                            rk, sk, stream);
                    }
                    else if (rk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_multi_party_relin_key_piece_method_II_stage_I(
                            rk, sk, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITCHING_METHOD_III
                    throw std::invalid_argument(
                        "Key Switching Type III is not supported for multi "
                        "party key generation.");
                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Generates a partial relinearization key piece for multiparty
         * computation (Stage 2 - Each).
         *
         * This function processes the partial relinearization key pieces from
         * Stage 1 to produce updated key pieces. It verifies the compatibility
         * of the input relinearization key parameters.
         *
         * @param rk_s1_common Reference to the shared Stage 1 relinearization
         * key.
         * @param rk_new Reference to the MultipartyRelinkey object for storing
         * the new key piece.
         * @param sk The Secretkey of the participant generating the updated
         * relinearization key piece.
         */
        __host__ void generate_multi_party_relin_key_piece(
            MultipartyRelinkey& rk_s1_common, MultipartyRelinkey& rk_new,
            Secretkey& sk, cudaStream_t stream = cudaStreamDefault)
        {
            if ((rk_s1_common.scheme_ != rk_new.scheme_) ||
                (rk_s1_common.key_type != rk_new.key_type))
            {
                throw std::invalid_argument("Invalid relinkey parameters!");
            }

            switch (static_cast<int>(rk_s1_common.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_multi_party_relin_key_piece_method_I_stage_II(
                        rk_s1_common, rk_new, sk, stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    if (rk_s1_common.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_multi_party_relin_key_piece_method_II_stage_II(
                            rk_s1_common, rk_new, sk, stream);
                    }
                    else if (rk_s1_common.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_multi_party_relin_key_piece_method_II_stage_II(
                            rk_s1_common, rk_new, sk, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITCHING_METHOD_III
                    throw std::invalid_argument(
                        "Key Switching Type III is not supported for multi "
                        "party key generation.");
                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Combines partial relinearization keys from all participants
         * (Stage 1 - Collective).
         *
         * This function aggregates all partial relinearization key pieces from
         * multiple participants into a single collective relinearization key
         * for Stage 1.
         *
         * @param all_rk Vector containing partial relinearization keys from all
         * participants.
         * @param rk Reference to the MultipartyRelinkey object for storing the
         * aggregated key.
         */
        __host__ void
        generate_multi_party_relin_key(std::vector<MultipartyRelinkey>& all_rk,
                                       MultipartyRelinkey& rk,
                                       cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Combines partial relinearization keys from all participants
         * (Stage 2 - Collective).
         *
         * In Stage 2, this function aggregates updated relinearization key
         * pieces from multiple participants into a single collective
         * relinearization key.
         *
         * @param all_rk Vector containing updated relinearization keys from all
         * participants.
         * @param rk_common_stage1 Reference to the shared Stage 1 collective
         * relinearization key.
         * @param rk Reference to the final Relinkey object that will store the
         * aggregated key.
         */
        __host__ void
        generate_multi_party_relin_key(std::vector<MultipartyRelinkey>& all_rk,
                                       MultipartyRelinkey& rk_common_stage1,
                                       Relinkey& rk,
                                       cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates a Galois key using a secret key.
         *
         * @param gk Reference to the Galoiskey object where the generated
         * Galois key will be stored.
         * @param sk Reference to the Secretkey object used to generate the
         * Galois key.
         */
        __host__ void
        generate_galois_key(Galoiskey& gk, Secretkey& sk,
                            cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(gk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_galois_key_method_I(gk, sk, stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (gk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_galois_key_method_II(gk, sk, stream);
                    }
                    else if (gk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_galois_key_method_II(gk, sk, stream);
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
         * @brief Generates a partial Galois key for multiparty computation
         * (Each participant).
         *
         * Each participant generates a partial Galois key piece using their
         * secret key. The generated key piece is scheme-dependent (BFV or CKKS)
         * and follows the specified key switching method.
         *
         * @param gk Reference to the MultipartyGaloiskey object where the
         * generated partial Galois key will be stored.
         * @param sk The Secretkey of the participant generating the partial
         * Galois key.
         * @throws std::invalid_argument If an unsupported key switching type or
         * scheme is specified.
         */
        __host__ void generate_multi_party_galios_key_piece(
            MultipartyGaloiskey& gk, Secretkey& sk,
            cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(gk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_multi_party_galois_key_piece_method_I(gk, sk,
                                                                   stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II
                    if (gk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_multi_party_galois_key_piece_method_II(
                            gk, sk, stream);
                    }
                    else if (gk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_multi_party_galois_key_piece_method_II(
                            gk, sk, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITCHING_METHOD_III
                    throw std::invalid_argument(
                        "Key Switching Type III is not supported for multi "
                        "party key generation.");
                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
        }

        /**
         * @brief Combines partial Galois keys from all participants into a
         * final Galois key (Collective).
         *
         * This function aggregates all partial Galois keys generated by the
         * participants into a single final Galois key for use in multiparty
         * computations.
         *
         * @param all_gk Vector containing the partial Galois keys from all
         * participants.
         * @param gk Reference to the Galoiskey object where the combined final
         * Galois key will be stored.
         */
        __host__ void generate_multi_party_galois_key(
            std::vector<MultipartyGaloiskey>& all_gk, Galoiskey& gk,
            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Generates a switch key for key switching between two secret
         * keys.
         *
         * @param swk Reference to the Switchkey object where the generated
         * switch key will be stored.
         * @param new_sk Reference to the new Secretkey object to switch to.
         * @param old_sk Reference to the old Secretkey object to switch from.
         */
        __host__ void
        generate_switch_key(Switchkey& swk, Secretkey& new_sk,
                            Secretkey& old_sk,
                            cudaStream_t stream = cudaStreamDefault)
        {
            switch (static_cast<int>(swk.key_type))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    generate_switch_key_method_I(swk, new_sk, old_sk, stream);
                    break;
                case 2: // KEYSWITCHING_METHOD_II

                    if (swk.scheme_ == scheme_type::bfv)
                    {
                        generate_bfv_switch_key_method_II(swk, new_sk, old_sk,
                                                          stream);
                    }
                    else if (swk.scheme_ == scheme_type::ckks)
                    {
                        generate_ckks_switch_key_method_II(swk, new_sk, old_sk,
                                                           stream);
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
        __host__ void generate_relin_key_method_I(Relinkey& rk, Secretkey& sk,
                                                  const cudaStream_t stream);

        __host__ void
        generate_bfv_relin_key_method_II(Relinkey& rk, Secretkey& sk,
                                         const cudaStream_t stream);

        __host__ void
        generate_bfv_relin_key_method_III(Relinkey& rk, Secretkey& sk,
                                          const cudaStream_t stream);

        __host__ void
        generate_ckks_relin_key_method_II(Relinkey& rk, Secretkey& sk,
                                          const cudaStream_t stream);

        __host__ void
        generate_ckks_relin_key_method_III(Relinkey& rk, Secretkey& sk,
                                           const cudaStream_t stream);

        __host__ void generate_galois_key_method_I(Galoiskey& gk, Secretkey& sk,
                                                   const cudaStream_t stream);

        __host__ void
        generate_bfv_galois_key_method_II(Galoiskey& gk, Secretkey& sk,
                                          const cudaStream_t stream);

        __host__ void
        generate_ckks_galois_key_method_II(Galoiskey& gk, Secretkey& sk,
                                           const cudaStream_t stream);

        __host__ void generate_switch_key_method_I(Switchkey& swk,
                                                   Secretkey& new_sk,
                                                   Secretkey& old_sk,
                                                   const cudaStream_t stream);

        __host__ void
        generate_bfv_switch_key_method_II(Switchkey& swk, Secretkey& new_sk,
                                          Secretkey& old_sk,
                                          const cudaStream_t stream);

        __host__ void
        generate_ckks_switch_key_method_II(Switchkey& swk, Secretkey& new_sk,
                                           Secretkey& old_sk,
                                           const cudaStream_t stream);

        __host__ void generate_multi_party_relin_key_piece_method_I_stage_I(
            MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream);

        __host__ void generate_multi_party_relin_key_piece_method_I_stage_II(
            MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
            Secretkey& sk, const cudaStream_t stream);

        __host__ void
        generate_bfv_multi_party_relin_key_piece_method_II_stage_I(
            MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream);

        __host__ void
        generate_bfv_multi_party_relin_key_piece_method_II_stage_II(
            MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
            Secretkey& sk, const cudaStream_t stream);

        __host__ void
        generate_ckks_multi_party_relin_key_piece_method_II_stage_I(
            MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream);

        __host__ void
        generate_ckks_multi_party_relin_key_piece_method_II_stage_II(
            MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
            Secretkey& sk, const cudaStream_t stream);

        __host__ void generate_multi_party_galois_key_piece_method_I(
            MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream);

        __host__ void generate_bfv_multi_party_galois_key_piece_method_II(
            MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream);

        __host__ void generate_ckks_multi_party_galois_key_piece_method_II(
            MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream);

      private:
        scheme_type scheme;
        int seed_;
        int offset_; // Absolute offset into sequence (curand)

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