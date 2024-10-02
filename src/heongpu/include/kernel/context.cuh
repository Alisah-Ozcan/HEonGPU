// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef CONTEXT_H
#define CONTEXT_H

#include "memorypool.cuh"
#include "devicevector.cuh"
#include "hostvector.cuh"
#include "common.cuh"
#include "util.cuh"
#include "contextpool.cuh"
#include "nttparameters.cuh"
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <gmp.h>
#include "defines.h"
#include "secstdparams.h"
#include "defaultmodulus.cuh"
#include <unordered_map>

namespace heongpu
{

    class Parameters
    {
        friend class Message;
        friend class Plaintext;
        friend class Ciphertext;

        friend class Secretkey;
        friend class Publickey;
        friend class Relinkey;
        friend class Galoiskey;
        friend class Switchkey;

        friend class HEKeyGenerator;
        friend class HEEncoder;
        friend class HEEncryptor;
        friend class HEDecryptor;
        friend class HEOperator;
        friend class HEStream;

      public:
        Parameters(const scheme_type scheme, const keyswitching_type ks_type,
                   const sec_level_type = sec_level_type::sec128);

        void set_poly_modulus_degree(size_t poly_modulus_degree);

        void set_coeff_modulus(const std::vector<int>& log_Q_bases_bit_sizes,
                               const std::vector<int>& log_P_bases_bit_sizes);

        void set_default_coeff_modulus(int P_modulus_size);

        void set_plain_modulus(const int plain_modulus);

        void generate();

        void print_parameters();

      public:
        inline int poly_modulus_degree() const noexcept { return n; }

        inline int log_poly_modulus_degree() const noexcept { return n_power; }

        inline int ciphertext_modulus_count() const noexcept { return Q_size; }

        inline int key_modulus_count() const noexcept { return Q_prime_size; }

        inline Modulus plain_modulus() const noexcept { return plain_modulus_; }

        inline std::vector<Modulus> key_modulus() const noexcept
        {
            return prime_vector;
        }

      private:
        bool check_coeffs(const std::vector<int>& log_Q_bases_bit_sizes,
                          const std::vector<int>& log_P_bases_bit_sizes);

        void generate_keyswitching_params(keyswitching_type type);

      private:
        scheme_type scheme_;
        sec_level_type sec_level_;
        keyswitching_type keyswitching_type_;

        bool coeff_modulus_specified = false;
        bool context_generated = false;

        // CKKS & BFV common parameters
        int n;
        int total_bits;
        int coeff_modulus;
        int bsk_modulus;
        int n_power;
        int total_coeff_bit_count;

        int Q_prime_size;
        int Q_size;
        int P_size;

        int total_bit_count_;

        std::vector<Modulus> prime_vector;
        std::vector<Data> base_q;

        std::vector<int> Qprime_mod_bit_sizes_;
        std::vector<int> Q_mod_bit_sizes_;
        std::vector<int> P_mod_bit_sizes_;

        std::shared_ptr<DeviceVector<Modulus>> modulus_;
        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;
        std::shared_ptr<DeviceVector<Data>> last_q_modinv_;
        std::shared_ptr<DeviceVector<Data>> half_p_;
        std::shared_ptr<DeviceVector<Data>> half_mod_;
        std::shared_ptr<DeviceVector<Data>> factor_;

        // BFV BEHZ multiplication parameters
        std::shared_ptr<DeviceVector<Modulus>> base_Bsk_;
        std::shared_ptr<DeviceVector<Root>> bsk_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> bsk_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse>> bsk_n_inverse_;

        Modulus m_tilde_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_Bsk_;
        std::shared_ptr<DeviceVector<Data>> inv_punctured_prod_mod_base_array_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_m_tilde_;

        Data inv_prod_q_mod_m_tilde_;
        std::shared_ptr<DeviceVector<Data>> inv_m_tilde_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data>> prod_q_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data>> inv_prod_q_mod_Bsk_;

        Modulus plain_modulus_;

        std::shared_ptr<DeviceVector<Data>> base_change_matrix_q_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_msk_;

        std::shared_ptr<DeviceVector<Data>> inv_punctured_prod_mod_B_array_;
        Data inv_prod_B_mod_m_sk_;
        std::shared_ptr<DeviceVector<Data>> prod_B_mod_q_;

        std::shared_ptr<DeviceVector<Modulus>> q_Bsk_merge_modulus_;
        std::shared_ptr<DeviceVector<Root>> q_Bsk_merge_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> q_Bsk_merge_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse>> q_Bsk_n_inverse_;

        // BFV decryption parameters
        std::shared_ptr<DeviceVector<Modulus>> plain_modulus2_;
        std::shared_ptr<DeviceVector<Ninverse>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root>> plain_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> plain_intt_tables_;

        Modulus gamma_;
        std::shared_ptr<DeviceVector<Data>> coeeff_div_plainmod_;
        Data Q_mod_t_;

        Data upper_threshold_;
        std::shared_ptr<DeviceVector<Data>> upper_halfincrement_;

        std::shared_ptr<DeviceVector<Data>> Qi_t_;
        std::shared_ptr<DeviceVector<Data>> Qi_gamma_;
        std::shared_ptr<DeviceVector<Data>> Qi_inverse_;

        Data mulq_inv_t_;
        Data mulq_inv_gamma_;
        Data inv_gamma_;

        // CKKS switchkey & rescale parameters
        std::shared_ptr<DeviceVector<Data>> rescaled_last_q_modinv_;
        std::shared_ptr<DeviceVector<Data>> rescaled_half_;
        std::shared_ptr<DeviceVector<Data>> rescaled_half_mod_;

        // CKKS encode & decode parameters || BFV noise budget calculation
        // parameters
        std::shared_ptr<DeviceVector<Data>> Mi_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data>> decryption_modulus_;

        // BFV switchkey parameters(for Method II & Method III, not Method I)
        int m;
        int l;
        int l_tilda;
        int d;
        int d_tilda;
        int r_prime;

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

        std::shared_ptr<DeviceVector<Data>> base_change_matrix_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data>> prod_D_to_Q_tilda_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;

        // CKKS switchkey parameters(for Method II & Method III, not Method I)
        int m_leveled;
        std::shared_ptr<std::vector<int>> l_leveled;
        std::shared_ptr<std::vector<int>> l_tilda_leveled;
        std::shared_ptr<std::vector<int>> d_leveled;
        std::shared_ptr<std::vector<int>> d_tilda_leveled;
        int r_prime_leveled;

        std::shared_ptr<DeviceVector<Modulus>> B_prime_leveled;
        std::shared_ptr<DeviceVector<Root>> B_prime_ntt_tables_leveled;
        std::shared_ptr<DeviceVector<Root>> B_prime_intt_tables_leveled;
        std::shared_ptr<DeviceVector<Ninverse>> B_prime_n_inverse_leveled;

        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_D_to_B_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_B_to_D_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>> Mi_inv_D_to_B_leveled;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_B_to_D_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_D_to_B_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_B_to_D_leveled;

        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_D_to_Qtilda_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            Mi_inv_D_to_Qtilda_leveled;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            prod_D_to_Qtilda_leveled;

        std::shared_ptr<std::vector<DeviceVector<int>>> I_j_leveled;
        std::shared_ptr<std::vector<DeviceVector<int>>> I_location_leveled;
        std::shared_ptr<std::vector<DeviceVector<int>>> Sk_pair_leveled;

        // int* prime_location_leveled;
        std::shared_ptr<DeviceVector<int>> prime_location_leveled;

      private:
        std::vector<Data> generate_last_q_modinv();

        std::vector<Data> generate_half();

        std::vector<Data> generate_half_mod(std::vector<Data> half);

        std::vector<Data> generate_factor();

        // RNS PARAMETER GENERATOR

        std::vector<Data> generate_Mi(std::vector<Modulus> primes, int size);

        std::vector<Data> generate_Mi_inv(std::vector<Modulus> primes,
                                          int size);

        std::vector<Data> generate_M(std::vector<Modulus> primes, int size);

        std::vector<Data>
        generate_upper_half_threshold(std::vector<Modulus> primes, int size);

        Data generate_Q_mod_t(std::vector<Modulus> primes, Modulus& plain_mod,
                              int size);

        std::vector<Data>
        generate_coeff_div_plain_modulus(std::vector<Modulus> primes,
                                         Modulus& plain_mod, int size);

        // BFV MULTIPLICATION PARAMETERS

        std::vector<Data>
        generate_base_matrix_q_Bsk(std::vector<Modulus> primes,
                                   std::vector<Modulus> bsk_mod, int size);

        // inv_punctured_prod_mod_base_array --> generate_Mi_inv

        std::vector<Data>
        generate_base_change_matrix_m_tilde(std::vector<Modulus> primes,
                                            Modulus mtilda, int size);

        Data generate_inv_prod_q_mod_m_tilde(std::vector<Modulus> primes,
                                             Modulus mtilda, int size);

        std::vector<Data>
        generate_inv_m_tilde_mod_Bsk(std::vector<Modulus> bsk_mod,
                                     Modulus mtilda);

        std::vector<Data> generate_prod_q_mod_Bsk(std::vector<Modulus> primes,
                                                  std::vector<Modulus> bsk_mod,
                                                  int size);

        std::vector<Data>
        generate_inv_prod_q_mod_Bsk(std::vector<Modulus> primes,
                                    std::vector<Modulus> bsk_mod, int size);

        std::vector<Data>
        generate_base_matrix_Bsk_q(std::vector<Modulus> primes,
                                   std::vector<Modulus> bsk_mod, int size);

        std::vector<Data>
        generate_base_change_matrix_msk(std::vector<Modulus> bsk_mod);

        std::vector<Data>
        generate_inv_punctured_prod_mod_B_array(std::vector<Modulus> bsk_mod);

        Data generate_inv_prod_B_mod_m_sk(std::vector<Modulus> bsk_mod);

        std::vector<Data> generate_prod_B_mod_q(std::vector<Modulus> primes,
                                                std::vector<Modulus> bsk_mod,
                                                int size);

        std::vector<Modulus>
        generate_q_Bsk_merge_modulus(std::vector<Modulus> primes,
                                     std::vector<Modulus> bsk_mod, int size);

        std::vector<Data>
        generate_q_Bsk_merge_root(std::vector<Data> primes_psi,
                                  std::vector<Data> bsk_mod_psi, int size);

        // BFV DECRYPTION PARAMETERS

        std::vector<Data> generate_Qi_t(std::vector<Modulus> primes,
                                        Modulus& plain_mod, int size);

        std::vector<Data> generate_Qi_gamma(std::vector<Modulus> primes,
                                            Modulus& gamma, int size);

        std::vector<Data> generate_Qi_inverse(std::vector<Modulus> primes,
                                              int size); // use generate_Mi_inv

        Data generate_mulq_inv_t(std::vector<Modulus> primes,
                                 Modulus& plain_mod, int size);

        Data generate_mulq_inv_gamma(std::vector<Modulus> primes,
                                     Modulus& gamma, int size);

        Data generate_inv_gamma(Modulus& plain_mod, Modulus& gamma);
    };

    //////////////////////////////////////////////////////////////////////////////////

    class HEStream
    {
        friend class Message;
        friend class Plaintext;
        friend class Ciphertext;

        friend class Secretkey;
        friend class Publickey;
        friend class Relinkey;
        friend class Galoiskey;

        friend class HEKeyGenerator;
        friend class HEEncoder;
        friend class HEEncryptor;
        friend class HEDecryptor;
        friend class HEOperator;

      public:
        __host__ HEStream() = delete;
        __host__ HEStream(Parameters context);

        operator cudaStream_t() const { return stream; }

        //~HEStream() { cudaStreamDestroy(stream); }

      private:
        cudaStream_t stream;

        scheme_type scheme;
        int n;

        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        int bsk_modulus_count_;

        ///////////////////////////////////

        int d;
        int d_tilda;
        int r_prime;

        std::shared_ptr<std::vector<int>> d_leveled_;
        std::shared_ptr<std::vector<int>> d_tilda_leveled_;
        int r_prime_leveled_;

        ///////////////////////////////////

        // Temp(to avoid allocation time)
        DeviceVector<COMPLEX> temp_complex;

        DeviceVector<Data> temp_data;
        Data* temp1_enc;
        Data* temp2_enc;

        DeviceVector<Data> temp_mul;
        Data* temp1_mul;
        Data* temp2_mul;

        DeviceVector<Data> temp_relin;
        Data* temp1_relin;
        Data* temp2_relin;

        DeviceVector<Data> temp_relin_new;
        Data* temp1_relin_new;
        Data* temp2_relin_new;
        Data* temp3_relin_new;

        DeviceVector<Data> temp_rescale;
        Data* temp1_rescale;
        Data* temp2_rescale;

        DeviceVector<Data> temp_rotation;
        Data* temp0_rotation;
        Data* temp1_rotation;
        Data* temp2_rotation;
        Data* temp3_rotation;
        Data* temp4_rotation;

        DeviceVector<Data> temp_plain_mul;
        Data* temp1_plain_mul;

        DeviceVector<Data> temp_mod_drop_;
        Data* temp_mod_drop;
    };

} // namespace heongpu
#endif // CONTEXT_H
