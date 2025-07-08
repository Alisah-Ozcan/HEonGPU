// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_BFV_CONTEXT_H
#define HEONGPU_BFV_CONTEXT_H

#include "util.cuh"
#include "schemes.h"
#include "devicevector.cuh"
#include "hostvector.cuh"
#include "secstdparams.h"
#include "defaultmodulus.cuh"
#include "random.cuh"
#include <gmp.h>
#include "contextpool.cuh"
#include <ostream>
#include <istream>

namespace heongpu
{
    template <> class HEContext<Scheme::BFV>
    {
        template <Scheme S> friend class Secretkey;
        template <Scheme S> friend class Publickey;
        template <Scheme S> friend class MultipartyPublickey;
        template <Scheme S> friend class Plaintext;
        template <Scheme S> friend class Ciphertext;
        template <Scheme S> friend class Relinkey;
        template <Scheme S> friend class MultipartyRelinkey;
        template <Scheme S> friend class Galoiskey;
        template <Scheme S> friend class MultipartyGaloiskey;
        template <Scheme S> friend class Switchkey;
        template <Scheme S> friend class HEEncoder;
        template <Scheme S> friend class HEKeyGenerator;
        template <Scheme S> friend class HEEncryptor;
        template <Scheme S> friend class HEDecryptor;
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;
        template <Scheme S> friend class HEMultiPartyManager;

      public:
        HEContext(const keyswitching_type ks_type,
                  const sec_level_type = sec_level_type::sec128);

        void set_poly_modulus_degree(size_t poly_modulus_degree);

        void set_coeff_modulus_bit_sizes(
            const std::vector<int>& log_Q_bases_bit_sizes,
            const std::vector<int>& log_P_bases_bit_sizes);

        void set_coeff_modulus_values(const std::vector<Data64>& log_Q_bases,
                                      const std::vector<Data64>& log_P_bases);

        void set_coeff_modulus_default_values(int P_modulus_size);

        void set_plain_modulus(const int plain_modulus);

        void generate();

        void print_parameters();

        inline int get_poly_modulus_degree() const noexcept { return n; }

        inline int get_log_poly_modulus_degree() const noexcept
        {
            return n_power;
        }

        inline int get_ciphertext_modulus_count() const noexcept
        {
            return Q_size;
        }

        inline int get_key_modulus_count() const noexcept
        {
            return Q_prime_size;
        }

        inline Modulus64 get_plain_modulus() const noexcept
        {
            return plain_modulus_;
        }

        inline std::vector<Modulus64> get_key_modulus() const noexcept
        {
            return prime_vector_;
        }

        HEContext() = default;

        void save(std::ostream& os) const;

        void load(std::istream& is);

      private:
        Data64 generate_Q_mod_t(std::vector<Modulus64> primes,
                                Modulus64& plain_mod, int size);

        std::vector<Data64>
        generate_coeff_div_plain_modulus(std::vector<Modulus64> primes,
                                         Modulus64& plain_mod, int size);

        // BFV MULTIPLICATION PARAMETERS
        std::vector<Data64>
        generate_base_matrix_q_Bsk(std::vector<Modulus64> primes,
                                   std::vector<Modulus64> bsk_mod, int size);

        std::vector<Data64>
        generate_base_change_matrix_m_tilde(std::vector<Modulus64> primes,
                                            Modulus64 mtilda, int size);

        Data64 generate_inv_prod_q_mod_m_tilde(std::vector<Modulus64> primes,
                                               Modulus64 mtilda, int size);

        std::vector<Data64>
        generate_inv_m_tilde_mod_Bsk(std::vector<Modulus64> bsk_mod,
                                     Modulus64 mtilda);

        std::vector<Data64>
        generate_prod_q_mod_Bsk(std::vector<Modulus64> primes,
                                std::vector<Modulus64> bsk_mod, int size);

        std::vector<Data64>
        generate_inv_prod_q_mod_Bsk(std::vector<Modulus64> primes,
                                    std::vector<Modulus64> bsk_mod, int size);

        std::vector<Data64>
        generate_base_matrix_Bsk_q(std::vector<Modulus64> primes,
                                   std::vector<Modulus64> bsk_mod, int size);

        std::vector<Data64>
        generate_base_change_matrix_msk(std::vector<Modulus64> bsk_mod);

        std::vector<Data64>
        generate_inv_punctured_prod_mod_B_array(std::vector<Modulus64> bsk_mod);

        Data64 generate_inv_prod_B_mod_m_sk(std::vector<Modulus64> bsk_mod);

        std::vector<Data64>
        generate_prod_B_mod_q(std::vector<Modulus64> primes,
                              std::vector<Modulus64> bsk_mod, int size);

        std::vector<Modulus64>
        generate_q_Bsk_merge_modulus(std::vector<Modulus64> primes,
                                     std::vector<Modulus64> bsk_mod, int size);

        std::vector<Data64>
        generate_q_Bsk_merge_root(std::vector<Data64> primes_psi,
                                  std::vector<Data64> bsk_mod_psi, int size);

        // BFV DECRYPTION PARAMETERS
        std::vector<Data64> generate_Qi_t(std::vector<Modulus64> primes,
                                          Modulus64& plain_mod, int size);

        std::vector<Data64> generate_Qi_gamma(std::vector<Modulus64> primes,
                                              Modulus64& gamma, int size);

        std::vector<Data64>
        generate_Qi_inverse(std::vector<Modulus64> primes,
                            int size); // use generate_Mi_inv

        Data64 generate_mulq_inv_t(std::vector<Modulus64> primes,
                                   Modulus64& plain_mod, int size);

        Data64 generate_mulq_inv_gamma(std::vector<Modulus64> primes,
                                       Modulus64& gamma, int size);

        Data64 generate_inv_gamma(Modulus64& plain_mod, Modulus64& gamma);

      private:
        bool poly_modulus_degree_specified_ = false;
        bool coeff_modulus_specified_ = false;
        bool plain_modulus_specified_ = false;
        bool context_generated_ = false;

        scheme_type scheme_;
        sec_level_type sec_level_;
        keyswitching_type keyswitching_type_;

        int n;
        int n_power;

        int coeff_modulus;
        int total_coeff_bit_count;

        int Q_prime_size;
        int Q_size;
        int P_size;

        std::vector<Modulus64> prime_vector_;
        std::vector<Data64> base_q;

        std::vector<int> Qprime_mod_bit_sizes_;
        std::vector<int> Q_mod_bit_sizes_;
        std::vector<int> P_mod_bit_sizes_;

        Modulus64 plain_modulus_;

        int bsk_modulus;
        int total_bit_count_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;
        std::shared_ptr<DeviceVector<Data64>> half_p_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;
        std::shared_ptr<DeviceVector<Data64>> factor_;

        // BFV BEHZ multiplication parameters
        std::shared_ptr<DeviceVector<Modulus64>> base_Bsk_;
        std::shared_ptr<DeviceVector<Root64>> bsk_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> bsk_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> bsk_n_inverse_;

        Modulus64 m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_Bsk_;
        std::shared_ptr<DeviceVector<Data64>>
            inv_punctured_prod_mod_base_array_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_m_tilde_;

        Data64 inv_prod_q_mod_m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> inv_m_tilde_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> prod_q_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> inv_prod_q_mod_Bsk_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_q_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_msk_;

        std::shared_ptr<DeviceVector<Data64>> inv_punctured_prod_mod_B_array_;
        Data64 inv_prod_B_mod_m_sk_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_mod_q_;

        std::shared_ptr<DeviceVector<Modulus64>> q_Bsk_merge_modulus_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> q_Bsk_n_inverse_;

        // BFV decryption parameters
        std::shared_ptr<DeviceVector<Modulus64>> plain_modulus2_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root64>> plain_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> plain_intt_tables_;

        Modulus64 gamma_;
        std::shared_ptr<DeviceVector<Data64>> coeeff_div_plainmod_;
        Data64 Q_mod_t_;

        Data64 upper_threshold_;
        std::shared_ptr<DeviceVector<Data64>> upper_halfincrement_;

        std::shared_ptr<DeviceVector<Data64>> Qi_t_;
        std::shared_ptr<DeviceVector<Data64>> Qi_gamma_;
        std::shared_ptr<DeviceVector<Data64>> Qi_inverse_;

        Data64 mulq_inv_t_;
        Data64 mulq_inv_gamma_;
        Data64 inv_gamma_;

        // BFV noise budget calculation parameters
        std::shared_ptr<DeviceVector<Data64>> Mi_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data64>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data64>> decryption_modulus_;

        // BFV switchkey parameters(for Method II & Method III, not Method I)
        int m;
        int l;
        int l_tilda;
        int d;
        int d_tilda;
        int r_prime;

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

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_Q_tilda_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;
    };

} // namespace heongpu
#endif // HEONGPU_BFV_CONTEXT_H
