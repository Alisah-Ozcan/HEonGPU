﻿// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "context.cuh"

namespace heongpu
{

    std::vector<Data> Parameters::generate_last_q_modinv()
    {
        std::vector<Data> last_q_modinv;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < (Q_prime_size - 1) - i; j++)
            {
                // TODO: Change here for BFV as well!!!
                Data temp_ = prime_vector[prime_vector.size() - 1 - i].value %
                             prime_vector[j].value;
                last_q_modinv.push_back(VALUE::modinv(temp_, prime_vector[j]));
            }
        }

        return last_q_modinv;
    }

    std::vector<Data> Parameters::generate_half()
    {
        std::vector<Data> half;
        for (int i = 0; i < P_size; i++)
        {
            half.push_back(prime_vector[prime_vector.size() - 1 - i].value >>
                           1);
        }

        return half;
    }

    std::vector<Data> Parameters::generate_half_mod(std::vector<Data> half)
    {
        std::vector<Data> half_mod;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < (Q_prime_size - 1) - i; j++)
            {
                half_mod.push_back(half[i] % prime_vector[j].value);
            }
        }

        return half_mod;
    }

    std::vector<Data> Parameters::generate_factor()
    {
        std::vector<Data> factor;
        for (int i = 0; i < P_size; i++)
        {
            for (int j = 0; j < Q_size; j++)
            {
                factor.push_back(
                    prime_vector[prime_vector.size() - 1 - i].value %
                    prime_vector[j].value);
            }
        }

        return factor;
    }

    std::vector<Data> Parameters::generate_Mi(std::vector<Modulus> primes,
                                              int size)
    {
        std::vector<Data> result_Mi(size * size, 0ULL);
        for (int i = 0; i < size; i++)
        {
            mpz_t result;
            mpz_init(result);
            mpz_set_ui(result, 1);

            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    mpz_mul_ui(result, result, primes[j].value);
                }
            }

            size_t mul_size;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mpz_export(
                NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

            for (int j = 0; j < mul_size; j++)
            {
                result_Mi[(size * i) + j] = ptr[j];
            }

            mpz_clear(result);
            free(ptr);
        }

        return result_Mi;
    }

    std::vector<Data> Parameters::generate_Mi_inv(std::vector<Modulus> primes,
                                                  int size)
    {
        std::vector<Data> result_Mi_inv;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data inner_prime = primes[j].value % primes[i].value;
                    temp = VALUE::mult(temp, inner_prime, primes[i]);
                }
            }
            result_Mi_inv.push_back(VALUE::modinv(temp, primes[i]));
        }

        return result_Mi_inv;
    }

    std::vector<Data> Parameters::generate_M(std::vector<Modulus> primes,
                                             int size)
    {
        std::vector<Data> result_M(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, primes[i].value);
        }

        size_t mul_size;
        uint64_t* ptr = reinterpret_cast<uint64_t*>(
            mpz_export(NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

        for (int j = 0; j < mul_size; j++)
        {
            result_M[j] = ptr[j];
        }

        mpz_clear(result);
        free(ptr);

        return result_M;
    }

    std::vector<Data>
    Parameters::generate_upper_half_threshold(std::vector<Modulus> primes,
                                              int size)
    {
        std::vector<Data> result_upper_half_threshold(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, primes[i].value);
        }

        mpz_add_ui(result, result, 1);
        mpz_div_2exp(result, result, 1);

        size_t mul_size;
        uint64_t* ptr = reinterpret_cast<uint64_t*>(
            mpz_export(NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result));

        for (int j = 0; j < mul_size; j++)
        {
            result_upper_half_threshold[j] = ptr[j];
        }

        mpz_clear(result);
        free(ptr);

        return result_upper_half_threshold;
    }

    Data Parameters::generate_Q_mod_t(std::vector<Modulus> primes,
                                      Modulus& plain_mod, int size)
    {
        Data result = 1;
        for (int i = 0; i < size; i++)
        {
            Data inner_prime = primes[i].value % plain_mod.value;
            result = VALUE::mult(result, inner_prime, plain_mod);
        }

        return result;
    }

    std::vector<Data>
    Parameters::generate_coeff_div_plain_modulus(std::vector<Modulus> primes,
                                                 Modulus& plain_mod, int size)
    {
        std::vector<Data> result_M(size, 0ULL);

        mpz_t result;
        mpz_init(result);
        mpz_set_ui(result, 1);

        for (int i = 0; i < size; i++)
        {
            mpz_mul_ui(result, result, primes[i].value);
        }

        mpz_div_ui(result, result, plain_mod.value);

        mpz_t result_mod;
        mpz_init(result_mod);

        for (int i = 0; i < size; i++)
        {
            mpz_mod_ui(result_mod, result, primes[i].value);

            size_t mul_size;
            uint64_t* ptr = reinterpret_cast<uint64_t*>(mpz_export(
                NULL, &mul_size, -1, sizeof(uint64_t), 0, 0, result_mod));

            result_M[i] = ptr[0];
            free(ptr);
        }

        mpz_clear(result);
        mpz_clear(result_mod);

        return result_M;
    }
    std::vector<Data> Parameters::generate_base_matrix_q_Bsk(
        std::vector<Modulus> primes, std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Data> base_matrix_q_Bsk;
        for (int k = 0; k < bsk_mod.size(); k++)
        {
            for (int i = 0; i < size; i++)
            {
                Data temp = 1;
                for (int j = 0; j < size; j++)
                {
                    if (i != j)
                    {
                        temp = VALUE::mult(temp, primes[j].value, bsk_mod[k]);
                    }
                }
                base_matrix_q_Bsk.push_back(temp);
            }
        }

        return base_matrix_q_Bsk;
    }

    // inv_punctured_prod_mod_base_array --> generate_Mi_inv

    std::vector<Data>
    Parameters::generate_base_change_matrix_m_tilde(std::vector<Modulus> primes,
                                                    Modulus mtilda, int size)
    {
        std::vector<Data> base_change_matrix_m_tilde;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data base = primes[j].value % mtilda.value;
                    temp = VALUE::mult(temp, base, mtilda);
                }
            }
            base_change_matrix_m_tilde.push_back(temp);
        }

        return base_change_matrix_m_tilde;
    }

    Data
    Parameters::generate_inv_prod_q_mod_m_tilde(std::vector<Modulus> primes,
                                                Modulus mtilda, int size)
    {
        Data inv_prod_q_mod_m_tilde = 1;
        for (int i = 0; i < size; i++)
        {
            Data base = primes[i].value % mtilda.value;
            inv_prod_q_mod_m_tilde =
                VALUE::mult(inv_prod_q_mod_m_tilde, base, mtilda);
        }

        inv_prod_q_mod_m_tilde =
            modInverse(inv_prod_q_mod_m_tilde, mtilda.value);

        return inv_prod_q_mod_m_tilde;
    }

    std::vector<Data>
    Parameters::generate_inv_m_tilde_mod_Bsk(std::vector<Modulus> bsk_mod,
                                             Modulus mtilda)
    {
        std::vector<Data> inv_m_tilde_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            inv_m_tilde_mod_Bsk.push_back(
                VALUE::modinv(mtilda.value, bsk_mod[i]));
        }

        return inv_m_tilde_mod_Bsk;
    }

    std::vector<Data>
    Parameters::generate_prod_q_mod_Bsk(std::vector<Modulus> primes,
                                        std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Data> prod_q_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                temp = VALUE::mult(temp, primes[j].value, bsk_mod[i]);
            }
            prod_q_mod_Bsk.push_back(temp);
        }

        return prod_q_mod_Bsk;
    }

    std::vector<Data> Parameters::generate_inv_prod_q_mod_Bsk(
        std::vector<Modulus> primes, std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Data> inv_prod_q_mod_Bsk;
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                temp = VALUE::mult(temp, primes[j].value, bsk_mod[i]);
            }
            inv_prod_q_mod_Bsk.push_back(VALUE::modinv(temp, bsk_mod[i]));
        }

        return inv_prod_q_mod_Bsk;
    }

    std::vector<Data> Parameters::generate_base_matrix_Bsk_q(
        std::vector<Modulus> primes, std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Data> base_matrix_Bsk_q;
        for (int k = 0; k < size; k++)
        {
            for (int i = 0; i < bsk_mod.size() - 1; i++)
            {
                Data temp = 1;
                for (int j = 0; j < bsk_mod.size() - 1; j++)
                {
                    if (i != j)
                    {
                        Data base = bsk_mod[j].value % primes[k].value;
                        temp = VALUE::mult(temp, base, primes[k]);
                    }
                }
                base_matrix_Bsk_q.push_back(temp);
            }
        }

        return base_matrix_Bsk_q;
    }

    std::vector<Data>
    Parameters::generate_base_change_matrix_msk(std::vector<Modulus> bsk_mod)
    {
        std::vector<Data> base_change_matrix_msk;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            Data temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                if (i != j)
                {
                    temp = VALUE::mult(temp, bsk_mod[j].value,
                                       bsk_mod[bsk_mod.size() - 1]);
                }
            }
            base_change_matrix_msk.push_back(temp);
        }

        return base_change_matrix_msk;
    }

    std::vector<Data> Parameters::generate_inv_punctured_prod_mod_B_array(
        std::vector<Modulus> bsk_mod)
    {
        std::vector<Data> inv_punctured_prod_mod_B_array;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            Data temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                if (i != j)
                {
                    temp = VALUE::mult(temp, bsk_mod[j].value, bsk_mod[i]);
                }
            }
            inv_punctured_prod_mod_B_array.push_back(
                VALUE::modinv(temp, bsk_mod[i]));
        }

        return inv_punctured_prod_mod_B_array;
    }

    Data Parameters::generate_inv_prod_B_mod_m_sk(std::vector<Modulus> bsk_mod)
    {
        Data inv_prod_B_mod_m_sk = 1;
        for (int i = 0; i < bsk_mod.size() - 1; i++)
        {
            inv_prod_B_mod_m_sk =
                VALUE::mult(inv_prod_B_mod_m_sk, bsk_mod[i].value,
                            bsk_mod[bsk_mod.size() - 1]);
        }

        inv_prod_B_mod_m_sk =
            VALUE::modinv(inv_prod_B_mod_m_sk, bsk_mod[bsk_mod.size() - 1]);

        return inv_prod_B_mod_m_sk;
    }

    std::vector<Data>
    Parameters::generate_prod_B_mod_q(std::vector<Modulus> primes,
                                      std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Data> prod_B_mod_q;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < bsk_mod.size() - 1; j++)
            {
                Data base = bsk_mod[j].value % primes[i].value;
                temp = VALUE::mult(temp, base, primes[i]);
            }
            prod_B_mod_q.push_back(temp);
        }

        return prod_B_mod_q;
    }

    std::vector<Modulus> Parameters::generate_q_Bsk_merge_modulus(
        std::vector<Modulus> primes, std::vector<Modulus> bsk_mod, int size)
    {
        std::vector<Modulus> q_Bsk_merge_modulus;
        for (int i = 0; i < size; i++)
        {
            q_Bsk_merge_modulus.push_back(primes[i]);
        }
        for (int i = 0; i < bsk_mod.size(); i++)
        {
            q_Bsk_merge_modulus.push_back(bsk_mod[i]);
        }

        return q_Bsk_merge_modulus;
    }

    std::vector<Data> Parameters::generate_q_Bsk_merge_root(
        std::vector<Data> primes_psi, std::vector<Data> bsk_mod_psi, int size)
    {
        std::vector<Data> q_Bsk_merge_psi;
        for (int i = 0; i < size; i++)
        {
            q_Bsk_merge_psi.push_back(primes_psi[i]);
        }
        for (int i = 0; i < bsk_mod_psi.size(); i++)
        {
            q_Bsk_merge_psi.push_back(bsk_mod_psi[i]);
        }

        return q_Bsk_merge_psi;
    }

    std::vector<Data> Parameters::generate_Qi_t(std::vector<Modulus> primes,
                                                Modulus& plain_mod, int size)
    {
        std::vector<Data> Qi_t;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data mod_ = primes[j].value % plain_mod.value;
                    temp = VALUE::mult(temp, mod_, plain_mod);
                }
            }
            Qi_t.push_back(temp);
        }

        return Qi_t;
    }

    std::vector<Data> Parameters::generate_Qi_gamma(std::vector<Modulus> primes,
                                                    Modulus& gamma, int size)
    {
        std::vector<Data> Qi_gamma;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data mod = primes[j].value % gamma.value;
                    temp = VALUE::mult(temp, mod, gamma);
                }
            }
            Qi_gamma.push_back(temp);
        }

        return Qi_gamma;
    }

    // use generate_Mi_inv
    std::vector<Data>
    Parameters::generate_Qi_inverse(std::vector<Modulus> primes, int size)
    {
        std::vector<Data> Qi_inverse;
        for (int i = 0; i < size; i++)
        {
            Data temp = 1;
            for (int j = 0; j < size; j++)
            {
                if (i != j)
                {
                    Data mod_ = primes[j].value % primes[i].value;
                    Data inv_ = VALUE::modinv(mod_, primes[i]);
                    temp = VALUE::mult(temp, inv_, primes[i]);
                }
            }
            Qi_inverse.push_back(temp);
        }

        return Qi_inverse;
    }

    Data Parameters::generate_mulq_inv_t(std::vector<Modulus> primes,
                                         Modulus& plain_mod, int size)
    {
        Data mulq_inv_t = 1;
        for (int i = 0; i < size; i++)
        {
            Data mod_ = primes[i].value % plain_mod.value;
            Data inv_ = VALUE::modinv(mod_, plain_mod);
            mulq_inv_t = VALUE::mult(mulq_inv_t, inv_, plain_mod);
        }

        mulq_inv_t = plain_mod.value - mulq_inv_t;

        return mulq_inv_t;
    }

    Data Parameters::generate_mulq_inv_gamma(std::vector<Modulus> primes,
                                             Modulus& gamma, int size)
    {
        Data mulq_inv_gamma = 1;
        for (int i = 0; i < size; i++)
        {
            Data mod_ = primes[i].value % gamma.value;
            Data inv_ = VALUE::modinv(mod_, gamma);
            mulq_inv_gamma = VALUE::mult(mulq_inv_gamma, inv_, gamma);
        }

        mulq_inv_gamma = gamma.value - mulq_inv_gamma;

        return mulq_inv_gamma;
    }

    Data Parameters::generate_inv_gamma(Modulus& plain_mod, Modulus& gamma)
    {
        Data mod_ = gamma.value % plain_mod.value;
        Data inv_gamma = VALUE::modinv(mod_, plain_mod);

        return inv_gamma;
    }

    void Parameters::print_parameters()
    {
        std::string scheme_string;
        switch (scheme_)
        {
            case heongpu::scheme_type::bfv:
                scheme_string = "BFV";
                break;
            case heongpu::scheme_type::ckks:
                scheme_string = "CKKS";
                break;
            default:
                throw std::runtime_error("invalid scheme");
        }

        std::cout
            << "==== HEonGPU a GPU Based Homomorphic Encryption Library ====\n"
            << std::endl;
        std::cout << "Encryption parameters:" << std::endl;
        std::cout << "-->   scheme: " << scheme_string << std::endl;
        std::cout << "-->   poly_modulus_degree: " << n << std::endl;
        std::cout << "-->   Q_tilta size: Q( ";

        for (std::size_t i = 0; i < Q_mod_bit_sizes_.size() - 1; i++)
        {
            std::cout << Q_mod_bit_sizes_[i] << " + ";
        }
        std::cout << Q_mod_bit_sizes_.back();
        std::cout << " ) + P( ";
        for (std::size_t i = 0; i < P_mod_bit_sizes_.size() - 1; i++)
        {
            std::cout << P_mod_bit_sizes_[i] << " + ";
        }
        std::cout << P_mod_bit_sizes_.back();
        std::cout << " ) bits" << std::endl;

        if (scheme_ == heongpu::scheme_type::bfv)
        {
            std::cout << "-->   plain_modulus: " << plain_modulus_.value
                      << std::endl;
        }

        std::cout << std::endl;
    }

    Parameters::Parameters(const scheme_type scheme,
                           const keyswitching_type ks_type,
                           const sec_level_type sec_level)
    {
        if (!coeff_modulus_specified)
        {
            switch (static_cast<int>(scheme))
            {
                case 1:
                    scheme_ = scheme_type::bfv;
                    break;
                case 2:
                    scheme_ = scheme_type::ckks;
                    break;
                default:
                    throw std::logic_error("invalid scheme type");
                    break;
            }

            sec_level_ = sec_level;

            if (ks_type == keyswitching_type::NONE)
            {
                throw std::logic_error("keyswitching_type can not be NONE!");
            }
            keyswitching_type_ = ks_type;
        }
        else
        {
            throw std::logic_error("parameters cannot be changed after the "
                                   "coeff_modulus is specified!");
        }
    }

    void Parameters::set_poly_modulus_degree(size_t poly_modulus_degree)
    {
        if (!coeff_modulus_specified)
        {
            if (!is_power_of_two(poly_modulus_degree))
            {
                throw std::logic_error(
                    "poly_modulus_degree have to be power of two");
            }

            if ((poly_modulus_degree > MAX_POLY_DEGREE) ||
                (poly_modulus_degree < MIN_POLY_DEGREE))
            {
                throw std::logic_error("poly_modulus_degree is not supported");
            }

            n = poly_modulus_degree;
            n_power = int(log2l(n));
        }
        else
        {
            throw std::logic_error("poly_modulus_degree cannot be changed "
                                   "after the coeff_modulus is specified!");
        }
    }

    bool Parameters::check_coeffs(const std::vector<int>& log_Q_bases_bit_sizes,
                                  const std::vector<int>& log_P_bases_bit_sizes)
    {
        int P_size = log_P_bases_bit_sizes.size();

        int total_P_modulus = 0; // TODO: calculate it with prod
        for (int i = 0; i < P_size; i++)
        {
            total_P_modulus = total_P_modulus + log_P_bases_bit_sizes[i];
        }

        int Q_size = log_Q_bases_bit_sizes.size();

        int remainder = Q_size % P_size;
        int quotient = Q_size / P_size;

        int counter = 0;
        for (int i = 0; i < quotient; i++)
        {
            int pair_sum = 0;
            for (int j = 0; j < P_size; j++)
            {
                pair_sum = pair_sum + log_Q_bases_bit_sizes[counter];
                counter++;
            }

            if (pair_sum > total_P_modulus)
            {
                return false;
            }
        }

        int pair_sum = 0;
        for (int j = 0; j < remainder; j++)
        {
            pair_sum = pair_sum + log_Q_bases_bit_sizes[counter];
            counter++;
        }

        if (pair_sum > total_P_modulus)
        {
            return false;
        }

        return true;
    }

    void
    Parameters::set_coeff_modulus(const std::vector<int>& log_Q_bases_bit_sizes,
                                  const std::vector<int>& log_P_bases_bit_sizes)
    {
        if (!context_generated)
        {
            if ((log_P_bases_bit_sizes.size() > 1) &&
                (keyswitching_type_ ==
                 keyswitching_type::KEYSWITCHING_METHOD_I))
            {
                throw std::logic_error("log_P_bases_bit_sizes cannot be higher "
                                       "than 1 for KEYSWITCHING_METHOD_I!");
            }

            if (!check_coeffs(log_Q_bases_bit_sizes, log_P_bases_bit_sizes))
            {
                throw std::logic_error(
                    "invalid parameters, P should be bigger than Q pairs!");
            }

            // Q' = Q x P
            Qprime_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            Qprime_mod_bit_sizes_.insert(Qprime_mod_bit_sizes_.end(),
                                         log_P_bases_bit_sizes.begin(),
                                         log_P_bases_bit_sizes.end());

            Q_mod_bit_sizes_ = log_Q_bases_bit_sizes;
            P_mod_bit_sizes_ = log_P_bases_bit_sizes;

            total_coeff_bit_count = 0; // TODO: calculate it with prod
            for (int i = 0; i < Qprime_mod_bit_sizes_.size(); i++)
            {
                total_coeff_bit_count =
                    total_coeff_bit_count + Qprime_mod_bit_sizes_[i];
            }

            int max_coeff_bit_count = 0;
            switch (sec_level_)
            {
                case sec_level_type::none:
                    break;
                case sec_level_type::sec128:
                    max_coeff_bit_count = heongpu_128bit_std_parms(n);
                    break;
                case sec_level_type::sec192:
                    max_coeff_bit_count = heongpu_192bit_std_parms(n);
                    break;
                case sec_level_type::sec256:
                    max_coeff_bit_count = heongpu_256bit_std_parms(n);
                    break;
                default:
                    throw std::runtime_error("invalid security level");
                    break;
            }

            if ((max_coeff_bit_count < total_coeff_bit_count) &&
                (sec_level_ != sec_level_type::none))
            {
                throw std::runtime_error(
                    "parameters do not align with the security recommendations "
                    "provided by the lattice-estimator");
            }

            // Q' bases size
            Q_prime_size = Qprime_mod_bit_sizes_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = log_Q_bases_bit_sizes.size();

            // P bases size
            P_size = Q_prime_size - Q_size;

            prime_vector =
                generate_primes(n,
                                Qprime_mod_bit_sizes_); // prime_vector

            for (int i = 0; i < prime_vector.size(); i++)
            {
                base_q.push_back(prime_vector[i].value);
            }

            coeff_modulus_specified = true;
        }
        else
        {
            throw std::logic_error("coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void Parameters::set_default_coeff_modulus(int P_modulus_size)
    {
        if (scheme_ != scheme_type::bfv)
        {
            throw std::logic_error(
                "Only BFV scheme has default coeff_modulus!");
        }

        if (!context_generated)
        {
            if ((P_modulus_size > 1) &&
                (keyswitching_type_ ==
                 keyswitching_type::KEYSWITCHING_METHOD_I))
            {
                throw std::logic_error("log_P_bases_bit_sizes cannot be higher "
                                       "than 1 for KEYSWITCHING_METHOD_I!");
            }

            total_coeff_bit_count = 0;
            switch (sec_level_)
            {
                case sec_level_type::sec128:
                {
                    total_coeff_bit_count = heongpu_128bit_std_parms(n);
                    prime_vector =
                        defaultparams::get_128bit_sec_modulus().at(n);

                    int size = prime_vector.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }
                }
                break;
                case sec_level_type::sec192:
                {
                    total_coeff_bit_count = heongpu_192bit_std_parms(n);
                    prime_vector =
                        defaultparams::get_192bit_sec_modulus().at(n);

                    int size = prime_vector.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }
                }
                break;
                case sec_level_type::sec256:
                {
                    total_coeff_bit_count = heongpu_256bit_std_parms(n);
                    prime_vector =
                        defaultparams::get_256bit_sec_modulus().at(n);

                    int size = prime_vector.size();
                    for (int i = 0; i < size - P_modulus_size; i++)
                    {
                        Q_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }

                    for (int i = size - P_modulus_size; i < size; i++)
                    {
                        P_mod_bit_sizes_.push_back(
                            int(log2(prime_vector[i].value) + 1));
                    }
                }
                break;
                default:
                    throw std::runtime_error("invalid security level");
                    break;
            }

            // Q' bases size
            Q_prime_size = prime_vector.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = Q_prime_size - P_modulus_size;

            // P bases size
            P_size = P_modulus_size;

            for (int i = 0; i < prime_vector.size(); i++)
            {
                base_q.push_back(prime_vector[i].value);
            }

            coeff_modulus_specified = true;
        }
        else
        {
            throw std::logic_error("coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void Parameters::set_plain_modulus(const int plain_modulus)
    {
        plain_modulus_ = Modulus(plain_modulus);
    }

    void Parameters::generate()
    {
        MemoryPool::instance().initialize();
        MemoryPool::instance().use_memory_pool(true);
        cudaDeviceSynchronize();

        // For kernel stack size
        cudaDeviceSetLimit(cudaLimitStackSize, 2048);

        modulus_ = std::make_shared<DeviceVector<Modulus>>(prime_vector);

        std::vector<Data> base_q_psi =
            generate_primitive_root_of_unity(n, prime_vector);
        std::vector<Root> Qprime_ntt_table =
            generate_ntt_table(base_q_psi, prime_vector, n_power);
        std::vector<Root> Qprime_intt_table =
            generate_intt_table(base_q_psi, prime_vector, n_power);
        std::vector<Ninverse> Qprime_n_inverse =
            generate_n_inverse(n, prime_vector);

        ntt_table_ = std::make_shared<DeviceVector<Root>>(Qprime_ntt_table);

        intt_table_ = std::make_shared<DeviceVector<Root>>(Qprime_intt_table);

        n_inverse_ = std::make_shared<DeviceVector<Ninverse>>(Qprime_n_inverse);

        std::vector<Data> last_q_modinv = generate_last_q_modinv();
        std::vector<Data> half = generate_half();
        std::vector<Data> half_mod = generate_half_mod(half);
        std::vector<Data> factor = generate_factor();

        last_q_modinv_ = std::make_shared<DeviceVector<Data>>(last_q_modinv);

        half_p_ = std::make_shared<DeviceVector<Data>>(half);

        half_mod_ = std::make_shared<DeviceVector<Data>>(half_mod);

        factor_ = std::make_shared<DeviceVector<Data>>(factor);

        if (scheme_ == scheme_type::bfv)
        {
            std::vector<Data> Mi = generate_Mi(prime_vector, Q_size);
            std::vector<Data> Mi_inv = generate_Mi_inv(prime_vector, Q_size);
            std::vector<Data> upper_half_threshold =
                generate_upper_half_threshold(prime_vector, Q_size);
            std::vector<Data> decryption_modulus =
                generate_M(prime_vector, Q_size);

            Mi_ = std::make_shared<DeviceVector<Data>>(Mi);

            Mi_inv_ = std::make_shared<DeviceVector<Data>>(Mi_inv);

            upper_half_threshold_ =
                std::make_shared<DeviceVector<Data>>(upper_half_threshold);

            decryption_modulus_ =
                std::make_shared<DeviceVector<Data>>(decryption_modulus);

            total_bit_count_ = calculate_big_integer_bit_count(
                decryption_modulus.data(), decryption_modulus.size());

            Modulus plain_mod = plain_modulus_;

            Data plain_psi = find_minimal_primitive_root(2 * n, plain_mod);

            std::vector<Root> plain_forward_table =
                generate_ntt_table({plain_psi}, {plain_mod}, n_power);
            std::vector<Root> plain_inverse_table =
                generate_intt_table({plain_psi}, {plain_mod}, n_power);
            Data n_ = n;
            std::vector<Ninverse> plain_n_inverse;
            std::vector<Modulus> plain_mod2;
            plain_n_inverse.push_back(VALUE::modinv(n_, plain_mod));
            plain_mod2.push_back(plain_mod);

            Data plain_upper_half_threshold = (plain_mod.value + 1) >> 1;

            std::vector<Data> plain_upper_half_increment;
            for (int i = 0; i < Q_size; i++)
            {
                plain_upper_half_increment.push_back(prime_vector[i].value -
                                                     plain_mod.value);
            }

            Modulus m_tilde((1ULL << 32));

            Data Q_mod_t = generate_Q_mod_t(prime_vector, plain_mod, Q_size);

            std::vector<Data> coeff_div_plain_modulus =
                generate_coeff_div_plain_modulus(prime_vector, plain_mod,
                                                 Q_size);

            bsk_modulus = prime_vector.size();
            if (calculate_bit_count(plain_mod.value) + total_coeff_bit_count +
                    32 >=
                MAX_MOD_BIT_COUNT * Q_size + MAX_MOD_BIT_COUNT)
            {
                bsk_modulus++;
            }

            std::vector<Modulus> base_Bsk_mod = generate_internal_primes(
                n,
                bsk_modulus + 1); // extra for gamma parameter

            Modulus gamma_mod = base_Bsk_mod[bsk_modulus];
            base_Bsk_mod.pop_back();

            std::vector<Data> base_Bsk_psi =
                generate_primitive_root_of_unity(n, base_Bsk_mod);
            std::vector<Root> Bsk_ntt_table =
                generate_ntt_table(base_Bsk_psi, base_Bsk_mod, n_power);
            std::vector<Root> Bsk_intt_table =
                generate_intt_table(base_Bsk_psi, base_Bsk_mod, n_power);
            std::vector<Ninverse> Bsk_n_inverse =
                generate_n_inverse(n, base_Bsk_mod);

            base_Bsk_ = std::make_shared<DeviceVector<Modulus>>(base_Bsk_mod);

            bsk_ntt_tables_ =
                std::make_shared<DeviceVector<Root>>(Bsk_ntt_table);

            bsk_intt_tables_ =
                std::make_shared<DeviceVector<Root>>(Bsk_intt_table);

            bsk_n_inverse_ =
                std::make_shared<DeviceVector<Ninverse>>(Bsk_n_inverse);

            std::vector<Data> base_matrix_q_Bsk =
                generate_base_matrix_q_Bsk(prime_vector, base_Bsk_mod, Q_size);

            std::vector<Data> inv_punctured_prod_mod_base_array =
                generate_Mi_inv(prime_vector, Q_size);

            std::vector<Data> base_change_matrix_m_tilde =
                generate_base_change_matrix_m_tilde(prime_vector, m_tilde,
                                                    Q_size);

            Data inv_prod_q_mod_m_tilde =
                generate_inv_prod_q_mod_m_tilde(prime_vector, m_tilde, Q_size);

            std::vector<Data> inv_m_tilde_mod_Bsk =
                generate_inv_m_tilde_mod_Bsk(base_Bsk_mod, m_tilde);

            std::vector<Data> prod_q_mod_Bsk =
                generate_prod_q_mod_Bsk(prime_vector, base_Bsk_mod, Q_size);

            std::vector<Data> inv_prod_q_mod_Bsk =
                generate_inv_prod_q_mod_Bsk(prime_vector, base_Bsk_mod, Q_size);

            std::vector<Data> base_matrix_Bsk_q =
                generate_base_matrix_Bsk_q(prime_vector, base_Bsk_mod, Q_size);

            std::vector<Data> base_change_matrix_msk =
                generate_base_change_matrix_msk(base_Bsk_mod);

            std::vector<Data> inv_punctured_prod_mod_B_array =
                generate_inv_punctured_prod_mod_B_array(base_Bsk_mod);

            Data inv_prod_B_mod_m_sk =
                generate_inv_prod_B_mod_m_sk(base_Bsk_mod);

            std::vector<Data> prod_B_mod_q =
                generate_prod_B_mod_q(prime_vector, base_Bsk_mod, Q_size);

            std::vector<Modulus> q_Bsk_merge_modulus =
                generate_q_Bsk_merge_modulus(prime_vector, base_Bsk_mod,
                                             Q_size);

            std::vector<Data> q_Bsk_merge_root =
                generate_q_Bsk_merge_root(base_q_psi, base_Bsk_psi, Q_size);

            std::vector<Root> q_Bsk_forward_tables = generate_ntt_table(
                q_Bsk_merge_root, q_Bsk_merge_modulus, n_power);
            std::vector<Root> q_Bsk_inverse_tables = generate_intt_table(
                q_Bsk_merge_root, q_Bsk_merge_modulus, n_power);
            std::vector<Ninverse> q_Bsk_n_inverse =
                generate_n_inverse(n, q_Bsk_merge_modulus);

            std::vector<Data> Qi_t =
                generate_Qi_t(prime_vector, plain_mod, Q_size);

            std::vector<Data> Qi_gamma =
                generate_Qi_gamma(prime_vector, gamma_mod, Q_size);

            std::vector<Data> Qi_inverse =
                generate_Qi_inverse(prime_vector, Q_size);

            Data mulq_inv_t =
                generate_mulq_inv_t(prime_vector, plain_mod, Q_size);

            Data mulq_inv_gamma =
                generate_mulq_inv_gamma(prime_vector, gamma_mod, Q_size);

            Data inv_gamma = generate_inv_gamma(plain_mod, gamma_mod);

            m_tilde_ = m_tilde;

            base_change_matrix_Bsk_ =
                std::make_shared<DeviceVector<Data>>(base_matrix_q_Bsk);

            inv_punctured_prod_mod_base_array_ =
                std::make_shared<DeviceVector<Data>>(
                    inv_punctured_prod_mod_base_array);

            base_change_matrix_m_tilde_ = std::make_shared<DeviceVector<Data>>(
                base_change_matrix_m_tilde);

            inv_prod_q_mod_m_tilde_ = inv_prod_q_mod_m_tilde;

            inv_m_tilde_mod_Bsk_ =
                std::make_shared<DeviceVector<Data>>(inv_m_tilde_mod_Bsk);

            prod_q_mod_Bsk_ =
                std::make_shared<DeviceVector<Data>>(prod_q_mod_Bsk);

            inv_prod_q_mod_Bsk_ =
                std::make_shared<DeviceVector<Data>>(inv_prod_q_mod_Bsk);

            base_change_matrix_q_ =
                std::make_shared<DeviceVector<Data>>(base_matrix_Bsk_q);

            base_change_matrix_msk_ =
                std::make_shared<DeviceVector<Data>>(base_change_matrix_msk);

            inv_punctured_prod_mod_B_array_ =
                std::make_shared<DeviceVector<Data>>(
                    inv_punctured_prod_mod_B_array);

            inv_prod_B_mod_m_sk_ = inv_prod_B_mod_m_sk;

            prod_B_mod_q_ = std::make_shared<DeviceVector<Data>>(prod_B_mod_q);

            q_Bsk_merge_modulus_ =
                std::make_shared<DeviceVector<Modulus>>(q_Bsk_merge_modulus);

            q_Bsk_merge_ntt_tables_ =
                std::make_shared<DeviceVector<Root>>(q_Bsk_forward_tables);

            q_Bsk_merge_intt_tables_ =
                std::make_shared<DeviceVector<Root>>(q_Bsk_inverse_tables);

            q_Bsk_n_inverse_ =
                std::make_shared<DeviceVector<Ninverse>>(q_Bsk_n_inverse);

            plain_modulus2_ =
                std::make_shared<DeviceVector<Modulus>>(plain_mod2);

            n_plain_inverse_ =
                std::make_shared<DeviceVector<Ninverse>>(plain_n_inverse);

            plain_ntt_tables_ =
                std::make_shared<DeviceVector<Root>>(plain_forward_table);

            plain_intt_tables_ =
                std::make_shared<DeviceVector<Root>>(plain_inverse_table);

            gamma_ = gamma_mod;

            coeeff_div_plainmod_ =
                std::make_shared<DeviceVector<Data>>(coeff_div_plain_modulus);

            Q_mod_t_ = Q_mod_t;

            upper_threshold_ = plain_upper_half_threshold;

            upper_halfincrement_ = std::make_shared<DeviceVector<Data>>(
                plain_upper_half_increment);

            Qi_t_ = std::make_shared<DeviceVector<Data>>(Qi_t);

            Qi_gamma_ = std::make_shared<DeviceVector<Data>>(Qi_gamma);

            Qi_inverse_ = std::make_shared<DeviceVector<Data>>(Qi_inverse);

            mulq_inv_t_ = mulq_inv_t;
            mulq_inv_gamma_ = mulq_inv_gamma;
            inv_gamma_ = inv_gamma;
        }
        else if (scheme_ == scheme_type::ckks)
        {
            // For Rescale parameters for all depth
            std::vector<Data> rescale_last_q_modinv;
            std::vector<Data> rescaled_half_mod;
            std::vector<Data> rescaled_half;
            for (int j = 0; j < (Q_size - 1); j++)
            {
                int inner = (Q_size - 1) - j;
                rescaled_half.push_back(prime_vector[inner].value >> 1);
                for (int i = 0; i < inner; i++)
                {
                    Data temp_ =
                        prime_vector[inner].value % prime_vector[i].value;
                    rescale_last_q_modinv.push_back(
                        VALUE::modinv(temp_, prime_vector[i]));
                    rescaled_half_mod.push_back(rescaled_half[j] %
                                                prime_vector[i].value);
                }
            }

            rescaled_last_q_modinv_ =
                std::make_shared<DeviceVector<Data>>(rescale_last_q_modinv);

            rescaled_half_mod_ =
                std::make_shared<DeviceVector<Data>>(rescaled_half_mod);

            rescaled_half_ =
                std::make_shared<DeviceVector<Data>>(rescaled_half);

            std::vector<Data> Mi;
            std::vector<Data> Mi_inv;
            std::vector<Data> upper_half_threshold;
            std::vector<Data> decryption_modulus;

            for (int i = 0; i < Q_size; i++)
            {
                int depth_Q_size = Q_size - i;

                // Mi
                std::vector<Data> Mi_inner =
                    generate_Mi(prime_vector, depth_Q_size);
                for (int j = 0; j < depth_Q_size * depth_Q_size; j++)
                {
                    Mi.push_back(Mi_inner[j]);
                }

                // Mi_inv
                std::vector<Data> Mi_inv_inner =
                    generate_Mi_inv(prime_vector, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    Mi_inv.push_back(Mi_inv_inner[j]);
                }

                // upper_half_threshold
                std::vector<Data> upper_half_threshold_inner =
                    generate_upper_half_threshold(prime_vector, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    upper_half_threshold.push_back(
                        upper_half_threshold_inner[j]);
                }

                // decryption_modulus
                std::vector<Data> M_inner =
                    generate_M(prime_vector, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    decryption_modulus.push_back(M_inner[j]);
                }
            }

            Mi_ = std::make_shared<DeviceVector<Data>>(Mi);

            Mi_inv_ = std::make_shared<DeviceVector<Data>>(Mi_inv);

            upper_half_threshold_ =
                std::make_shared<DeviceVector<Data>>(upper_half_threshold);

            decryption_modulus_ =
                std::make_shared<DeviceVector<Data>>(decryption_modulus);

            // prime_location_leveled
            std::vector<int> prime_loc;
            int counter = Q_size;
            for (int i = 0; i < Q_size - 1; i++)
            {
                for (int j = 0; j < counter; j++)
                {
                    prime_loc.push_back(j);
                }
                counter--;
                for (int j = 0; j < P_size; j++)
                {
                    prime_loc.push_back(Q_size + j);
                }
            }

            prime_location_leveled =
                std::make_shared<DeviceVector<int>>(prime_loc);
        }

        generate_keyswitching_params(keyswitching_type_);
    }

    void Parameters::generate_keyswitching_params(keyswitching_type type)
    {
        keyswitching_type_ = type;

        switch (static_cast<int>(keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
                // Deafult
                break;
            case 2: // KEYSWITCHING_METHOD_II

                if (scheme_ == scheme_type::bfv)
                {
                    KeySwitchParameterGenerator pool(n, base_q, P_size, scheme_,
                                                     keyswitching_type_);

                    m = pool.m;
                    l = pool.first_Q_;
                    l_tilda = pool.first_Qtilda_;

                    d = pool.d_;

                    std::vector<Data> base_change_matrix_D_to_Q_tilda_inner =
                        pool.base_change_matrix_D_to_Qtilda();
                    base_change_matrix_D_to_Q_tilda_ =
                        std::make_shared<DeviceVector<Data>>(
                            base_change_matrix_D_to_Q_tilda_inner);

                    std::vector<Data> Mi_inv_D_to_Q_tilda_inner =
                        pool.Mi_inv_D_to_Qtilda();
                    Mi_inv_D_to_Q_tilda_ = std::make_shared<DeviceVector<Data>>(
                        Mi_inv_D_to_Q_tilda_inner);

                    std::vector<Data> prod_D_to_Q_tilda_inner =
                        pool.prod_D_to_Qtilda();
                    prod_D_to_Q_tilda_ = std::make_shared<DeviceVector<Data>>(
                        prod_D_to_Q_tilda_inner);

                    std::vector<int> I_j_inner = pool.I_j();
                    I_j_ = std::make_shared<DeviceVector<int>>(I_j_inner);

                    std::vector<int> I_location_inner = pool.I_location();
                    I_location_ =
                        std::make_shared<DeviceVector<int>>(I_location_inner);

                    std::vector<int> Sk_pair_inner = pool.sk_pair();
                    Sk_pair_ =
                        std::make_shared<DeviceVector<int>>(Sk_pair_inner);
                }
                else if (scheme_ == scheme_type::ckks)
                {
                    KeySwitchParameterGenerator pool_ckks(
                        n, base_q, P_size, scheme_, keyswitching_type_);

                    m_leveled = pool_ckks.m;
                    l_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_Q_);
                    l_tilda_leveled = std::make_shared<std::vector<int>>(
                        pool_ckks.level_Qtilda_);

                    d_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_d_);

                    std::vector<std::vector<Data>>
                        base_change_matrix_D_to_Qtilda_vec =
                            pool_ckks.level_base_change_matrix_D_to_Qtilda();

                    std::vector<std::vector<Data>> Mi_inv_D_to_Qtilda_vec =
                        pool_ckks.level_Mi_inv_D_to_Qtilda();

                    std::vector<std::vector<Data>> prod_D_to_Qtilda_vec =
                        pool_ckks.level_prod_D_to_Qtilda();

                    std::vector<std::vector<int>> I_j_vec =
                        pool_ckks.level_I_j();

                    std::vector<std::vector<int>> I_location_vec =
                        pool_ckks.level_I_location();

                    std::vector<std::vector<int>> Sk_pair_new_vec =
                        pool_ckks.level_sk_pair();

                    base_change_matrix_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    Mi_inv_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    prod_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    I_j_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    I_location_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    Sk_pair_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();

                    for (int pool_lp = 0;
                         pool_lp < base_change_matrix_D_to_Qtilda_vec.size();
                         pool_lp++)
                    {
                        DeviceVector<Data>
                            base_change_matrix_D_to_Qtilda_leveled_inner(
                                base_change_matrix_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data> Mi_inv_D_to_Qtilda_leveled_inner(
                            Mi_inv_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data> prod_D_to_Qtilda_leveled_inner(
                            prod_D_to_Qtilda_vec[pool_lp]);

                        base_change_matrix_D_to_Qtilda_leveled->push_back(
                            std::move(
                                base_change_matrix_D_to_Qtilda_leveled_inner));
                        Mi_inv_D_to_Qtilda_leveled->push_back(
                            std::move(Mi_inv_D_to_Qtilda_leveled_inner));
                        prod_D_to_Qtilda_leveled->push_back(
                            std::move(prod_D_to_Qtilda_leveled_inner));

                        DeviceVector<int> I_j_vec_inner(I_j_vec[pool_lp]);
                        DeviceVector<int> I_location_vec_inner(
                            I_location_vec[pool_lp]);
                        DeviceVector<int> Sk_pair_new_vec_inner(
                            Sk_pair_new_vec[pool_lp]);

                        I_j_leveled->push_back(std::move(I_j_vec_inner));
                        I_location_leveled->push_back(
                            std::move(I_location_vec_inner));
                        Sk_pair_leveled->push_back(
                            std::move(Sk_pair_new_vec_inner));
                    }
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }
                break;
            case 3: // KEYSWITCHING_METHOD_III

                if (scheme_ == scheme_type::bfv)
                {
                    KeySwitchParameterGenerator pool(n, base_q, P_size, scheme_,
                                                     keyswitching_type_);

                    m = pool.m;
                    l = pool.first_Q_;
                    l_tilda = pool.first_Qtilda_;

                    d = pool.d_;
                    d_tilda = pool.d_tilda_;
                    r_prime = pool.r_prime_;

                    std::vector<Modulus> B_prime_inner = pool.B_prime;
                    B_prime_ =
                        std::make_shared<DeviceVector<Modulus>>(B_prime_inner);

                    std::vector<Root> B_prime_ntt_tables_inner =
                        pool.B_prime_ntt_tables();
                    B_prime_ntt_tables_ = std::make_shared<DeviceVector<Root>>(
                        B_prime_ntt_tables_inner);

                    std::vector<Root> B_prime_intt_tables_inner =
                        pool.B_prime_intt_tables();
                    B_prime_intt_tables_ = std::make_shared<DeviceVector<Root>>(
                        B_prime_intt_tables_inner);

                    std::vector<Ninverse> B_prime_n_inverse_inner =
                        pool.B_prime_n_inverse();
                    B_prime_n_inverse_ =
                        std::make_shared<DeviceVector<Ninverse>>(
                            B_prime_n_inverse_inner);

                    std::vector<Data> base_change_matrix_D_to_B_inner =
                        pool.base_change_matrix_D_to_B();
                    base_change_matrix_D_to_B_ =
                        std::make_shared<DeviceVector<Data>>(
                            base_change_matrix_D_to_B_inner);

                    std::vector<Data> base_change_matrix_B_to_D_inner =
                        pool.base_change_matrix_B_to_D();
                    base_change_matrix_B_to_D_ =
                        std::make_shared<DeviceVector<Data>>(
                            base_change_matrix_B_to_D_inner);

                    std::vector<Data> Mi_inv_D_to_B_inner =
                        pool.Mi_inv_D_to_B();
                    Mi_inv_D_to_B_ = std::make_shared<DeviceVector<Data>>(
                        Mi_inv_D_to_B_inner);

                    std::vector<Data> Mi_inv_B_to_D_inner =
                        pool.Mi_inv_B_to_D();
                    Mi_inv_B_to_D_ = std::make_shared<DeviceVector<Data>>(
                        Mi_inv_B_to_D_inner);

                    std::vector<Data> prod_D_to_B_inner = pool.prod_D_to_B();
                    prod_D_to_B_ =
                        std::make_shared<DeviceVector<Data>>(prod_D_to_B_inner);

                    std::vector<Data> prod_B_to_D_inner = pool.prod_B_to_D();
                    prod_B_to_D_ =
                        std::make_shared<DeviceVector<Data>>(prod_B_to_D_inner);

                    std::vector<int> I_j_inner = pool.I_j_2();
                    I_j_ = std::make_shared<DeviceVector<int>>(I_j_inner);

                    std::vector<int> I_location_inner = pool.I_location_2();
                    I_location_ =
                        std::make_shared<DeviceVector<int>>(I_location_inner);

                    std::vector<int> sk_pair_inner = pool.sk_pair();
                    Sk_pair_ =
                        std::make_shared<DeviceVector<int>>(sk_pair_inner);
                }
                else if (scheme_ == scheme_type::ckks)
                {
                    KeySwitchParameterGenerator pool_ckks(
                        n, base_q, P_size, scheme_, keyswitching_type_);

                    m_leveled = pool_ckks.m;
                    l_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_Q_);
                    l_tilda_leveled = std::make_shared<std::vector<int>>(
                        pool_ckks.level_Qtilda_);

                    d_leveled =
                        std::make_shared<std::vector<int>>(pool_ckks.level_d_);
                    d_tilda_leveled = std::make_shared<std::vector<int>>(
                        pool_ckks.level_d_tilda_);
                    r_prime_leveled = pool_ckks.r_prime_;

                    std::vector<Modulus> B_prime_inner = pool_ckks.B_prime;
                    B_prime_leveled =
                        std::make_shared<DeviceVector<Modulus>>(B_prime_inner);

                    std::vector<Root> B_prime_ntt_tables_leveled_inner =
                        pool_ckks.B_prime_ntt_tables();
                    B_prime_ntt_tables_leveled =
                        std::make_shared<DeviceVector<Root>>(
                            B_prime_ntt_tables_leveled_inner);

                    std::vector<Root> B_prime_intt_tables_leveled_inner =
                        pool_ckks.B_prime_intt_tables();
                    B_prime_intt_tables_leveled =
                        std::make_shared<DeviceVector<Root>>(
                            B_prime_intt_tables_leveled_inner);

                    std::vector<Ninverse> B_prime_n_inverse_leveled_inner =
                        pool_ckks.B_prime_n_inverse();
                    B_prime_n_inverse_leveled =
                        std::make_shared<DeviceVector<Ninverse>>(
                            B_prime_n_inverse_leveled_inner);

                    std::vector<std::vector<Data>>
                        base_change_matrix_D_to_B_vec =
                            pool_ckks.level_base_change_matrix_D_to_B();

                    std::vector<std::vector<Data>>
                        base_change_matrix_B_to_D_vec =
                            pool_ckks.level_base_change_matrix_B_to_D();

                    std::vector<std::vector<Data>> Mi_inv_D_to_B_vec =
                        pool_ckks.level_Mi_inv_D_to_B();
                    std::vector<Data> Mi_inv_B_to_D_vec =
                        pool_ckks.level_Mi_inv_B_to_D();

                    Mi_inv_B_to_D_leveled =
                        std::make_shared<DeviceVector<Data>>(Mi_inv_B_to_D_vec);

                    std::vector<std::vector<Data>> prod_D_to_B_vec =
                        pool_ckks.level_prod_D_to_B();
                    std::vector<std::vector<Data>> prod_B_to_D_vec =
                        pool_ckks.level_prod_B_to_D();

                    std::vector<std::vector<int>> I_j_vec =
                        pool_ckks.level_I_j_2();
                    std::vector<std::vector<int>> I_location_vec =
                        pool_ckks.level_I_location_2();
                    std::vector<std::vector<int>> Sk_pair_new_vec =
                        pool_ckks.level_sk_pair();

                    base_change_matrix_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    base_change_matrix_B_to_D_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    Mi_inv_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    prod_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();
                    prod_B_to_D_leveled =
                        std::make_shared<std::vector<DeviceVector<Data>>>();

                    I_j_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    I_location_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();
                    Sk_pair_leveled =
                        std::make_shared<std::vector<DeviceVector<int>>>();

                    for (int pool_lp = 0;
                         pool_lp < base_change_matrix_D_to_B_vec.size();
                         pool_lp++)
                    {
                        DeviceVector<Data>
                            base_change_matrix_D_to_B_leveled_inner(
                                base_change_matrix_D_to_B_vec[pool_lp]);
                        DeviceVector<Data>
                            base_change_matrix_B_to_D_leveled_inner(
                                base_change_matrix_B_to_D_vec[pool_lp]);
                        DeviceVector<Data> Mi_inv_D_to_B_leveled_inner(
                            Mi_inv_D_to_B_vec[pool_lp]);
                        DeviceVector<Data> prod_D_to_B_leveled_inner(
                            prod_D_to_B_vec[pool_lp]);
                        DeviceVector<Data> prod_B_to_D_leveled_inner(
                            prod_B_to_D_vec[pool_lp]);

                        base_change_matrix_D_to_B_leveled->push_back(
                            std::move(base_change_matrix_D_to_B_leveled_inner));
                        base_change_matrix_B_to_D_leveled->push_back(
                            std::move(base_change_matrix_B_to_D_leveled_inner));
                        Mi_inv_D_to_B_leveled->push_back(
                            std::move(Mi_inv_D_to_B_leveled_inner));
                        prod_D_to_B_leveled->push_back(
                            std::move(prod_D_to_B_leveled_inner));
                        prod_B_to_D_leveled->push_back(
                            std::move(prod_B_to_D_leveled_inner));

                        DeviceVector<int> I_j_vec_inner(I_j_vec[pool_lp]);
                        DeviceVector<int> I_location_vec_inner(
                            I_location_vec[pool_lp]);
                        DeviceVector<int> Sk_pair_new_vec_inner(
                            Sk_pair_new_vec[pool_lp]);
                        I_j_leveled->push_back(std::move(I_j_vec_inner));
                        I_location_leveled->push_back(
                            std::move(I_location_vec_inner));
                        Sk_pair_leveled->push_back(
                            std::move(Sk_pair_new_vec_inner));
                    }
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }
                break;

            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__ HEStream::HEStream(Parameters context)
    {
        n = context.n;

        scheme = context.scheme_;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        bsk_modulus_count_ = context.bsk_modulus;

        d = context.d;
        d_tilda = context.d_tilda;
        r_prime = context.r_prime;

        d_leveled_ = context.d_leveled;
        d_tilda_leveled_ = context.d_tilda_leveled;
        r_prime_leveled_ = context.r_prime_leveled;

        cudaStreamCreate(&stream);

        temp_complex = DeviceVector<COMPLEX>(n, stream);

        temp_data = DeviceVector<Data>(
            (3 * n * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        temp1_enc = temp_data.data();
        temp2_enc = temp1_enc + (3 * n * Q_prime_size_);

        if (scheme == scheme_type::bfv)
        {
            temp_mul =
                DeviceVector<Data>((4 * n * (bsk_modulus_count_ + Q_size_)) +
                                   (3 * n * (bsk_modulus_count_ + Q_size_)));
            temp1_mul = temp_mul.data();
            temp2_mul = temp1_mul + (4 * n * (bsk_modulus_count_ + Q_size_));

            if (context.keyswitching_type_ ==
                keyswitching_type::KEYSWITCHING_METHOD_I)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (n * Q_size_ * Q_prime_size_) +
                    (2 * n * Q_prime_size_));
                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_II)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (n * Q_size_) +
                    (2 * n * Q_prime_size_ * d) + (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (n * Q_size_);
                temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_III)
            {
                temp_relin_new = DeviceVector<Data>(
                    (n * d * r_prime) + (2 * n * d_tilda * r_prime) +
                    (2 * n * Q_prime_size_));
                temp1_relin_new = temp_relin_new.data();
                temp2_relin_new = temp1_relin_new + (n * d * r_prime);
                temp3_relin_new = temp2_relin_new + (2 * n * d_tilda * r_prime);
            }
            else
            {
            }

            temp_rescale = DeviceVector<Data>((2 * n * Q_prime_size_) +
                                              (2 * n * Q_prime_size_));
            temp1_rescale = temp_rescale.data();
            temp2_rescale = temp1_rescale + (2 * n * Q_prime_size_);

            temp_plain_mul = DeviceVector<Data>(n * Q_size_);
            temp1_plain_mul = temp_plain_mul.data();
        }
        else if (scheme == scheme_type::ckks)
        {
            if (context.keyswitching_type_ ==
                keyswitching_type::KEYSWITCHING_METHOD_I)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (2 * n * Q_size_) +
                    (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (2 * n * Q_size_);
                temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_II)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                    (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                    (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (2 * n * Q_size_);
                temp3_rotation = temp2_rotation + (n * Q_size_);
                temp4_rotation =
                    temp3_rotation +
                    (2 * n * d_leveled_->operator[](0) * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_III)
            {
                temp_relin_new = DeviceVector<Data>(
                    (n * d_leveled_->operator[](0) * r_prime_leveled_) +
                    (2 * n * d_tilda_leveled_->operator[](0) *
                     r_prime_leveled_) +
                    (2 * n * Q_prime_size_));
                temp1_relin_new = temp_relin_new.data();
                temp2_relin_new =
                    temp1_relin_new +
                    (n * d_leveled_->operator[](0) * r_prime_leveled_);
                temp3_relin_new =
                    temp2_relin_new + (2 * n * d_tilda_leveled_->operator[](0) *
                                       r_prime_leveled_);
            }
            else
            {
            }

            temp_rescale = DeviceVector<Data>((2 * n * Q_prime_size_) +
                                              (2 * n * Q_prime_size_));
            temp1_rescale = temp_rescale.data();
            temp2_rescale = temp1_rescale + (2 * n * Q_prime_size_);

            // cudaMalloc(&temp_mod_drop, n * Q_size_ * sizeof(Data));
            temp_mod_drop_ = DeviceVector<Data>(n * Q_size_);
            temp_mod_drop = temp_mod_drop_.data();
        }
        else
        {
            throw std::invalid_argument("Invalid Scheme Type");
        }
    }

} // namespace heongpu