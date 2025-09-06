// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ckks/context.cuh"

namespace heongpu
{
    HEContext<Scheme::CKKS>::HEContext(const keyswitching_type ks_type,
                                       const sec_level_type sec_level)
    {
        if (!coeff_modulus_specified_)
        {
            scheme_ = scheme_type::ckks;
            sec_level_ = sec_level;

            if (ks_type == keyswitching_type::NONE)
            {
                throw std::logic_error("Key switching type can not be NONE!");
            }
            keyswitching_type_ = ks_type;
        }
        else
        {
            throw std::logic_error("Parameters cannot be changed after the "
                                   "coeff_modulus is specified!");
        }
    }

    void
    HEContext<Scheme::CKKS>::set_poly_modulus_degree(size_t poly_modulus_degree)
    {
        if ((!coeff_modulus_specified_) && (!poly_modulus_degree_specified_))
        {
            if (!is_power_of_two(poly_modulus_degree))
            {
                throw std::logic_error(
                    "Poly modulus degree have to be power of two");
            }

            if ((poly_modulus_degree > MAX_POLY_DEGREE) ||
                (poly_modulus_degree < MIN_POLY_DEGREE))
            {
                throw std::logic_error("Poly modulus degree is not supported");
            }

            n = poly_modulus_degree;
            n_power = int(log2l(n));

            poly_modulus_degree_specified_ = true;
        }
        else
        {
            throw std::logic_error("Poly modulus degree cannot be changed "
                                   "after the coeff_modulus is specified!");
        }
    }

    void HEContext<Scheme::CKKS>::set_coeff_modulus_bit_sizes(
        const std::vector<int>& log_Q_bases_bit_sizes,
        const std::vector<int>& log_P_bases_bit_sizes)
    {
        if ((!coeff_modulus_specified_) && (!context_generated_) &&
            (poly_modulus_degree_specified_))
        {
            if ((log_P_bases_bit_sizes.size() > 1) &&
                (keyswitching_type_ ==
                 keyswitching_type::KEYSWITCHING_METHOD_I))
            {
                throw std::logic_error("log_P_bases_bit_sizes cannot be higher "
                                       "than 1 for KEYSWITCHING_METHOD_I!");
            }

            if ((log_P_bases_bit_sizes.size() < 2) &&
                (keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_II ||
                 keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_III))
            {
                throw std::logic_error("log_P_bases_bit_sizes cannot be lower "
                                       "than 2 for KEYSWITCHING_METHOD_II and!"
                                       "KEYSWITCHING_METHOD_III");
            }

            if (!coefficient_validator(log_Q_bases_bit_sizes,
                                       log_P_bases_bit_sizes))
            {
                throw std::logic_error("P should be bigger than Q pairs!");
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
                    throw std::runtime_error("Invalid security level");
                    break;
            }

            if ((max_coeff_bit_count < total_coeff_bit_count) &&
                (sec_level_ != sec_level_type::none))
            {
                throw std::runtime_error(
                    "Parameters do not align with the security recommendations "
                    "provided by the lattice-estimator");
            }

            // Q' bases size
            Q_prime_size = Qprime_mod_bit_sizes_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = log_Q_bases_bit_sizes.size();

            // P bases size
            P_size = Q_prime_size - Q_size;

            prime_vector_ =
                generate_primes(n,
                                Qprime_mod_bit_sizes_); // prime_vector_

            for (int i = 0; i < prime_vector_.size(); i++)
            {
                base_q.push_back(prime_vector_[i].value);
            }

            coeff_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("Coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContext<Scheme::CKKS>::set_coeff_modulus_values(
        const std::vector<Data64>& log_Q_bases,
        const std::vector<Data64>& log_P_bases)
    {
        if ((!coeff_modulus_specified_) && (!context_generated_) &&
            (poly_modulus_degree_specified_))
        {
            std::vector<int> log_Q_bases_bit_sizes;
            for (int i = 0; i < log_Q_bases.size(); i++)
            {
                log_Q_bases_bit_sizes.push_back(
                    calculate_bit_size(log_Q_bases[i]));
            }

            std::vector<int> log_P_bases_bit_sizes;
            for (int i = 0; i < log_P_bases.size(); i++)
            {
                log_P_bases_bit_sizes.push_back(
                    calculate_bit_size(log_P_bases[i]));
            }

            if ((log_P_bases_bit_sizes.size() > 1) &&
                (keyswitching_type_ ==
                 keyswitching_type::KEYSWITCHING_METHOD_I))
            {
                throw std::logic_error("log_P_bases_bit_sizes cannot be higher "
                                       "than 1 for KEYSWITCHING_METHOD_I!");
            }

            if (!coefficient_validator(log_Q_bases_bit_sizes,
                                       log_P_bases_bit_sizes))
            {
                throw std::logic_error(
                    "Invalid parameters, P should be bigger than Q pairs!");
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
                    throw std::runtime_error("Invalid security level");
                    break;
            }

            if ((max_coeff_bit_count < total_coeff_bit_count) &&
                (sec_level_ != sec_level_type::none))
            {
                throw std::runtime_error(
                    "Parameters do not align with the security recommendations "
                    "provided by the lattice-estimator");
            }

            // Q' bases size
            Q_prime_size = Qprime_mod_bit_sizes_.size();
            coeff_modulus = Q_prime_size; // not required

            // Q bases size
            Q_size = log_Q_bases_bit_sizes.size();

            // P bases size
            P_size = Q_prime_size - Q_size;

            for (int i = 0; i < log_Q_bases.size(); i++)
            {
                Modulus64 mod_in(log_Q_bases[i]);
                prime_vector_.push_back(mod_in);
            }

            for (int i = 0; i < log_P_bases.size(); i++)
            {
                Modulus64 mod_in(log_P_bases[i]);
                prime_vector_.push_back(mod_in);
            }

            for (int i = 0; i < prime_vector_.size(); i++)
            {
                base_q.push_back(prime_vector_[i].value);
            }

            coeff_modulus_specified_ = true;
        }
        else
        {
            throw std::logic_error("coeff_modulus cannot be changed after the "
                                   "context is generated!");
        }
    }

    void HEContext<Scheme::CKKS>::generate()
    {
        if ((!context_generated_) && (poly_modulus_degree_specified_) &&
            (coeff_modulus_specified_))
        {
            // Memory pool initialization
            MemoryPool::instance().initialize();
            MemoryPool::instance().use_memory_pool(true);
            cudaDeviceSynchronize();

            // DRNG initialization
            std::vector<unsigned char> generated_entropy(16); // for 128 bit
            if (1 !=
                RAND_bytes(generated_entropy.data(), generated_entropy.size()))
                throw std::runtime_error("RAND_bytes failed");
            std::vector<unsigned char> generated_nonce(8); // for 128 bit
            if (1 !=
                RAND_bytes(generated_entropy.data(), generated_entropy.size()))
                throw std::runtime_error("RAND_bytes failed");
            std::vector<unsigned char> personalization_string = {};
            RandomNumberGenerator::instance().initialize(
                generated_entropy, generated_nonce, personalization_string,
                rngongpu::SecurityLevel::AES128, false);
            cudaDeviceSynchronize();

            // For kernel stack size
            cudaDeviceSetLimit(cudaLimitStackSize, 2048);

            modulus_ = std::make_shared<DeviceVector<Modulus64>>(prime_vector_);

            std::vector<Data64> base_q_psi =
                generate_primitive_root_of_unity(n, prime_vector_);
            std::vector<Root64> Qprime_ntt_table =
                generate_ntt_table(base_q_psi, prime_vector_, n_power);
            std::vector<Root64> Qprime_intt_table =
                generate_intt_table(base_q_psi, prime_vector_, n_power);
            std::vector<Ninverse64> Qprime_n_inverse =
                generate_n_inverse(n, prime_vector_);

            ntt_table_ =
                std::make_shared<DeviceVector<Root64>>(Qprime_ntt_table);

            intt_table_ =
                std::make_shared<DeviceVector<Root64>>(Qprime_intt_table);

            n_inverse_ =
                std::make_shared<DeviceVector<Ninverse64>>(Qprime_n_inverse);

            std::vector<Data64> last_q_modinv =
                calculate_last_q_modinv(prime_vector_, Q_prime_size, P_size);
            std::vector<Data64> half = calculate_half(prime_vector_, P_size);
            std::vector<Data64> half_mod =
                calculate_half_mod(prime_vector_, half, Q_prime_size, P_size);
            std::vector<Data64> factor =
                calculate_factor(prime_vector_, Q_size, P_size);

            last_q_modinv_ =
                std::make_shared<DeviceVector<Data64>>(last_q_modinv);

            half_p_ = std::make_shared<DeviceVector<Data64>>(half);

            half_mod_ = std::make_shared<DeviceVector<Data64>>(half_mod);

            factor_ = std::make_shared<DeviceVector<Data64>>(factor);

            ///////////////////////
            ///////////////////////
            ///////////////////////

            // For Rescale parameters for all depth
            std::vector<Data64> rescale_last_q_modinv;
            std::vector<Data64> rescaled_half_mod;
            std::vector<Data64> rescaled_half;
            for (int j = 0; j < (Q_size - 1); j++)
            {
                int inner = (Q_size - 1) - j;
                rescaled_half.push_back(prime_vector_[inner].value >> 1);
                for (int i = 0; i < inner; i++)
                {
                    Data64 temp_ =
                        prime_vector_[inner].value % prime_vector_[i].value;
                    rescale_last_q_modinv.push_back(
                        OPERATOR64::modinv(temp_, prime_vector_[i]));
                    rescaled_half_mod.push_back(rescaled_half[j] %
                                                prime_vector_[i].value);
                }
            }

            rescaled_last_q_modinv_ =
                std::make_shared<DeviceVector<Data64>>(rescale_last_q_modinv);

            rescaled_half_mod_ =
                std::make_shared<DeviceVector<Data64>>(rescaled_half_mod);

            rescaled_half_ =
                std::make_shared<DeviceVector<Data64>>(rescaled_half);

            std::vector<Data64> Mi;
            std::vector<Data64> Mi_inv;
            std::vector<Data64> upper_half_threshold;
            std::vector<Data64> decryption_modulus;

            for (int i = 0; i < Q_size; i++)
            {
                int depth_Q_size = Q_size - i;

                // Mi
                std::vector<Data64> Mi_inner =
                    calculate_Mi(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size * depth_Q_size; j++)
                {
                    Mi.push_back(Mi_inner[j]);
                }

                // Mi_inv
                std::vector<Data64> Mi_inv_inner =
                    calculate_Mi_inv(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    Mi_inv.push_back(Mi_inv_inner[j]);
                }

                // upper_half_threshold
                std::vector<Data64> upper_half_threshold_inner =
                    calculate_upper_half_threshold(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    upper_half_threshold.push_back(
                        upper_half_threshold_inner[j]);
                }

                // decryption_modulus
                std::vector<Data64> M_inner =
                    calculate_M(prime_vector_, depth_Q_size);
                for (int j = 0; j < depth_Q_size; j++)
                {
                    decryption_modulus.push_back(M_inner[j]);
                }
            }

            Mi_ = std::make_shared<DeviceVector<Data64>>(Mi);

            Mi_inv_ = std::make_shared<DeviceVector<Data64>>(Mi_inv);

            upper_half_threshold_ =
                std::make_shared<DeviceVector<Data64>>(upper_half_threshold);

            decryption_modulus_ =
                std::make_shared<DeviceVector<Data64>>(decryption_modulus);

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

            //////////////////////////////////

            switch (static_cast<int>(keyswitching_type_))
            {
                case 1: // KEYSWITCHING_METHOD_I
                    // Deafult
                    break;
                case 2: // KEYSWITCHING_METHOD_II
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

                    std::vector<std::vector<Data64>>
                        base_change_matrix_D_to_Qtilda_vec =
                            pool_ckks.level_base_change_matrix_D_to_Qtilda();

                    std::vector<std::vector<Data64>> Mi_inv_D_to_Qtilda_vec =
                        pool_ckks.level_Mi_inv_D_to_Qtilda();

                    std::vector<std::vector<Data64>> prod_D_to_Qtilda_vec =
                        pool_ckks.level_prod_D_to_Qtilda();

                    std::vector<std::vector<int>> I_j_vec =
                        pool_ckks.level_I_j();

                    std::vector<std::vector<int>> I_location_vec =
                        pool_ckks.level_I_location();

                    std::vector<std::vector<int>> Sk_pair_new_vec =
                        pool_ckks.level_sk_pair();

                    base_change_matrix_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    Mi_inv_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    prod_D_to_Qtilda_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
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
                        DeviceVector<Data64>
                            base_change_matrix_D_to_Qtilda_leveled_inner(
                                base_change_matrix_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data64> Mi_inv_D_to_Qtilda_leveled_inner(
                            Mi_inv_D_to_Qtilda_vec[pool_lp]);
                        DeviceVector<Data64> prod_D_to_Qtilda_leveled_inner(
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
                break;
                case 3: // KEYSWITCHING_METHOD_III
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

                    std::vector<Modulus64> B_prime_inner = pool_ckks.B_prime;
                    B_prime_leveled = std::make_shared<DeviceVector<Modulus64>>(
                        B_prime_inner);

                    std::vector<Root64> B_prime_ntt_tables_leveled_inner =
                        pool_ckks.B_prime_ntt_tables();
                    B_prime_ntt_tables_leveled =
                        std::make_shared<DeviceVector<Root64>>(
                            B_prime_ntt_tables_leveled_inner);

                    std::vector<Root64> B_prime_intt_tables_leveled_inner =
                        pool_ckks.B_prime_intt_tables();
                    B_prime_intt_tables_leveled =
                        std::make_shared<DeviceVector<Root64>>(
                            B_prime_intt_tables_leveled_inner);

                    std::vector<Ninverse64> B_prime_n_inverse_leveled_inner =
                        pool_ckks.B_prime_n_inverse();
                    B_prime_n_inverse_leveled =
                        std::make_shared<DeviceVector<Ninverse64>>(
                            B_prime_n_inverse_leveled_inner);

                    std::vector<std::vector<Data64>>
                        base_change_matrix_D_to_B_vec =
                            pool_ckks.level_base_change_matrix_D_to_B();

                    std::vector<std::vector<Data64>>
                        base_change_matrix_B_to_D_vec =
                            pool_ckks.level_base_change_matrix_B_to_D();

                    std::vector<std::vector<Data64>> Mi_inv_D_to_B_vec =
                        pool_ckks.level_Mi_inv_D_to_B();
                    std::vector<Data64> Mi_inv_B_to_D_vec =
                        pool_ckks.level_Mi_inv_B_to_D();

                    Mi_inv_B_to_D_leveled =
                        std::make_shared<DeviceVector<Data64>>(
                            Mi_inv_B_to_D_vec);

                    std::vector<std::vector<Data64>> prod_D_to_B_vec =
                        pool_ckks.level_prod_D_to_B();
                    std::vector<std::vector<Data64>> prod_B_to_D_vec =
                        pool_ckks.level_prod_B_to_D();

                    std::vector<std::vector<int>> I_j_vec =
                        pool_ckks.level_I_j_2();
                    std::vector<std::vector<int>> I_location_vec =
                        pool_ckks.level_I_location_2();
                    std::vector<std::vector<int>> Sk_pair_new_vec =
                        pool_ckks.level_sk_pair();

                    base_change_matrix_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    base_change_matrix_B_to_D_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    Mi_inv_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    prod_D_to_B_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();
                    prod_B_to_D_leveled =
                        std::make_shared<std::vector<DeviceVector<Data64>>>();

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
                        DeviceVector<Data64>
                            base_change_matrix_D_to_B_leveled_inner(
                                base_change_matrix_D_to_B_vec[pool_lp]);
                        DeviceVector<Data64>
                            base_change_matrix_B_to_D_leveled_inner(
                                base_change_matrix_B_to_D_vec[pool_lp]);
                        DeviceVector<Data64> Mi_inv_D_to_B_leveled_inner(
                            Mi_inv_D_to_B_vec[pool_lp]);
                        DeviceVector<Data64> prod_D_to_B_leveled_inner(
                            prod_D_to_B_vec[pool_lp]);
                        DeviceVector<Data64> prod_B_to_D_leveled_inner(
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
                break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }
            context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Context is already generated!");
        }
    }

    void HEContext<Scheme::CKKS>::print_parameters()
    {
        if (context_generated_)
        {
            std::string scheme_string = "CKKS";

            std::cout << "==== HEonGPU a GPU Based Homomorphic Encryption "
                         "Library ====\n"
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

            std::cout << std::endl;
        }
        else
        {
            std::cout << "Parameters is not generated yet!" << std::endl;
        }
    }

    void HEContext<Scheme::CKKS>::save(std::ostream& os) const
    {
        if ((poly_modulus_degree_specified_) && (coeff_modulus_specified_))
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &sec_level_, sizeof(sec_level_));

            os.write((char*) &keyswitching_type_, sizeof(keyswitching_type_));

            os.write((char*) &n, sizeof(n));

            os.write((char*) &n_power, sizeof(n_power));

            os.write((char*) &coeff_modulus, sizeof(coeff_modulus));

            os.write((char*) &total_coeff_bit_count,
                     sizeof(total_coeff_bit_count));

            os.write((char*) &Q_prime_size, sizeof(Q_prime_size));

            os.write((char*) &Q_size, sizeof(Q_size));

            os.write((char*) &P_size, sizeof(P_size));

            uint32_t prime_vector_count = prime_vector_.size();
            os.write((char*) &prime_vector_count, sizeof(prime_vector_count));
            os.write((char*) prime_vector_.data(),
                     sizeof(Modulus64) * prime_vector_count);

            uint32_t base_q_count = base_q.size();
            os.write((char*) &base_q_count, sizeof(base_q_count));
            os.write((char*) base_q.data(), sizeof(Data64) * base_q_count);

            uint32_t Qprime_mod_bit_sizes_count = Qprime_mod_bit_sizes_.size();
            os.write((char*) &Qprime_mod_bit_sizes_count,
                     sizeof(Qprime_mod_bit_sizes_count));
            os.write((char*) Qprime_mod_bit_sizes_.data(),
                     sizeof(int) * Qprime_mod_bit_sizes_count);

            uint32_t Q_mod_bit_sizes_count = Q_mod_bit_sizes_.size();
            os.write((char*) &Q_mod_bit_sizes_count,
                     sizeof(Q_mod_bit_sizes_count));
            os.write((char*) Q_mod_bit_sizes_.data(),
                     sizeof(int) * Q_mod_bit_sizes_count);

            uint32_t P_mod_bit_sizes_count = P_mod_bit_sizes_.size();
            os.write((char*) &P_mod_bit_sizes_count,
                     sizeof(P_mod_bit_sizes_count));
            os.write((char*) P_mod_bit_sizes_.data(),
                     sizeof(int) * P_mod_bit_sizes_count);
        }
        else
        {
            throw std::runtime_error(
                "Context has no enough parameters to serialize!");
        }
    }

    void HEContext<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!context_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            is.read((char*) &sec_level_, sizeof(sec_level_));

            is.read((char*) &keyswitching_type_, sizeof(keyswitching_type_));

            is.read((char*) &n, sizeof(n));

            is.read((char*) &n_power, sizeof(n_power));

            is.read((char*) &coeff_modulus, sizeof(coeff_modulus));

            is.read((char*) &total_coeff_bit_count,
                    sizeof(total_coeff_bit_count));

            is.read((char*) &Q_prime_size, sizeof(Q_prime_size));

            is.read((char*) &Q_size, sizeof(Q_size));

            is.read((char*) &P_size, sizeof(P_size));

            uint32_t prime_vector_count;
            is.read((char*) &prime_vector_count, sizeof(prime_vector_count));
            prime_vector_.resize(prime_vector_count);
            is.read((char*) prime_vector_.data(),
                    sizeof(Modulus64) * prime_vector_count);

            uint32_t base_q_count;
            is.read((char*) &base_q_count, sizeof(base_q_count));
            base_q.resize(base_q_count);
            is.read((char*) base_q.data(), sizeof(Data64) * base_q_count);

            uint32_t Qprime_mod_bit_sizes_count;
            is.read((char*) &Qprime_mod_bit_sizes_count,
                    sizeof(Qprime_mod_bit_sizes_count));
            Qprime_mod_bit_sizes_.resize(Qprime_mod_bit_sizes_count);
            is.read((char*) Qprime_mod_bit_sizes_.data(),
                    sizeof(int) * Qprime_mod_bit_sizes_count);

            uint32_t Q_mod_bit_sizes_count;
            is.read((char*) &Q_mod_bit_sizes_count,
                    sizeof(Q_mod_bit_sizes_count));
            Q_mod_bit_sizes_.resize(Q_mod_bit_sizes_count);
            is.read((char*) Q_mod_bit_sizes_.data(),
                    sizeof(int) * Q_mod_bit_sizes_count);

            uint32_t P_mod_bit_sizes_count;
            is.read((char*) &P_mod_bit_sizes_count,
                    sizeof(P_mod_bit_sizes_count));
            P_mod_bit_sizes_.resize(P_mod_bit_sizes_count);
            is.read((char*) P_mod_bit_sizes_.data(),
                    sizeof(int) * P_mod_bit_sizes_count);

            poly_modulus_degree_specified_ = true;
            coeff_modulus_specified_ = true;
            context_generated_ = false;

            this->generate();
        }
        else
        {
            throw std::runtime_error("Context has been already exist!");
        }
    }

    template class HEContext<Scheme::CKKS>;

} // namespace heongpu
