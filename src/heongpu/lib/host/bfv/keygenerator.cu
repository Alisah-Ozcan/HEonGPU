// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "bfv/keygenerator.cuh"

namespace heongpu
{
    __host__
    HEKeyGenerator<Scheme::BFV>::HEKeyGenerator(HEContext<Scheme::BFV>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        new_seed_ = RNGSeed();

        n = context.n;
        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        modulus_ = context.modulus_;
        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;
        n_inverse_ = context.n_inverse_;
        factor_ = context.factor_;

        d_ = context.d;
        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        B_prime_ = context.B_prime_;
        B_prime_ntt_tables_ = context.B_prime_ntt_tables_;
        B_prime_intt_tables_ = context.B_prime_intt_tables_;
        B_prime_n_inverse_ = context.B_prime_n_inverse_;

        base_change_matrix_D_to_B_ = context.base_change_matrix_D_to_B_;
        base_change_matrix_B_to_D_ = context.base_change_matrix_B_to_D_;
        Mi_inv_D_to_B_ = context.Mi_inv_D_to_B_;
        Mi_inv_B_to_D_ = context.Mi_inv_B_to_D_;
        prod_D_to_B_ = context.prod_D_to_B_;
        prod_B_to_D_ = context.prod_B_to_D_;

        I_j_ = context.I_j_;
        I_location_ = context.I_location_;
        Sk_pair_ = context.Sk_pair_;
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_secret_key(
        Secretkey<Scheme::BFV>& sk, const ExecutionOptions& options)
    {
        if (sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                DeviceVector<int> secret_key_without_rns((n), options.stream_);

                secretkey_gen_kernel<<<dim3((n >> 8), 1, 1), 256, 0,
                                       options.stream_>>>(
                    secret_key_without_rns.data(), sk_.hamming_weight_, n_power,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                DeviceVector<Data64> secret_key_rns(
                    (sk_.coeff_modulus_count() * n), options.stream_);

                secretkey_rns_kernel<<<dim3((n >> 8), 1, 1), 256, 0,
                                       options.stream_>>>(
                    secret_key_without_rns.data(), secret_key_rns.data(),
                    modulus_->data(), n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = options.stream_};

                gpuntt::GPU_NTT_Inplace(secret_key_rns.data(),
                                        ntt_table_->data(), modulus_->data(),
                                        cfg_ntt, Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                sk_.in_ntt_domain_ = true;
                sk_.secret_key_generated_ = true;

                sk_.memory_set(std::move(secret_key_rns));
            },
            options, true);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_public_key(
        Publickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (pk.public_key_generated_)
        {
            throw std::logic_error("Publickey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    pk,
                    [&](Publickey<Scheme::BFV>& pk_)
                    {
                        DeviceVector<Data64> output_memory(
                            (2 * Q_prime_size_ * n), options.stream_);

                        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * n,
                                                      options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly = error_poly + (Q_prime_size_ * n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, 1, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, 1, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(errors_a.data(),
                                                ntt_table_->data(),
                                                modulus_->data(), cfg_ntt,
                                                Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        publickey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk_.data(), error_poly,
                            a_poly, modulus_->data(), n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        pk_.memory_set(std::move(output_memory));

                        pk_.in_ntt_domain_ = true;
                        pk_.public_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::BFV>::generate_multi_party_public_key_piece(
        MultipartyPublickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (pk.public_key_generated_)
        {
            throw std::logic_error("Publickey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    pk,
                    [&](MultipartyPublickey<Scheme::BFV>& pk_)
                    {
                        DeviceVector<Data64> output_memory(
                            (2 * Q_prime_size_ * n), options.stream_);

                        RNGSeed common_seed = pk_.seed();

                        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * n,
                                                      options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly = error_poly + (Q_prime_size_ * n);

                        RandomNumberGenerator::instance().set(
                            common_seed.key_, common_seed.nonce_,
                            common_seed.personalization_string_,
                            options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, 1, options.stream_);

                        RNGSeed gen_seed;
                        RandomNumberGenerator::instance().set(
                            gen_seed.key_, gen_seed.nonce_,
                            gen_seed.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, 1, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(errors_a.data(),
                                                ntt_table_->data(),
                                                modulus_->data(), cfg_ntt,
                                                Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        publickey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk_.data(), error_poly,
                            a_poly, modulus_->data(), n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        pk_.memory_set(std::move(output_memory));

                        pk_.in_ntt_domain_ = true;
                        pk_.public_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_multi_party_public_key(
        std::vector<MultipartyPublickey<Scheme::BFV>>& all_pk,
        Publickey<Scheme::BFV>& pk, const ExecutionOptions& options)
    {
        int participant_count = all_pk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_pk[i].public_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyPublickey is not generated!");
            }
        }

        input_vector_storage_manager(
            all_pk,
            [&](std::vector<MultipartyPublickey<Scheme::BFV>>& all_pk_)
            {
                output_storage_manager(
                    pk,
                    [&](Publickey<Scheme::BFV>& pk_)
                    {
                        DeviceVector<Data64> output_memory(
                            (2 * Q_prime_size_ * n), options.stream_);

                        global_memory_replace_kernel<<<
                            dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                            options.stream_>>>(all_pk[0].data(),
                                               output_memory.data(), n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1; i < participant_count; i++)
                        {
                            threshold_pk_addition<<<dim3((n >> 8),
                                                         Q_prime_size_, 1),
                                                    256, 0, options.stream_>>>(
                                all_pk[i].data(), output_memory.data(),
                                output_memory.data(), modulus_->data(), n_power,
                                false);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }

                        pk_.memory_set(std::move(output_memory));

                        pk_.in_ntt_domain_ = true;
                        pk_.public_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_relin_key_method_I(
        Relinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::BFV>& rk_)
                    {
                        DeviceVector<Data64> errors_a(
                            2 * Q_prime_size_ * Q_size_ * n, options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly =
                            error_poly + (Q_prime_size_ * Q_size_ * n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, Q_size_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, Q_size_,
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        relinkey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                            output_memory.data(), sk_.data(), error_poly,
                            a_poly, modulus_->data(), factor_->data(), n_power,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::
        generate_multi_party_relin_key_piece_method_I_stage_I(
            MultipartyRelinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
                    {
                        RNGSeed common_seed = rk.seed();

                        DeviceVector<Data64> random_values(
                            Q_prime_size_ * ((3 * Q_size_) + 1) * n,
                            options.stream_);
                        Data64* e0 = random_values.data();
                        Data64* e1 = e0 + (Q_prime_size_ * Q_size_ * n);
                        Data64* u = e1 + (Q_prime_size_ * Q_size_ * n);
                        Data64* common_a = u + (Q_prime_size_ * n);

                        RandomNumberGenerator::instance().set(
                            common_seed.key_, common_seed.nonce_,
                            common_seed.personalization_string_,
                            options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                common_a, modulus_->data(), n_power,
                                Q_prime_size_, Q_size_, options.stream_);

                        RNGSeed gen_seed1;
                        RandomNumberGenerator::instance().set(
                            gen_seed1.key_, gen_seed1.nonce_,
                            gen_seed1.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, e0, modulus_->data(), n_power,
                                Q_prime_size_, 2 * Q_size_, options.stream_);

                        RandomNumberGenerator::instance().set(
                            new_seed_.key_, new_seed_.nonce_,
                            new_seed_.personalization_string_, options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                u, modulus_->data(), n_power, Q_prime_size_, 1,
                                options.stream_);

                        RNGSeed gen_seed2;
                        RandomNumberGenerator::instance().set(
                            gen_seed2.key_, gen_seed2.nonce_,
                            gen_seed2.personalization_string_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            random_values.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt,
                            Q_prime_size_ * ((2 * Q_size_) + 1), Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_piece_method_I_stage_I_kernel<<<
                            dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                            options.stream_>>>(
                            output_memory.data(), sk.data(), common_a, u, e0,
                            e1, modulus_->data(), factor_->data(), n_power,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::
        generate_multi_party_relin_key_piece_method_I_stage_II(
            MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
            MultipartyRelinkey<Scheme::BFV>& rk_stage_2,
            Secretkey<Scheme::BFV>& sk, const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!rk_stage_1.relin_key_generated_)
        {
            throw std::logic_error("Relinkey1 is not generated!");
        }

        if (rk_stage_2.relin_key_generated_)
        {
            throw std::logic_error("Relinkey2 is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_2_)
                            {
                                DeviceVector<Data64> random_values(
                                    Q_prime_size_ * ((2 * Q_size_) + 1) * n,
                                    options.stream_);
                                Data64* e0 = random_values.data();
                                Data64* e1 = e0 + (Q_prime_size_ * Q_size_ * n);
                                Data64* u = e1 + (Q_prime_size_ * Q_size_ * n);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, e0, modulus_->data(),
                                        n_power, Q_prime_size_, 2,
                                        options.stream_);

                                RandomNumberGenerator::instance().set(
                                    new_seed_.key_, new_seed_.nonce_,
                                    new_seed_.personalization_string_,
                                    options.stream_);
                                RandomNumberGenerator::instance()
                                    .modular_ternary_random_number_generation(
                                        u, modulus_->data(), n_power,
                                        Q_prime_size_, 1, options.stream_);

                                RNGSeed gen_seed;
                                RandomNumberGenerator::instance().set(
                                    gen_seed.key_, gen_seed.nonce_,
                                    gen_seed.personalization_string_,
                                    options.stream_);

                                gpuntt::ntt_rns_configuration<Data64> cfg_ntt =
                                    {.n_power = n_power,
                                     .ntt_type = gpuntt::FORWARD,
                                     .reduction_poly =
                                         gpuntt::ReductionPolynomial::X_N_plus,
                                     .zero_padding = false,
                                     .stream = options.stream_};

                                gpuntt::GPU_NTT_Inplace(
                                    random_values.data(), ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_prime_size_ * ((2 * Q_size_) + 1),
                                    Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    rk_stage_2_.relinkey_size_,
                                    options.stream_);

                                multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    rk_stage_1_.data(), output_memory.data(),
                                    sk.data(), u, e0, e1, modulus_->data(),
                                    n_power, Q_prime_size_, Q_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                rk_stage_2_.memory_set(
                                    std::move(output_memory));

                                rk_stage_2_.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::
        generate_bfv_multi_party_relin_key_piece_method_II_stage_I(
            MultipartyRelinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
                    {
                        RNGSeed common_seed = rk.seed();

                        DeviceVector<Data64> random_values(
                            Q_prime_size_ * ((3 * d_) + 1) * n,
                            options.stream_);
                        Data64* e0 = random_values.data();
                        Data64* e1 = e0 + (Q_prime_size_ * d_ * n);
                        Data64* u = e1 + (Q_prime_size_ * d_ * n);
                        Data64* common_a = u + (Q_prime_size_ * n);

                        RandomNumberGenerator::instance().set(
                            common_seed.key_, common_seed.nonce_,
                            common_seed.personalization_string_,
                            options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                common_a, modulus_->data(), n_power,
                                Q_prime_size_, d_, options.stream_);

                        RNGSeed gen_seed1;
                        RandomNumberGenerator::instance().set(
                            gen_seed1.key_, gen_seed1.nonce_,
                            gen_seed1.personalization_string_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, e0, modulus_->data(), n_power,
                                Q_prime_size_, 2 * d_, options.stream_);

                        RandomNumberGenerator::instance().set(
                            new_seed_.key_, new_seed_.nonce_,
                            new_seed_.personalization_string_, options.stream_);
                        RandomNumberGenerator::instance()
                            .modular_ternary_random_number_generation(
                                u, modulus_->data(), n_power, Q_prime_size_, 1,
                                options.stream_);

                        RNGSeed gen_seed2;
                        RandomNumberGenerator::instance().set(
                            gen_seed2.key_, gen_seed2.nonce_,
                            gen_seed2.personalization_string_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            random_values.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt,
                            Q_prime_size_ * ((2 * d_) + 1), Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_piece_method_II_stage_I_kernel<<<
                            dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                            options.stream_>>>(
                            output_memory.data(), sk.data(), common_a, u, e0,
                            e1, modulus_->data(), factor_->data(),
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::
        generate_bfv_multi_party_relin_key_piece_method_II_stage_II(
            MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
            MultipartyRelinkey<Scheme::BFV>& rk_stage_2,
            Secretkey<Scheme::BFV>& sk, const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!rk_stage_1.relin_key_generated_)
        {
            throw std::logic_error("Relinkey1 is not generated!");
        }

        if (rk_stage_2.relin_key_generated_)
        {
            throw std::logic_error("Relinkey2 is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                input_storage_manager(
                    rk_stage_1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_1_)
                    {
                        output_storage_manager(
                            rk_stage_2,
                            [&](MultipartyRelinkey<Scheme::BFV>& rk_stage_2_)
                            {
                                DeviceVector<Data64> random_values(
                                    Q_prime_size_ * ((2 * d_) + 1) * n,
                                    options.stream_);
                                Data64* e0 = random_values.data();
                                Data64* e1 = e0 + (Q_prime_size_ * d_ * n);
                                Data64* u = e1 + (Q_prime_size_ * d_ * n);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, e0, modulus_->data(),
                                        n_power, Q_prime_size_, 2,
                                        options.stream_);

                                RandomNumberGenerator::instance().set(
                                    new_seed_.key_, new_seed_.nonce_,
                                    new_seed_.personalization_string_,
                                    options.stream_);
                                RandomNumberGenerator::instance()
                                    .modular_ternary_random_number_generation(
                                        u, modulus_->data(), n_power,
                                        Q_prime_size_, 1, options.stream_);

                                RNGSeed gen_seed;
                                RandomNumberGenerator::instance().set(
                                    gen_seed.key_, gen_seed.nonce_,
                                    gen_seed.personalization_string_,
                                    options.stream_);

                                gpuntt::ntt_rns_configuration<Data64> cfg_ntt =
                                    {.n_power = n_power,
                                     .ntt_type = gpuntt::FORWARD,
                                     .reduction_poly =
                                         gpuntt::ReductionPolynomial::X_N_plus,
                                     .zero_padding = false,
                                     .stream = options.stream_};

                                gpuntt::GPU_NTT_Inplace(
                                    random_values.data(), ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_prime_size_ * ((2 * d_) + 1),
                                    Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    rk_stage_2.relinkey_size_, options.stream_);

                                multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    rk_stage_1.data(), output_memory.data(),
                                    sk.data(), u, e0, e1, modulus_->data(),
                                    n_power, Q_prime_size_, d_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                rk_stage_2.memory_set(std::move(output_memory));

                                rk_stage_2.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_multi_party_relin_key(
        std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk,
        MultipartyRelinkey<Scheme::BFV>& rk, const ExecutionOptions& options)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_rk[i].relin_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey is not generated!");
            }
        }

        int dimension;
        switch (static_cast<int>(rk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = rk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = rk.d_;
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

        input_vector_storage_manager(
            all_rk,
            [&](std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk_)
            {
                output_storage_manager(
                    rk,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_)
                    {
                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        multi_party_relinkey_method_I_stage_I_kernel<<<
                            dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                            options.stream_>>>(all_rk[0].data(),
                                               output_memory.data(),
                                               modulus_->data(), n_power,
                                               Q_prime_size_, dimension, true);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        for (int i = 1; i < participant_count; i++)
                        {
                            multi_party_relinkey_method_I_stage_I_kernel<<<
                                dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                options.stream_>>>(
                                all_rk[i].data(), output_memory.data(),
                                modulus_->data(), n_power, Q_prime_size_,
                                dimension, false);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_multi_party_relin_key(
        std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk,
        MultipartyRelinkey<Scheme::BFV>& rk_common_stage1,
        Relinkey<Scheme::BFV>& rk, const ExecutionOptions& options)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_rk[i].relin_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey is not generated!");
            }
        }

        if (!rk_common_stage1.relin_key_generated_)
        {
            throw std::logic_error("Common Relinkey is not generated!");
        }

        int dimension;
        switch (static_cast<int>(rk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = rk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = rk.d_;
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

        input_vector_storage_manager(
            all_rk,
            [&](std::vector<MultipartyRelinkey<Scheme::BFV>>& all_rk_)
            {
                input_storage_manager(
                    rk_common_stage1,
                    [&](MultipartyRelinkey<Scheme::BFV>& rk_common_stage1_)
                    {
                        output_storage_manager(
                            rk,
                            [&](Relinkey<Scheme::BFV>& rk_)
                            {
                                DeviceVector<Data64> output_memory(
                                    rk_.relinkey_size_, options.stream_);

                                multi_party_relinkey_method_I_stage_II_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    all_rk[0].data(), rk_common_stage1.data(),
                                    output_memory.data(), modulus_->data(),
                                    n_power, Q_prime_size_, dimension);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                for (int i = 1; i < participant_count; i++)
                                {
                                    multi_party_relinkey_method_I_stage_II_kernel<<<
                                        dim3((n >> 8), Q_prime_size_, 1), 256,
                                        0, options.stream_>>>(
                                        all_rk[i].data(), output_memory.data(),
                                        modulus_->data(), n_power,
                                        Q_prime_size_, dimension);
                                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                                }

                                rk_.memory_set(std::move(output_memory));

                                rk_.relin_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_bfv_relin_key_method_II(
        Relinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::BFV>& rk_)
                    {
                        DeviceVector<Data64> errors_a(
                            2 * Q_prime_size_ * d_ * n, options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                      1),
                                                 256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), Sk_pair_->data(),
                            n_power, Q_prime_size_, d_, Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::BFV>::generate_bfv_relin_key_method_III(
        Relinkey<Scheme::BFV>& rk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (rk.relin_key_generated_)
        {
            throw std::logic_error("Relinkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::BFV>& rk_)
                    {
                        DeviceVector<Data64> errors_a(
                            2 * Q_prime_size_ * d_ * n, options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> temp_calculation(
                            2 * Q_prime_size_ * d_ * n, options.stream_);

                        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                      1),
                                                 256, 0, options.stream_>>>(
                            temp_calculation.data(), sk.data(), error_poly,
                            a_poly, modulus_->data(), factor_->data(),
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::INVERSE,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .mod_inverse = n_inverse_->data(),
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            temp_calculation.data(), intt_table_->data(),
                            modulus_->data(), cfg_intt, 2 * Q_prime_size_ * d_,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        relinkey_DtoB_kernel<<<dim3((n >> 8), d_tilda_,
                                                    (d_ << 1)),
                                               256, 0, options.stream_>>>(
                            temp_calculation.data(), output_memory.data(),
                            modulus_->data(), B_prime_->data(),
                            base_change_matrix_D_to_B_->data(),
                            Mi_inv_D_to_B_->data(), prod_D_to_B_->data(),
                            I_j_->data(), I_location_->data(), n_power,
                            Q_prime_size_, d_tilda_, d_, r_prime_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        gpuntt::GPU_NTT_Inplace(
                            output_memory.data(), B_prime_ntt_tables_->data(),
                            B_prime_->data(), cfg_ntt,
                            2 * d_tilda_ * d_ * r_prime_, r_prime_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_galois_key_method_I(
        Galoiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n,
                                              options.stream_);
                Data64* error_poly = errors_a.data();
                Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, Q_size_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, Q_size_,
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois.second, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois.second] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois.second] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, modulus_->data(), n_power, Q_prime_size_,
                            Q_size_, options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, Q_size_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, Q_size_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, Q_size_,
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois_] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois_] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, modulus_->data(), n_power, Q_prime_size_,
                            Q_size_, options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, Q_size_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::BFV>::generate_bfv_galois_key_method_II(
        Galoiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n,
                                              options.stream_);
                Data64* error_poly = errors_a.data();
                Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois.second, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois.second] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois.second] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, modulus_->data(), n_power, Q_prime_size_,
                            d_, options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                            modulus_->data(), cfg_ntt,
                                            d_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_, options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois_] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois_] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_uniform_random_number_generation(
                            a_poly, modulus_->data(), n_power, Q_prime_size_,
                            d_, options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                            modulus_->data(), cfg_ntt,
                                            d_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::BFV>::generate_multi_party_galois_key_piece_method_I(
        MultipartyGaloiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        RNGSeed common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n,
                                      options.stream_);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, options.stream_);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(a_poly, modulus_->data(),
                                                      n_power, Q_prime_size_,
                                                      Q_size_, options.stream_);

        RNGSeed gen_seed1;
        RandomNumberGenerator::instance().set(gen_seed1.key_, gen_seed1.nonce_,
                                              gen_seed1.personalization_string_,
                                              options.stream_);

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, Q_size_,
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois.second, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois.second] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois.second] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, Q_size_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, Q_size_,
                                options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            n_power, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois_] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois_] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, Q_size_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                           256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::
        generate_bfv_multi_party_galois_key_piece_method_II(
            MultipartyGaloiskey<Scheme::BFV>& gk, Secretkey<Scheme::BFV>& sk,
            const ExecutionOptions& options)
    {
        if (!sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (gk.galois_key_generated_)
        {
            throw std::logic_error("Galoiskey is already generated!");
        }

        RNGSeed common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n,
                                      options.stream_);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, options.stream_);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(a_poly, modulus_->data(),
                                                      n_power, Q_prime_size_,
                                                      d_, options.stream_);

        RNGSeed gen_seed1;
        RandomNumberGenerator::instance().set(gen_seed1.key_, gen_seed1.nonce_,
                                              gen_seed1.personalization_string_,
                                              options.stream_);

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::BFV>& sk_)
            {
                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois.second, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois.second] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois.second] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(
                                gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                            modulus_->data(), cfg_ntt,
                                            d_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }
                else
                {
                    for (auto& galois_ : gk.custom_galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_, d_, options.stream_);

                        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                            .n_power = n_power,
                            .ntt_type = gpuntt::FORWARD,
                            .reduction_poly =
                                gpuntt::ReductionPolynomial::X_N_plus,
                            .zero_padding = false,
                            .stream = options.stream_};

                        gpuntt::GPU_NTT_Inplace(
                            error_poly, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, d_ * Q_prime_size_, Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_->data(), n_power, Q_prime_size_, d_,
                            Q_size_, P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        if (options.storage_ == storage_type::DEVICE)
                        {
                            gk.device_location_[galois_] =
                                std::move(output_memory);
                        }
                        else
                        {
                            gk.host_location_[galois_] =
                                HostVector<Data64>(gk.galoiskey_size_);
                            cudaMemcpyAsync(gk.host_location_[galois_].data(),
                                            output_memory.data(),
                                            gk.galoiskey_size_ * sizeof(Data64),
                                            cudaMemcpyDeviceToHost,
                                            options.stream_);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());
                        }
                    }

                    // Columns Rotate
                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_, options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                            modulus_->data(), cfg_ntt,
                                            d_ * Q_prime_size_, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    if (options.storage_ == storage_type::DEVICE)
                    {
                        gk.zero_device_location_ = std::move(output_memory);
                    }
                    else
                    {
                        gk.zero_host_location_ =
                            HostVector<Data64>(gk.galoiskey_size_);
                        cudaMemcpyAsync(
                            gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());
                    }
                }

                gk.galois_key_generated_ = true;
                gk.storage_type_ = options.storage_;
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_multi_party_galois_key(
        std::vector<MultipartyGaloiskey<Scheme::BFV>>& all_gk,
        Galoiskey<Scheme::BFV>& gk, const ExecutionOptions& options)
    {
        int participant_count = all_gk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common galois!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if ((gk.customized != all_gk[i].customized) ||
                (gk.group_order_ != all_gk[i].group_order_))
            {
                throw std::invalid_argument(
                    "MultipartyGaloiskey context is not valid || "
                    "MultipartyGaloiskey is not generated!");
            }
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (!all_gk[i].galois_key_generated_)
            {
                throw std::invalid_argument(
                    "MultipartyGaloiskey is not generated!");
            }
        }

        std::vector<storage_type> input_storage_types;
        for (int i = 0; i < participant_count; i++)
        {
            input_storage_types.push_back(all_gk[i].storage_type_);

            if (all_gk[i].storage_type_ == storage_type::DEVICE)
            {
                for (const auto& pair : all_gk[i].device_location_)
                {
                    if (pair.second.size() < all_gk[i].galoiskey_size_)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys size is not valid || "
                            "MultipartyGaloiskeys is not generated!");
                    }
                }
            }
            else
            {
                for (const auto& pair : all_gk[i].host_location_)
                {
                    if (pair.second.size() < all_gk[i].galoiskey_size_)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys size is not valid || "
                            "MultipartyGaloiskeys is not generated!");
                    }
                }

                all_gk[i].store_in_device();
            }
        }

        int dimension;
        switch (static_cast<int>(gk.key_type))
        {
            case 1: // KEYSWITCHING_METHOD_I
                dimension = gk.Q_size_;
                break;
            case 2: // KEYSWITCHING_METHOD_II
                dimension = gk.d_;
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

        for (auto& galois : gk.galois_elt)
        {
            DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                               options.stream_);

            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, options.stream_>>>(
                output_memory.data(),
                all_gk[0].device_location_[galois.second].data(),
                modulus_->data(), n_power, Q_prime_size_, dimension, true);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            for (int i = 1; i < participant_count; i++)
            {
                multi_party_galoiskey_gen_method_I_II_kernel<<<
                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                    options.stream_>>>(
                    output_memory.data(),
                    all_gk[i].device_location_[galois.second].data(),
                    modulus_->data(), n_power, Q_prime_size_, dimension, false);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            if (options.storage_ == storage_type::DEVICE)
            {
                gk.device_location_[galois.second] = std::move(output_memory);
            }
            else
            {
                gk.host_location_[galois.second] =
                    HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.host_location_[galois.second].data(),
                                output_memory.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }

        DeviceVector<Data64> output_memory(gk.galoiskey_size_, options.stream_);

        multi_party_galoiskey_gen_method_I_II_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, options.stream_>>>(
            output_memory.data(), all_gk[0].zero_device_location_.data(),
            modulus_->data(), n_power, Q_prime_size_, dimension, true);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, options.stream_>>>(
                output_memory.data(), all_gk[i].zero_device_location_.data(),
                modulus_->data(), n_power, Q_prime_size_, dimension, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (options.storage_ == storage_type::DEVICE)
        {
            gk.zero_device_location_ = std::move(output_memory);
        }
        else
        {
            gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
            cudaMemcpyAsync(gk.zero_host_location_.data(), output_memory.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, options.stream_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (options.keep_initial_condition_)
        {
            for (int i = 0; i < participant_count; i++)
            {
                if (input_storage_types[i] == storage_type::DEVICE)
                {
                    // pass
                }
                else
                {
                    all_gk[i].store_in_host();
                }
            }
        }
    }

    __host__ void HEKeyGenerator<Scheme::BFV>::generate_switch_key_method_I(
        Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
        Secretkey<Scheme::BFV>& old_sk, const ExecutionOptions& options)
    {
        if (!old_sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!new_sk.secret_key_generated_)
        {
            throw std::logic_error("Ner Secretkey is not generated!");
        }

        if (swk.switch_key_generated_)
        {
            throw std::logic_error("Switchkey is already generated!");
        }

        input_storage_manager(
            old_sk,
            [&](Secretkey<Scheme::BFV>& old_sk_)
            {
                input_storage_manager(
                    new_sk,
                    [&](Secretkey<Scheme::BFV>& new_sk_)
                    {
                        output_storage_manager(
                            swk,
                            [&](Switchkey<Scheme::BFV>& swk_)
                            {
                                DeviceVector<Data64> errors_a(
                                    2 * Q_prime_size_ * Q_size_ * n,
                                    options.stream_);
                                Data64* error_poly = errors_a.data();
                                Data64* a_poly =
                                    error_poly + (Q_prime_size_ * Q_size_ * n);

                                RandomNumberGenerator::instance()
                                    .modular_uniform_random_number_generation(
                                        a_poly, modulus_->data(), n_power,
                                        Q_prime_size_, Q_size_,
                                        options.stream_);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, error_poly,
                                        modulus_->data(), n_power,
                                        Q_prime_size_, Q_size_,
                                        options.stream_);

                                gpuntt::ntt_rns_configuration<Data64> cfg_ntt =
                                    {.n_power = n_power,
                                     .ntt_type = gpuntt::FORWARD,
                                     .reduction_poly =
                                         gpuntt::ReductionPolynomial::X_N_plus,
                                     .zero_padding = false,
                                     .stream = options.stream_};

                                gpuntt::GPU_NTT_Inplace(
                                    error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_size_ * Q_prime_size_, Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    swk_.switchkey_size_, options.stream_);

                                switchkey_gen_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    output_memory.data(), new_sk.data(),
                                    old_sk.data(), error_poly, a_poly,
                                    modulus_->data(), factor_->data(), n_power,
                                    Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                swk.memory_set(std::move(output_memory));

                                swk.switch_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::BFV>::generate_bfv_switch_key_method_II(
        Switchkey<Scheme::BFV>& swk, Secretkey<Scheme::BFV>& new_sk,
        Secretkey<Scheme::BFV>& old_sk, const ExecutionOptions& options)
    {
        if (!old_sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is not generated!");
        }

        if (!new_sk.secret_key_generated_)
        {
            throw std::logic_error("Ner Secretkey is not generated!");
        }

        if (swk.switch_key_generated_)
        {
            throw std::logic_error("Switchkey is already generated!");
        }

        input_storage_manager(
            old_sk,
            [&](Secretkey<Scheme::BFV>& old_sk_)
            {
                input_storage_manager(
                    new_sk,
                    [&](Secretkey<Scheme::BFV>& new_sk_)
                    {
                        output_storage_manager(
                            swk,
                            [&](Switchkey<Scheme::BFV>& swk_)
                            {
                                DeviceVector<Data64> errors_a(
                                    2 * Q_prime_size_ * d_ * n,
                                    options.stream_);
                                Data64* error_poly = errors_a.data();
                                Data64* a_poly =
                                    error_poly + (Q_prime_size_ * d_ * n);

                                RandomNumberGenerator::instance()
                                    .modular_uniform_random_number_generation(
                                        a_poly, modulus_->data(), n_power,
                                        Q_prime_size_, d_, options.stream_);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, error_poly,
                                        modulus_->data(), n_power,
                                        Q_prime_size_, d_, options.stream_);

                                gpuntt::ntt_rns_configuration<Data64> cfg_ntt =
                                    {.n_power = n_power,
                                     .ntt_type = gpuntt::FORWARD,
                                     .reduction_poly =
                                         gpuntt::ReductionPolynomial::X_N_plus,
                                     .zero_padding = false,
                                     .stream = options.stream_};

                                gpuntt::GPU_NTT_Inplace(
                                    error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_ * Q_prime_size_, Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    swk_.switchkey_size_, options.stream_);

                                switchkey_gen_II_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    output_memory.data(), new_sk.data(),
                                    old_sk.data(), error_poly, a_poly,
                                    modulus_->data(), factor_->data(),
                                    Sk_pair_->data(), n_power, Q_prime_size_,
                                    d_, Q_size_, P_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                swk.memory_set(std::move(output_memory));

                                swk.switch_key_generated_ = true;
                            },
                            options);
                    },
                    options, false);
            },
            options, false);
    }

} // namespace heongpu