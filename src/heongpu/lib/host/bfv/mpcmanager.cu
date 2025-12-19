// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "bfv/mpcmanager.cuh"

namespace heongpu
{
    __host__ HEMultiPartyManager<Scheme::BFV>::HEMultiPartyManager(
        HEContext<Scheme::BFV>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        n = context.n;

        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;

        Q_size_ = context.Q_size;

        P_size_ = context.P_size;

        modulus_ = context.modulus_;

        last_q_modinv_ = context.last_q_modinv_;

        ntt_table_ = context.ntt_table_;

        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        half_ = context.half_p_;

        half_mod_ = context.half_mod_;

        factor_ = context.factor_;

        Sk_pair_ = context.Sk_pair_;

        d_ = context.d;

        plain_modulus_ = context.plain_modulus_;
        plain_modulus2_ = context.plain_modulus2_;

        Q_mod_t_ = context.Q_mod_t_;

        upper_threshold_ = context.upper_threshold_;

        coeeff_div_plainmod_ = context.coeeff_div_plainmod_;

        ////////////////////////////////////////

        gamma_ = context.gamma_;

        Qi_t_ = context.Qi_t_;

        Qi_gamma_ = context.Qi_gamma_;

        Qi_inverse_ = context.Qi_inverse_;

        mulq_inv_t_ = context.mulq_inv_t_;

        mulq_inv_gamma_ = context.mulq_inv_gamma_;

        inv_gamma_ = context.inv_gamma_;

        new_seed_ = RNGSeed();
    }

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_public_key_stage1(
        MultipartyPublickey<Scheme::BFV>& pk, Secretkey<Scheme::BFV>& sk,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory((2 * Q_prime_size_ * n), stream);

        RNGSeed common_seed = pk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);
        RandomNumberGenerator::instance()
            .modular_ternary_random_number_generation(
                a_poly, modulus_->data(), n_power, Q_prime_size_, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, modulus_->data(), n_power,
                Q_prime_size_, 1, stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(errors_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        publickey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                               stream>>>(output_memory.data(), sk.data(),
                                         error_poly, a_poly, modulus_->data(),
                                         n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        pk.memory_set(std::move(output_memory));
    }

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_public_key_stage2(
        std::vector<MultipartyPublickey<Scheme::BFV>>& all_pk,
        Publickey<Scheme::BFV>& pk, const cudaStream_t stream)
    {
        int participant_count = all_pk.size();

        DeviceVector<Data64> output_memory((2 * Q_prime_size_ * n), stream);

        global_memory_replace_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                                       stream>>>(all_pk[0].data(),
                                                 output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            threshold_pk_addition<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    stream>>>(
                all_pk[i].data(), output_memory.data(), output_memory.data(),
                modulus_->data(), n_power, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        pk.memory_set(std::move(output_memory));
    }

    //

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_relin_key_method_I_stage_1(
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_relin_key_method_I_stage_3(
        MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
        MultipartyRelinkey<Scheme::BFV>& rk_stage_2, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
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
                                     .ntt_layout = gpuntt::PerPolynomial,                                     
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

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_bfv_relin_key_method_II_stage_1(
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_bfv_relin_key_method_II_stage_3(
        MultipartyRelinkey<Scheme::BFV>& rk_stage_1,
        MultipartyRelinkey<Scheme::BFV>& rk_stage_2, Secretkey<Scheme::BFV>& sk,
        const ExecutionOptions& options)
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
                                     .ntt_layout = gpuntt::PerPolynomial,                                     
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_relin_key_stage_2(
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_relin_key_stage_4(
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

    //

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::generate_galois_key_method_I_stage_1(
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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
                        .ntt_layout = gpuntt::PerPolynomial,                        
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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
                        .ntt_layout = gpuntt::PerPolynomial,                        
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
    HEMultiPartyManager<Scheme::BFV>::generate_galois_key_method_II_stage_1(
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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
                        .ntt_layout = gpuntt::PerPolynomial,                        
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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
                        .ntt_layout = gpuntt::PerPolynomial,                        
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

    __host__ void HEMultiPartyManager<Scheme::BFV>::generate_galois_key_stage_2(
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

        gk.storage_type_ = options.storage_;
    }

    //

    __host__ void HEMultiPartyManager<Scheme::BFV>::partial_decrypt_stage_1(
        Ciphertext<Scheme::BFV>& ciphertext, Secretkey<Scheme::BFV>& sk,
        Ciphertext<Scheme::BFV>& partial_ciphertext, const cudaStream_t stream)
    {
        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);

        DeviceVector<Data64> temp_memory(n * Q_size_, stream);
        Data64* ct1_temp = temp_memory.data();

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!ciphertext.in_ntt_domain_)
        {
            gpuntt::GPU_NTT(ct1, ct1_temp, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_, Q_size_);

            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1_temp, sk.data(), ct1_temp, modulus_->data(), n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1, sk.data(), ct1_temp, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        gpuntt::GPU_INTT(ct1_temp, output_memory.data() + (Q_size_ * n),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        Q_size_, Q_size_);

        DeviceVector<Data64> error_poly(Q_size_ * n, stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly.data(), modulus_->data(), n_power,
                Q_size_, 1, stream);

        // TODO: Optimize it!
        addition<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            output_memory.data() + (Q_size_ * n), error_poly.data(),
            output_memory.data() + (Q_size_ * n), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                       stream>>>(ct0, output_memory.data(),
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        partial_ciphertext.memory_set(std::move(output_memory));
    }

    __host__ void HEMultiPartyManager<Scheme::BFV>::partial_decrypt_stage_2(
        std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
        Plaintext<Scheme::BFV>& plaintext, const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(n, stream);

        int cipher_count = ciphertexts.size();

        DeviceVector<Data64> temp_sum(Q_size_ << n_power, stream);

        Data64* ct0 = ciphertexts[0].data();
        Data64* ct1 = ciphertexts[0].data() + (Q_size_ << n_power);
        addition<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            ct0, ct1, temp_sum.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            Data64* ct1_i = ciphertexts[i].data() + (Q_size_ << n_power);

            addition<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1_i, temp_sum.data(), temp_sum.data(), modulus_->data(),
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        decryption_fusion_bfv_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            temp_sum.data(), output_memory.data(), modulus_->data(),
            plain_modulus_, gamma_, Qi_t_->data(), Qi_gamma_->data(),
            Qi_inverse_->data(), mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::distributed_bootstrapping_stage1(
        Ciphertext<Scheme::BFV>& common, Ciphertext<Scheme::BFV>& output,
        Secretkey<Scheme::BFV>& secret_key, const RNGSeed& seed,
        const cudaStream_t stream)
    {
        RNGSeed common_seed = seed;

        DeviceVector<Data64> temp_vector(4 * Q_size_ * n, stream);
        Data64* error_poly = temp_vector.data();
        Data64* a_poly = error_poly + (2 * Q_size_ * n);
        Data64* random_t_poly = a_poly + (Q_size_ * n);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                a_poly, modulus_->data(), n_power, Q_size_, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        RandomNumberGenerator::instance()
            .modular_gaussian_random_number_generation(
                error_std_dev, error_poly, modulus_->data(), n_power, Q_size_,
                2, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                random_t_poly, plain_modulus2_->data(), n_power, 1, 1, stream);

        DeviceVector<Data64> output_memory((2 * Q_size_ * n), stream);

        Data64* ct0 = common.data();
        Data64* ct1 = common.data() + (Q_size_ << n_power);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!common.in_ntt_domain_)
        {
            DeviceVector<Data64> temp_memory(n * Q_size_, stream);
            gpuntt::GPU_NTT(ct1, temp_memory.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, Q_size_, Q_size_);

            col_boot_dec_mul_with_sk<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream>>>(
                temp_memory.data(), a_poly, secret_key.data(),
                output_memory.data(), modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            col_boot_dec_mul_with_sk<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream>>>(
                ct1, a_poly, secret_key.data(), output_memory.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(output_memory.data(), intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_size_,
                                Q_size_);

        col_boot_add_random_and_errors<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                         stream>>>(
            output_memory.data(), error_poly, random_t_poly, modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEMultiPartyManager<Scheme::BFV>::distributed_bootstrapping_stage2(
        std::vector<Ciphertext<Scheme::BFV>>& ciphertexts,
        Ciphertext<Scheme::BFV>& common, Ciphertext<Scheme::BFV>& output,
        const RNGSeed& seed, const cudaStream_t stream)
    {
        RNGSeed common_seed = seed;
        int cipher_count = ciphertexts.size();

        DeviceVector<Data64> h_memory((2 * Q_size_ * n), stream);

        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream>>>(ciphertexts[0].data(),
                                                 h_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            addition<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
                ciphertexts[i].data(), h_memory.data(), h_memory.data(),
                modulus_->data(), n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        DeviceVector<Data64> rand_message_memory(n, stream);

        Data64* h0 = h_memory.data();
        Data64* h1 = h_memory.data() + (Q_size_ << n_power);

        decryption_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            common.data(), h0, rand_message_memory.data(), modulus_->data(),
            plain_modulus_, gamma_, Qi_t_->data(), Qi_gamma_->data(),
            Qi_inverse_->data(), mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //////////////////////////////

        DeviceVector<Data64> output_memory((2 * Q_size_ * n), stream);

        Data64* ct0 = output_memory.data();
        Data64* ct1 = output_memory.data() + (Q_size_ << n_power);

        RandomNumberGenerator::instance().set(
            common_seed.key_, common_seed.nonce_,
            common_seed.personalization_string_, stream);

        RandomNumberGenerator::instance()
            .modular_uniform_random_number_generation(
                ct1, modulus_->data(), n_power, Q_size_, 1, stream);

        RNGSeed gen_seed;
        RandomNumberGenerator::instance().set(gen_seed.key_, gen_seed.nonce_,
                                              gen_seed.personalization_string_,
                                              stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,            
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_INTT_Inplace(ct1, intt_table_->data(), modulus_->data(),
                                cfg_intt, Q_size_, Q_size_);

        col_boot_enc<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            ct0, h1, rand_message_memory.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power, Q_size_);

        output.memory_set(std::move(output_memory));
    }

} // namespace heongpu