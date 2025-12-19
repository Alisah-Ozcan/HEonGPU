// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ckks/keygenerator.cuh"

namespace heongpu
{
    __host__ HEKeyGenerator<Scheme::CKKS>::HEKeyGenerator(
        HEContext<Scheme::CKKS>& context)
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

        d_leveled_ = context.d_leveled;
        d_tilda_leveled_ = context.d_tilda_leveled;
        r_prime_leveled_ = context.r_prime_leveled;

        B_prime_leveled_ = context.B_prime_leveled;
        B_prime_ntt_tables_leveled_ = context.B_prime_ntt_tables_leveled;
        B_prime_intt_tables_leveled_ = context.B_prime_intt_tables_leveled;
        B_prime_n_inverse_leveled_ = context.B_prime_n_inverse_leveled;

        Mi_inv_B_to_D_leveled_ = context.Mi_inv_B_to_D_leveled;
        base_change_matrix_D_to_B_leveled_ =
            context.base_change_matrix_D_to_B_leveled;
        base_change_matrix_B_to_D_leveled_ =
            context.base_change_matrix_B_to_D_leveled;
        Mi_inv_D_to_B_leveled_ = context.Mi_inv_D_to_B_leveled;
        prod_D_to_B_leveled_ = context.prod_D_to_B_leveled;
        prod_B_to_D_leveled_ = context.prod_B_to_D_leveled;

        I_j_leveled_ = context.I_j_leveled;
        I_location_leveled_ = context.I_location_leveled;
        Sk_pair_leveled_ = context.Sk_pair_leveled;

        prime_location_leveled_ = context.prime_location_leveled;
    }

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_secret_key(
        Secretkey<Scheme::CKKS>& sk, const ExecutionOptions& options)
    {
        if (sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
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
                    .ntt_layout = gpuntt::PerPolynomial,                    
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

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_secret_key_v2(
        Secretkey<Scheme::CKKS>& sk, const ExecutionOptions& options)
    {
        if (sk.secret_key_generated_)
        {
            throw std::logic_error("Secretkey is already generated!");
        }

        input_storage_manager(
            sk,
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                DeviceVector<int> secret_key_without_rns((n), options.stream_);

                std::vector<int> nonzero_positions_host;
                std::vector<int> nonzero_values_host;

                std::vector<int> index(n);
                for (int i = 0; i < n; i++)
                {
                    index[i] = i;
                }

                std::mt19937 rng(seed_);
                for (int i = 0; i < sk_.hamming_weight_; i++)
                {
                    std::uniform_int_distribution<int> dist(i, n - 1);
                    int j = dist(rng);

                    std::swap(index[i], index[j]);
                    nonzero_positions_host.push_back(index[i]);

                    std::uniform_int_distribution<int> value_dist(0, 1);
                    int value = (value_dist(rng) == 0) ? -1 : 1;
                    nonzero_values_host.push_back(value);
                }

                DeviceVector<int> nonzero_positions_device =
                    DeviceVector<int>(nonzero_positions_host, options.stream_);
                DeviceVector<int> nonzero_values_device =
                    DeviceVector<int>(nonzero_values_host, options.stream_);

                secretkey_gen_kernel_v2<<<dim3((n >> 8), 1, 1), 256, 0,
                                          options.stream_>>>(
                    secret_key_without_rns.data(),
                    nonzero_positions_device.data(),
                    nonzero_values_device.data(), sk_.hamming_weight_, n_power);
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
                    .ntt_layout = gpuntt::PerPolynomial,                    
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

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_public_key(
        Publickey<Scheme::CKKS>& pk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    pk,
                    [&](Publickey<Scheme::CKKS>& pk_)
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_relin_key_method_I(
        Relinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::CKKS>& rk_)
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
                            .ntt_layout = gpuntt::PerPolynomial,                            
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

    __host__ void
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_relin_key_method_II(
        Relinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::CKKS>& rk_)
                    {
                        DeviceVector<Data64> errors_a(
                            2 * Q_prime_size_ * d_leveled_->operator[](0) * n,
                            options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly =
                            error_poly +
                            (Q_prime_size_ * d_leveled_->operator[](0) * n);

                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_leveled_->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_,
                                d_leveled_->operator[](0), options.stream_);

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
                            cfg_ntt, d_leveled_->operator[](0) * Q_prime_size_,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        DeviceVector<Data64> output_memory(rk_.relinkey_size_,
                                                           options.stream_);

                        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                      1),
                                                 256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(),
                            Sk_pair_leveled_->operator[](0).data(), n_power,
                            Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                            P_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        rk_.memory_set(std::move(output_memory));

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_relin_key_method_III(
        Relinkey<Scheme::CKKS>& rk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                output_storage_manager(
                    rk,
                    [&](Relinkey<Scheme::CKKS>& rk_)
                    {
                        int max_depth = Q_size_ - 1;
                        DeviceVector<Data64> temp_calculation(
                            2 * Q_prime_size_ * d_leveled_->operator[](0) * n,
                            options.stream_);
                        DeviceVector<Data64> errors_a(
                            2 * Q_prime_size_ * d_leveled_->operator[](0) * n,
                            options.stream_);
                        Data64* error_poly = errors_a.data();
                        Data64* a_poly =
                            error_poly +
                            (Q_prime_size_ * d_leveled_->operator[](0) * n);

                        rk.device_location_leveled_.resize(max_depth);

                        for (int i = 0; i < max_depth; i++)
                        {
                            int d = d_leveled_->operator[](i);
                            int d_tilda = d_tilda_leveled_->operator[](i);
                            int r_prime = r_prime_leveled_;

                            int counter = Q_prime_size_;
                            int location = 0;
                            for (int j = 0; j < i; j++)
                            {
                                location += counter;
                                counter--;
                            }

                            int depth_mod_size = Q_prime_size_ - i;

                            RandomNumberGenerator::instance()
                                .modular_uniform_random_number_generation(
                                    a_poly, modulus_->data(), n_power,
                                    depth_mod_size,
                                    prime_location_leveled_->data() + location,
                                    d, options.stream_);

                            RandomNumberGenerator::instance()
                                .modular_gaussian_random_number_generation(
                                    error_std_dev, error_poly, modulus_->data(),
                                    n_power, depth_mod_size,
                                    prime_location_leveled_->data() + location,
                                    d, options.stream_);

                            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                                .n_power = n_power,
                                .ntt_type = gpuntt::FORWARD,
                                .ntt_layout = gpuntt::PerPolynomial,                                
                                .reduction_poly =
                                    gpuntt::ReductionPolynomial::X_N_plus,
                                .zero_padding = false,
                                .stream = options.stream_};

                            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                                error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * depth_mod_size,
                                depth_mod_size,
                                prime_location_leveled_->data() + location);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());

                            relinkey_gen_II_leveled_kernel<<<
                                dim3((n >> 8), depth_mod_size, 1), 256, 0,
                                options.stream_>>>(
                                temp_calculation.data(), sk.data(), error_poly,
                                a_poly, modulus_->data(), factor_->data(),
                                Sk_pair_leveled_->operator[](i).data(), n_power,
                                depth_mod_size, d, Q_size_, P_size_,
                                prime_location_leveled_->data() + location);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());

                            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                                .n_power = n_power,
                                .ntt_type = gpuntt::INVERSE,
                                .ntt_layout = gpuntt::PerPolynomial,                                
                                .reduction_poly =
                                    gpuntt::ReductionPolynomial::X_N_plus,
                                .zero_padding = false,
                                .mod_inverse = n_inverse_->data(),
                                .stream = options.stream_};

                            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                                temp_calculation.data(), intt_table_->data(),
                                modulus_->data(), cfg_intt,
                                2 * depth_mod_size * d, depth_mod_size,
                                prime_location_leveled_->data() + location);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());

                            DeviceVector<Data64> output_memory(
                                rk_.relinkey_size_, options.stream_);

                            relinkey_DtoB_leveled_kernel<<<
                                dim3((n >> 8), d_tilda, (d << 1)), 256, 0,
                                options.stream_>>>(
                                temp_calculation.data(), output_memory.data(),
                                modulus_->data(), B_prime_leveled_->data(),
                                base_change_matrix_D_to_B_leveled_->operator[](
                                                                      i)
                                    .data(),
                                Mi_inv_D_to_B_leveled_->operator[](i).data(),
                                prod_D_to_B_leveled_->operator[](i).data(),
                                I_j_leveled_->operator[](i).data(),
                                I_location_leveled_->operator[](i).data(),
                                n_power, depth_mod_size, d_tilda, d, r_prime,
                                prime_location_leveled_->data() + location);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());

                            gpuntt::GPU_NTT_Inplace(
                                output_memory.data(),
                                B_prime_ntt_tables_leveled_->data(),
                                B_prime_leveled_->data(), cfg_ntt,
                                2 * d_tilda * d * r_prime, r_prime);
                            HEONGPU_CUDA_CHECK(cudaGetLastError());

                            rk_.memory_set(std::move(output_memory), i);
                        }

                        rk_.relin_key_generated_ = true;
                    },
                    options);
            },
            options, false);
    }

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_galois_key_method_I(
        Galoiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
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
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_galois_key_method_II(
        Galoiskey<Scheme::CKKS>& gk, Secretkey<Scheme::CKKS>& sk,
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
            [&](Secretkey<Scheme::CKKS>& sk_)
            {
                DeviceVector<Data64> errors_a(2 * Q_prime_size_ *
                                                  d_leveled_->operator[](0) * n,
                                              options.stream_);
                Data64* error_poly = errors_a.data();
                Data64* a_poly = error_poly + (Q_prime_size_ *
                                               d_leveled_->operator[](0) * n);

                if (!gk.customized)
                {
                    // Positive Row Shift
                    for (auto& galois : gk.galois_elt)
                    {
                        RandomNumberGenerator::instance()
                            .modular_uniform_random_number_generation(
                                a_poly, modulus_->data(), n_power,
                                Q_prime_size_, d_leveled_->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_,
                                d_leveled_->operator[](0), options.stream_);

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
                            cfg_ntt, d_leveled_->operator[](0) * Q_prime_size_,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois.second, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_leveled_->operator[](0).data(), n_power,
                            Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                            P_size_);
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
                            d_leveled_->operator[](0), options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_leveled_->operator[](0),
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,                        
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d_leveled_->operator[](0) * Q_prime_size_,
                        Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_leveled_->operator[](0).data(), n_power,
                        Q_prime_size_, d_leveled_->operator[](0), Q_size_,
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
                                Q_prime_size_, d_leveled_->operator[](0),
                                options.stream_);

                        RandomNumberGenerator::instance()
                            .modular_gaussian_random_number_generation(
                                error_std_dev, error_poly, modulus_->data(),
                                n_power, Q_prime_size_,
                                d_leveled_->operator[](0), options.stream_);

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
                            cfg_ntt, d_leveled_->operator[](0) * Q_prime_size_,
                            Q_prime_size_);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        int inv_galois = modInverse(galois_, 2 * n);

                        DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                           options.stream_);

                        galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_,
                                                       1),
                                                  256, 0, options.stream_>>>(
                            output_memory.data(), sk.data(), error_poly, a_poly,
                            modulus_->data(), factor_->data(), inv_galois,
                            Sk_pair_leveled_->operator[](0).data(), n_power,
                            Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                            P_size_);
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
                            d_leveled_->operator[](0), options.stream_);

                    RandomNumberGenerator::instance()
                        .modular_gaussian_random_number_generation(
                            error_std_dev, error_poly, modulus_->data(),
                            n_power, Q_prime_size_, d_leveled_->operator[](0),
                            options.stream_);

                    gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                        .n_power = n_power,
                        .ntt_type = gpuntt::FORWARD,
                        .ntt_layout = gpuntt::PerPolynomial,                        
                        .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                        .zero_padding = false,
                        .stream = options.stream_};

                    gpuntt::GPU_NTT_Inplace(
                        error_poly, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d_leveled_->operator[](0) * Q_prime_size_,
                        Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    DeviceVector<Data64> output_memory(gk.galoiskey_size_,
                                                       options.stream_);

                    galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                              256, 0, options.stream_>>>(
                        output_memory.data(), sk.data(), error_poly, a_poly,
                        modulus_->data(), factor_->data(), gk.galois_elt_zero,
                        Sk_pair_leveled_->operator[](0).data(), n_power,
                        Q_prime_size_, d_leveled_->operator[](0), Q_size_,
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

    __host__ void HEKeyGenerator<Scheme::CKKS>::generate_switch_key_method_I(
        Switchkey<Scheme::CKKS>& swk, Secretkey<Scheme::CKKS>& new_sk,
        Secretkey<Scheme::CKKS>& old_sk, const ExecutionOptions& options)
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
            [&](Secretkey<Scheme::CKKS>& old_sk_)
            {
                input_storage_manager(
                    new_sk,
                    [&](Secretkey<Scheme::CKKS>& new_sk_)
                    {
                        output_storage_manager(
                            swk,
                            [&](Switchkey<Scheme::CKKS>& swk_)
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
                                     .ntt_layout = gpuntt::PerPolynomial,                                     
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
    HEKeyGenerator<Scheme::CKKS>::generate_ckks_switch_key_method_II(
        Switchkey<Scheme::CKKS>& swk, Secretkey<Scheme::CKKS>& new_sk,
        Secretkey<Scheme::CKKS>& old_sk, const ExecutionOptions& options)
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
            [&](Secretkey<Scheme::CKKS>& old_sk_)
            {
                input_storage_manager(
                    new_sk,
                    [&](Secretkey<Scheme::CKKS>& new_sk_)
                    {
                        output_storage_manager(
                            swk,
                            [&](Switchkey<Scheme::CKKS>& swk_)
                            {
                                DeviceVector<Data64> errors_a(
                                    2 * Q_prime_size_ *
                                        d_leveled_->operator[](0) * n,
                                    options.stream_);
                                Data64* error_poly = errors_a.data();
                                Data64* a_poly =
                                    error_poly +
                                    (Q_prime_size_ * d_leveled_->operator[](0) *
                                     n);

                                RandomNumberGenerator::instance()
                                    .modular_uniform_random_number_generation(
                                        a_poly, modulus_->data(), n_power,
                                        Q_prime_size_,
                                        d_leveled_->operator[](0),
                                        options.stream_);

                                RandomNumberGenerator::instance()
                                    .modular_gaussian_random_number_generation(
                                        error_std_dev, error_poly,
                                        modulus_->data(), n_power,
                                        Q_prime_size_,
                                        d_leveled_->operator[](0),
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
                                    error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_leveled_->operator[](0) * Q_prime_size_,
                                    Q_prime_size_);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                DeviceVector<Data64> output_memory(
                                    swk_.switchkey_size_, options.stream_);

                                switchkey_gen_II_kernel<<<
                                    dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    options.stream_>>>(
                                    output_memory.data(), new_sk.data(),
                                    old_sk.data(), error_poly, a_poly,
                                    modulus_->data(), factor_->data(),
                                    Sk_pair_leveled_->operator[](0).data(),
                                    n_power, Q_prime_size_,
                                    d_leveled_->operator[](0), Q_size_,
                                    P_size_);
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