// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "keygenerator.cuh"

namespace heongpu
{
    __host__ HEKeyGenerator::HEKeyGenerator(Parameters& context)
    {
        scheme = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

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

        d_leveled_ = context.d_leveled;
        d_tilda_leveled_ = context.d_tilda_leveled;
        r_prime_leveled_ = context.r_prime_leveled;

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

    __host__ void HEKeyGenerator::generate_secret_key(Secretkey& sk,
                                                      cudaStream_t stream)
    {
        if (sk.secretkey_.size() < n)
        {
            sk.secretkey_ = DeviceVector<int>(n, stream);
        }

        secretkey_gen_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            sk.secretkey_.data(), sk.hamming_weight_, n_power, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (sk.location_.size() < Q_prime_size_ * n)
        {
            sk.location_ = DeviceVector<Data64>(Q_prime_size_ * n, stream);
        }

        secretkey_rns_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            sk.secretkey_.data(), sk.data(), modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(sk.data(), ntt_table_->data(), modulus_->data(),
                                cfg_ntt, Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        sk.in_ntt_domain_ = true;
    }

    __host__ void HEKeyGenerator::generate_public_key(Publickey& pk,
                                                      Secretkey& sk,
                                                      cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretky size is not valid || Secretkey is not generated!");
        }

        if (pk.locations_.size() < (2 * Q_prime_size_ * n))
        {
            pk.locations_ = DeviceVector<Data64>(2 * Q_prime_size_ * n, stream);
        }

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                           256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(errors_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        publickey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                               stream>>>(pk.data(), sk.data(), error_poly,
                                         a_poly, modulus_->data(), n_power,
                                         Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        pk.in_ntt_domain_ = true;
    }

    __host__ void HEKeyGenerator::generate_multi_party_public_key_piece(
        MultipartyPublickey& pk, Secretkey& sk, cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        if (pk.locations_.size() < (2 * Q_prime_size_ * n))
        {
            pk.locations_ = DeviceVector<Data64>(2 * Q_prime_size_ * n, stream);
        }

        int common_seed = pk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                           256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(errors_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        publickey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                               stream>>>(pk.data(), sk.data(), error_poly,
                                         a_poly, modulus_->data(), n_power,
                                         Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        pk.in_ntt_domain_ = true;
    }

    __host__ void HEKeyGenerator::generate_multi_party_public_key(
        std::vector<MultipartyPublickey>& all_pk, Publickey& pk,
        cudaStream_t stream)
    {
        int participant_count = all_pk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
        }

        for (int i = 0; i < participant_count; i++)
        {
            if ((all_pk[i].locations_.size() < (2 * Q_prime_size_ * n)))
            {
                throw std::invalid_argument(
                    "MultipartyPublickey size is not valid || "
                    "MultipartyPublickey is not generated!");
            }
        }

        if ((pk.locations_.size() < (2 * Q_prime_size_ * n)))
        {
            pk.locations_ = DeviceVector<Data64>(2 * Q_prime_size_ * n, stream);
        }

        global_memory_replace_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                                       stream>>>(all_pk[0].data(), pk.data(),
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            threshold_pk_addition<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                    stream>>>(all_pk[i].data(), pk.data(),
                                              pk.data(), modulus_->data(),
                                              n_power, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void
    HEKeyGenerator::generate_relin_key_method_I(Relinkey& rk, Secretkey& sk,
                                                const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_.resize(rk.relinkey_size_, stream);
        relinkey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                              stream>>>(
            rk.device_location_.data(), sk.data(), error_poly, a_poly,
            modulus_->data(), factor_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void
    HEKeyGenerator::generate_multi_party_relin_key_piece_method_I_stage_I(
        MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = rk.seed();

        DeviceVector<Data64> random_values(
            Q_prime_size_ * ((3 * Q_size_) + 1) * n, stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * Q_size_ * n);
        Data64* u = e1 + (Q_prime_size_ * Q_size_ * n);
        Data64* common_a = u + (Q_prime_size_ * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            common_a, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), 2 * Q_size_, 1), 256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            random_values.data(), ntt_table_->data(), modulus_->data(), cfg_ntt,
            Q_prime_size_ * ((2 * Q_size_) + 1), Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        multi_party_relinkey_piece_method_I_stage_I_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            rk.device_location_.data(), sk.data(), common_a, u, e0, e1,
            modulus_->data(), factor_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void
    HEKeyGenerator::generate_multi_party_relin_key_piece_method_I_stage_II(
        MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
        Secretkey& sk, const cudaStream_t stream)
    {
        if (rk_stage_1.relinkey_size_ != rk_stage_2.relinkey_size_)
        {
            throw std::invalid_argument(
                "MultipartyRelinkey contexts are not valid!");
        }

        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        if (rk_stage_1.store_in_gpu_)
        {
            if (rk_stage_1.device_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }
        }
        else
        {
            if (rk_stage_1.host_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }

            rk_stage_1.store_in_device();
        }

        DeviceVector<Data64> random_values(
            Q_prime_size_ * ((2 * Q_size_) + 1) * n, stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * Q_size_ * n);
        Data64* u = e1 + (Q_prime_size_ * Q_size_ * n);

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 2, 1),
                                                           256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            random_values.data(), ntt_table_->data(), modulus_->data(), cfg_ntt,
            Q_prime_size_ * ((2 * Q_size_) + 1), Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk_stage_2.device_location_ =
            DeviceVector<Data64>(rk_stage_2.relinkey_size_, stream);

        if (rk_stage_1.store_in_gpu_)
        {
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                rk_stage_1.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_location(rk_stage_1.device_location_,
                                               stream);
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp_location.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk_stage_2.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk_stage_2.host_location_ =
                HostVector<Data64>(rk_stage_2.relinkey_size_);
            cudaMemcpyAsync(rk_stage_2.host_location_.data(),
                            rk_stage_2.device_location_.data(),
                            rk_stage_2.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk_stage_2.device_location_.resize(0, stream);
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_multi_party_relin_key_piece_method_II_stage_I(
        MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = rk.seed();

        DeviceVector<Data64> random_values(Q_prime_size_ * ((3 * d_) + 1) * n,
                                           stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * d_ * n);
        Data64* u = e1 + (Q_prime_size_ * d_ * n);
        Data64* common_a = u + (Q_prime_size_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), d_, 1),
                                                          256, 0, stream>>>(
            common_a, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), 2 * d_, 1), 256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(random_values.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_prime_size_ * ((2 * d_) + 1), Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        multi_party_relinkey_piece_method_II_stage_I_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            rk.device_location_.data(), sk.data(), common_a, u, e0, e1,
            modulus_->data(), factor_->data(), Sk_pair_->data(), n_power,
            Q_prime_size_, d_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_multi_party_relin_key_piece_method_II_stage_II(
        MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
        Secretkey& sk, const cudaStream_t stream)
    {
        if (rk_stage_1.relinkey_size_ != rk_stage_2.relinkey_size_)
        {
            throw std::invalid_argument(
                "MultipartyRelinkey contexts are not valid!");
        }

        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        if (rk_stage_1.store_in_gpu_)
        {
            if (rk_stage_1.device_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }
        }
        else
        {
            if (rk_stage_1.host_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }

            rk_stage_1.store_in_device();
        }

        DeviceVector<Data64> random_values(Q_prime_size_ * ((2 * d_) + 1) * n,
                                           stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * d_ * n);
        Data64* u = e1 + (Q_prime_size_ * d_ * n);

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 2, 1),
                                                           256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(random_values.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_prime_size_ * ((2 * d_) + 1), Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk_stage_2.device_location_ =
            DeviceVector<Data64>(rk_stage_2.relinkey_size_, stream);

        if (rk_stage_1.store_in_gpu_)
        {
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                rk_stage_1.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                d_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_location(rk_stage_1.device_location_,
                                               stream);
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp_location.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                d_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk_stage_2.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk_stage_2.host_location_ =
                HostVector<Data64>(rk_stage_2.relinkey_size_);
            cudaMemcpyAsync(rk_stage_2.host_location_.data(),
                            rk_stage_2.device_location_.data(),
                            rk_stage_2.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk_stage_2.device_location_.resize(0, stream);
        }
    }

    __host__ void
    HEKeyGenerator::generate_ckks_multi_party_relin_key_piece_method_II_stage_I(
        MultipartyRelinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = rk.seed();

        DeviceVector<Data64> random_values(
            Q_prime_size_ * ((3 * d_leveled_->operator[](0)) + 1) * n, stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * d_leveled_->operator[](0) * n);
        Data64* u = e1 + (Q_prime_size_ * d_leveled_->operator[](0) * n);
        Data64* common_a = u + (Q_prime_size_ * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            common_a, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), 2 * d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            random_values.data(), ntt_table_->data(), modulus_->data(), cfg_ntt,
            Q_prime_size_ * ((2 * d_leveled_->operator[](0)) + 1),
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        multi_party_relinkey_piece_method_II_stage_I_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            rk.device_location_.data(), sk.data(), common_a, u, e0, e1,
            modulus_->data(), factor_->data(),
            Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
            d_leveled_->operator[](0), Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::
        generate_ckks_multi_party_relin_key_piece_method_II_stage_II(
            MultipartyRelinkey& rk_stage_1, MultipartyRelinkey& rk_stage_2,
            Secretkey& sk, const cudaStream_t stream)
    {
        if (rk_stage_1.relinkey_size_ != rk_stage_2.relinkey_size_)
        {
            throw std::invalid_argument(
                "MultipartyRelinkey contexts are not valid!");
        }

        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        if (rk_stage_1.store_in_gpu_)
        {
            if (rk_stage_1.device_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }
        }
        else
        {
            if (rk_stage_1.host_location_.size() < rk_stage_1.relinkey_size_)
            {
                throw std::invalid_argument(
                    "MultipartyRelinkey size is not valid || "
                    "MultipartyRelinkey is not generated!");
            }

            rk_stage_1.store_in_device();
        }

        DeviceVector<Data64> random_values(
            Q_prime_size_ * ((2 * d_leveled_->operator[](0)) + 1) * n, stream);
        Data64* e0 = random_values.data();
        Data64* e1 = e0 + (Q_prime_size_ * d_leveled_->operator[](0) * n);
        Data64* u = e1 + (Q_prime_size_ * d_leveled_->operator[](0) * n);

        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 2, 1),
                                                           256, 0, stream>>>(
            e0, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_ternary_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                          256, 0, stream>>>(
            u, modulus_->data(), n_power, Q_prime_size_, seed_,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            random_values.data(), ntt_table_->data(), modulus_->data(), cfg_ntt,
            Q_prime_size_ * ((2 * d_leveled_->operator[](0)) + 1),
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk_stage_2.device_location_ =
            DeviceVector<Data64>(rk_stage_2.relinkey_size_, stream);

        if (rk_stage_1.store_in_gpu_)
        {
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                rk_stage_1.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0));
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_location(rk_stage_1.device_location_,
                                               stream);
            multi_party_relinkey_piece_method_I_II_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp_location.data(), rk_stage_2.device_location_.data(),
                sk.data(), u, e0, e1, modulus_->data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0));
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk_stage_2.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk_stage_2.host_location_ =
                HostVector<Data64>(rk_stage_2.relinkey_size_);
            cudaMemcpyAsync(rk_stage_2.host_location_.data(),
                            rk_stage_2.device_location_.data(),
                            rk_stage_2.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk_stage_2.device_location_.resize(0, stream);
        }
    }

    //////////////////////
    //////////////////////

    __host__ void HEKeyGenerator::generate_multi_party_relin_key(
        std::vector<MultipartyRelinkey>& all_rk, MultipartyRelinkey& rk,
        cudaStream_t stream)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
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

        for (int i = 0; i < participant_count; i++)
        {
            if (all_rk[i].store_in_gpu_)
            {
                if (all_rk[i].device_location_.size() <
                    all_rk[i].relinkey_size_)
                {
                    throw std::invalid_argument(
                        "MultipartyRelinkey size is not valid || "
                        "MultipartyRelinkey is not generated!");
                }
            }
            else
            {
                if (all_rk[i].host_location_.size() < all_rk[i].relinkey_size_)
                {
                    throw std::invalid_argument(
                        "MultipartyRelinkey size is not valid || "
                        "MultipartyRelinkey is not generated!");
                }

                all_rk[i].store_in_device();
            }
        }

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        multi_party_relinkey_method_I_stage_I_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            all_rk[0].data(), rk.data(), modulus_->data(), n_power,
            Q_prime_size_, dimension, true);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            multi_party_relinkey_method_I_stage_I_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                all_rk[i].data(), rk.device_location_.data(), modulus_->data(),
                n_power, Q_prime_size_, dimension, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_multi_party_relin_key(
        std::vector<MultipartyRelinkey>& all_rk,
        MultipartyRelinkey& rk_common_stage1, Relinkey& rk, cudaStream_t stream)
    {
        int participant_count = all_rk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
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

        for (int i = 0; i < participant_count; i++)
        {
            if (all_rk[i].store_in_gpu_)
            {
                if (all_rk[i].device_location_.size() <
                    all_rk[i].relinkey_size_)
                {
                    throw std::invalid_argument(
                        "MultipartyRelinkey size is not valid || "
                        "MultipartyRelinkey is not generated!");
                }
            }
            else
            {
                if (all_rk[i].host_location_.size() < all_rk[i].relinkey_size_)
                {
                    throw std::invalid_argument(
                        "MultipartyRelinkey size is not valid || "
                        "MultipartyRelinkey is not generated!");
                }

                all_rk[i].store_in_device();
            }
        }

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);

        if (!rk_common_stage1.store_in_gpu_)
        {
            rk_common_stage1.store_in_device();
        }

        multi_party_relinkey_method_I_stage_II_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            all_rk[0].data(), rk_common_stage1.data(),
            rk.device_location_.data(), modulus_->data(), n_power,
            Q_prime_size_, dimension);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            multi_party_relinkey_method_I_stage_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                all_rk[i].data(), rk.device_location_.data(), modulus_->data(),
                n_power, Q_prime_size_, dimension);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_bfv_relin_key_method_II(
        Relinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), d_, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), d_, 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d_ * Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                 stream>>>(
            rk.device_location_.data(), sk.data(), error_poly, a_poly,
            modulus_->data(), factor_->data(), Sk_pair_->data(), n_power,
            Q_prime_size_, d_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_bfv_relin_key_method_III(
        Relinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), d_, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), d_, 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d_ * Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Data64> temp_calculation(2 * Q_prime_size_ * d_ * n,
                                              stream);

        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                 stream>>>(
            temp_calculation.data(), sk.data(), error_poly, a_poly,
            modulus_->data(), factor_->data(), Sk_pair_->data(), n_power,
            Q_prime_size_, d_, Q_size_, P_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp_calculation.data(), intt_table_->data(),
                                modulus_->data(), cfg_intt,
                                2 * Q_prime_size_ * d_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        relinkey_DtoB_kernel<<<dim3((n >> 8), d_tilda_, (d_ << 1)), 256, 0,
                               stream>>>(
            temp_calculation.data(), rk.device_location_.data(),
            modulus_->data(), B_prime_->data(),
            base_change_matrix_D_to_B_->data(), Mi_inv_D_to_B_->data(),
            prod_D_to_B_->data(), I_j_->data(), I_location_->data(), n_power,
            Q_prime_size_, d_tilda_, d_, r_prime_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(
            rk.device_location_.data(), B_prime_ntt_tables_->data(),
            B_prime_->data(), cfg_ntt, 2 * d_tilda_ * d_ * r_prime_, r_prime_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_ckks_relin_key_method_II(
        Relinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (Q_prime_size_ * d_leveled_->operator[](0) * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        rk.device_location_ = DeviceVector<Data64>(rk.relinkey_size_, stream);
        relinkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                 stream>>>(
            rk.device_location_.data(), sk.data(), error_poly, a_poly,
            modulus_->data(), factor_->data(),
            Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
            d_leveled_->operator[](0), Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            rk.host_location_ = HostVector<Data64>(rk.relinkey_size_);
            cudaMemcpyAsync(rk.host_location_.data(),
                            rk.device_location_.data(),
                            rk.relinkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_ckks_relin_key_method_III(
        Relinkey& rk, Secretkey& sk, const cudaStream_t stream)
    {
        int max_depth = Q_size_ - 1;
        DeviceVector<Data64> temp_calculation(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        DeviceVector<Data64> errors_a(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (Q_prime_size_ * d_leveled_->operator[](0) * n);

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

            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), d, 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, depth_mod_size, seed_,
                offset_, prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, depth_mod_size, seed_,
                offset_, prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                2 * depth_mod_size, depth_mod_size,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            relinkey_gen_II_leveled_kernel<<<dim3((n >> 8), depth_mod_size, 1),
                                             256, 0, stream>>>(
                temp_calculation.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(),
                Sk_pair_leveled_->operator[](i).data(), n_power, depth_mod_size,
                d, Q_size_, P_size_,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = n_power,
                .ntt_type = gpuntt::INVERSE,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = n_inverse_->data(),
                .stream = stream};

            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                temp_calculation.data(), intt_table_->data(), modulus_->data(),
                cfg_intt, 2 * depth_mod_size * d, depth_mod_size,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.device_location_leveled_.push_back(
                DeviceVector<Data64>(rk.relinkey_size_leveled_[i], stream));
            relinkey_DtoB_leveled_kernel<<<dim3((n >> 8), d_tilda, (d << 1)),
                                           256, 0, stream>>>(
                temp_calculation.data(), rk.device_location_leveled_[i].data(),
                modulus_->data(), B_prime_leveled_->data(),
                base_change_matrix_D_to_B_leveled_->operator[](i).data(),
                Mi_inv_D_to_B_leveled_->operator[](i).data(),
                prod_D_to_B_leveled_->operator[](i).data(),
                I_j_leveled_->operator[](i).data(),
                I_location_leveled_->operator[](i).data(), n_power,
                depth_mod_size, d_tilda, d, r_prime,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT_Inplace(rk.device_location_leveled_[i].data(),
                                    B_prime_ntt_tables_leveled_->data(),
                                    B_prime_leveled_->data(), cfg_ntt,
                                    2 * d_tilda * d * r_prime, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (rk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            for (int i = 0; i < max_depth; i++)
            {
                rk.host_location_leveled_.push_back(
                    HostVector<Data64>(rk.relinkey_size_leveled_[i]));

                cudaMemcpyAsync(rk.host_location_leveled_[i].data(),
                                rk.device_location_leveled_[i].data(),
                                rk.relinkey_size_leveled_[i] * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                rk.device_location_leveled_[i].resize(0, stream);
            }
            rk.device_location_leveled_.resize(0);
            rk.device_location_leveled_.shrink_to_fit();
        }
    }

    __host__ void
    HEKeyGenerator::generate_galois_key_method_I(Galoiskey& gk, Secretkey& sk,
                                                 const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        Q_size_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                       stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_size_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                   stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero, n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        Q_size_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                       stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_size_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                   stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero, n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void HEKeyGenerator::generate_bfv_galois_key_method_II(
        Galoiskey& gk, Secretkey& sk, const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        d_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, Sk_pair_->data(), n_power, Q_prime_size_, d_,
                    Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        d_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                    P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void HEKeyGenerator::generate_ckks_galois_key_method_II(
        Galoiskey& gk, Secretkey& sk, const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (Q_prime_size_ * d_leveled_->operator[](0) * n);

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(a_poly, modulus_->data(), n_power, Q_prime_size_,
                              seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(error_poly, modulus_->data(), n_power,
                              Q_prime_size_, seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(
                    error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                    d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(
                error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_uniform_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(a_poly, modulus_->data(), n_power, Q_prime_size_,
                              seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(error_poly, modulus_->data(), n_power,
                              Q_prime_size_, seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(
                    error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                    d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_uniform_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                a_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(
                error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void
    HEKeyGenerator::generate_multi_party_galois_key_piece_method_I(
        MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        Q_size_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                       stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_size_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                   stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero, n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        Q_size_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                       stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    Q_size_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                   stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero, n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_multi_party_galois_key_piece_method_II(
        MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = gk.seed();

        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), d_, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        d_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, Sk_pair_->data(), n_power, Q_prime_size_, d_,
                    Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                    error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                    offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                        modulus_->data(), cfg_ntt,
                                        d_ * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                    P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_, 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt,
                                    d_ * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void
    HEKeyGenerator::generate_ckks_multi_party_galois_key_piece_method_II(
        MultipartyGaloiskey& gk, Secretkey& sk, const cudaStream_t stream)
    {
        if (sk.location_.size() < (Q_prime_size_ * n))
        {
            throw std::invalid_argument(
                "Secretkey size is not valid || Secretkey is not generated!");
        }

        int common_seed = gk.seed();

        DeviceVector<Data64> errors_a(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (Q_prime_size_ * d_leveled_->operator[](0) * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, common_seed,
            0); // offset should be zero
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(error_poly, modulus_->data(), n_power,
                              Q_prime_size_, seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(
                    error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                    d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois.second, 2 * n);

                gk.device_location_[galois.second] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois.second].data(), sk.data(),
                    error_poly, a_poly, modulus_->data(), factor_->data(),
                    inv_galois, Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(
                error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                modular_gaussian_random_number_generation_kernel<<<
                    dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0,
                    stream>>>(error_poly, modulus_->data(), n_power,
                              Q_prime_size_, seed_, offset_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                offset_++;

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = stream};

                gpuntt::GPU_NTT_Inplace(
                    error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                    d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                int inv_galois = modInverse(galois_, 2 * n);

                gk.device_location_[galois_] =
                    DeviceVector<Data64>(gk.galoiskey_size_, stream);
                galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                          0, stream>>>(
                    gk.device_location_[galois_].data(), sk.data(), error_poly,
                    a_poly, modulus_->data(), factor_->data(), inv_galois,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            // Columns Rotate
            modular_gaussian_random_number_generation_kernel<<<
                dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
                error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
                offset_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            offset_++;

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(
                error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
                d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.zero_device_location_ =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);
            galoiskey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                      stream>>>(
                gk.zero_device_location_.data(), sk.data(), error_poly, a_poly,
                modulus_->data(), factor_->data(), gk.galois_elt_zero,
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            if (gk.store_in_gpu_)
            {
                // pass
            }
            else
            {
                for (auto& galois_ : gk.device_location_)
                {
                    gk.host_location_[galois_.first] =
                        HostVector<Data64>(gk.galoiskey_size_);
                    cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                    galois_.second.data(),
                                    gk.galoiskey_size_ * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }

                gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.zero_host_location_.data(),
                                gk.zero_device_location_.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.device_location_.clear();
                gk.zero_device_location_.resize(0, stream);
            }
        }
    }

    __host__ void HEKeyGenerator::generate_multi_party_galois_key(
        std::vector<MultipartyGaloiskey>& all_gk, Galoiskey& gk,
        cudaStream_t stream)
    {
        int participant_count = all_gk.size();

        if (participant_count == 0)
        {
            throw std::invalid_argument(
                "No participant to generate common publickey!");
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

        for (int i = 0; i < participant_count; i++)
        {
            if ((gk.customized != all_gk[i].customized) ||
                (gk.group_order_ != all_gk[i].group_order_))
            {
                throw std::invalid_argument(
                    "MultipartyGaloiskey context is not valid || "
                    "MultipartyGaloiskey is not generated2!");
            }
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (gk.customized)
            {
                if ((gk.custom_galois_elt != all_gk[i].custom_galois_elt) ||
                    (gk.galois_elt_zero != all_gk[i].galois_elt_zero))
                {
                    throw std::invalid_argument(
                        "MultipartyGaloiskeys galois index do not match!");
                }
            }
            else
            {
                if ((gk.galois_elt.size() != all_gk[i].galois_elt.size()) ||
                    (gk.galois_elt_zero != all_gk[i].galois_elt_zero))
                {
                    throw std::invalid_argument(
                        "MultipartyGaloiskeys galois index do not match!");
                }

                for (const auto& pair : gk.galois_elt)
                {
                    auto it = all_gk[i].galois_elt.find(pair.first);
                    if (it == all_gk[i].galois_elt.end() ||
                        it->second != pair.second)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys galois index do not match!");
                    }
                }
            }
        }

        for (int i = 0; i < participant_count; i++)
        {
            if (all_gk[i].store_in_gpu_)
            {
                for (const auto& pair : all_gk[i].device_location_)
                {
                    if (pair.second.size() < all_gk[i].galoiskey_size_)
                    {
                        throw std::invalid_argument(
                            "MultipartyGaloiskeys size is not valid || "
                            "MultipartyGaloiskeys is not generated3!");
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
                            "MultipartyGaloiskeys is not generated4!");
                    }
                }

                all_gk[i].store_in_device();
            }
        }

        for (auto& galois : gk.galois_elt)
        {
            gk.device_location_[galois.second] =
                DeviceVector<Data64>(gk.galoiskey_size_, stream);

            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                gk.device_location_[galois.second].data(),
                all_gk[0].device_location_[galois.second].data(),
                modulus_->data(), n_power, Q_prime_size_, dimension, true);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            for (int i = 1; i < participant_count; i++)
            {
                multi_party_galoiskey_gen_method_I_II_kernel<<<
                    dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                    gk.device_location_[galois.second].data(),
                    all_gk[i].device_location_[galois.second].data(),
                    modulus_->data(), n_power, Q_prime_size_, dimension, false);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }

        ///////////

        gk.zero_device_location_ =
            DeviceVector<Data64>(gk.galoiskey_size_, stream);

        multi_party_galoiskey_gen_method_I_II_kernel<<<
            dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
            gk.zero_device_location_.data(),
            all_gk[0].zero_device_location_.data(), modulus_->data(), n_power,
            Q_prime_size_, dimension, true);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < participant_count; i++)
        {
            multi_party_galoiskey_gen_method_I_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                gk.zero_device_location_.data(),
                all_gk[i].zero_device_location_.data(), modulus_->data(),
                n_power, Q_prime_size_, dimension, false);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        if (gk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            for (auto& galois_ : gk.device_location_)
            {
                gk.host_location_[galois_.first] =
                    HostVector<Data64>(gk.galoiskey_size_);
                cudaMemcpyAsync(gk.host_location_[galois_.first].data(),
                                galois_.second.data(),
                                gk.galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            gk.zero_host_location_ = HostVector<Data64>(gk.galoiskey_size_);
            cudaMemcpyAsync(gk.zero_host_location_.data(),
                            gk.zero_device_location_.data(),
                            gk.galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gk.device_location_.clear();
            gk.zero_device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_switch_key_method_I(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * Q_size_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * Q_size_ * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        swk.device_location_ =
            DeviceVector<Data64>(swk.switchkey_size_, stream);
        switchkey_gen_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                               stream>>>(
            swk.device_location_.data(), new_sk.data(), old_sk.data(),
            error_poly, a_poly, modulus_->data(), factor_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            swk.host_location_ = HostVector<Data64>(swk.switchkey_size_);
            cudaMemcpyAsync(swk.host_location_.data(),
                            swk.device_location_.data(),
                            swk.switchkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_bfv_switch_key_method_II(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(2 * Q_prime_size_ * d_ * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly = error_poly + (Q_prime_size_ * d_ * n);

        modular_uniform_random_number_generation_kernel<<<dim3((n >> 8), d_, 1),
                                                          256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), d_, 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d_ * Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        swk.device_location_ =
            DeviceVector<Data64>(swk.switchkey_size_, stream);
        switchkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream>>>(
            swk.device_location_.data(), new_sk.data(), old_sk.data(),
            error_poly, a_poly, modulus_->data(), factor_->data(),
            Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            swk.host_location_ = HostVector<Data64>(swk.switchkey_size_);
            cudaMemcpyAsync(swk.host_location_.data(),
                            swk.device_location_.data(),
                            swk.switchkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.device_location_.resize(0, stream);
        }
    }

    __host__ void HEKeyGenerator::generate_ckks_switch_key_method_II(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk,
        const cudaStream_t stream)
    {
        DeviceVector<Data64> errors_a(
            2 * Q_prime_size_ * d_leveled_->operator[](0) * n, stream);
        Data64* error_poly = errors_a.data();
        Data64* a_poly =
            error_poly + (Q_prime_size_ * d_leveled_->operator[](0) * n);

        modular_uniform_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            a_poly, modulus_->data(), n_power, Q_prime_size_, seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        modular_gaussian_random_number_generation_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](0), 1), 256, 0, stream>>>(
            error_poly, modulus_->data(), n_power, Q_prime_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            error_poly, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](0) * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        swk.device_location_ =
            DeviceVector<Data64>(swk.switchkey_size_, stream);
        switchkey_gen_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream>>>(
            swk.device_location_.data(), new_sk.data(), old_sk.data(),
            error_poly, a_poly, modulus_->data(), factor_->data(),
            Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
            d_leveled_->operator[](0), Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            // pass
        }
        else
        {
            swk.host_location_ = HostVector<Data64>(swk.switchkey_size_);
            cudaMemcpyAsync(swk.host_location_.data(),
                            swk.device_location_.data(),
                            swk.switchkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.device_location_.resize(0, stream);
        }
    }

} // namespace heongpu