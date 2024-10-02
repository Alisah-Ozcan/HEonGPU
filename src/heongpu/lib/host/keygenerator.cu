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

    __host__ void HEKeyGenerator::generate_secret_key(Secretkey& sk)
    {
        sk_kernel<<<dim3((n >> 8), 1, 1), 256>>>(sk.data(), modulus_->data(),
                                                 n_power, Q_prime_size_,
                                                 seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(sk.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEKeyGenerator::generate_public_key(Publickey& pk,
                                                      Secretkey& sk)
    {
        DeviceVector<Data> errors_a(2 * Q_prime_size_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(errors_a.data(),
                                                    modulus_->data(), n_power,
                                                    Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(errors_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        pk_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256>>>(
            pk.data(), sk.data(), errors_a.data(), modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEKeyGenerator::generate_relin_key_method_I(Relinkey& rk,
                                                              Secretkey& sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);

        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            rk.device_location_ = DeviceVector<Data>(rk.relinkey_size_);
            relinkey_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                rk.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), n_power, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(rk.relinkey_size_);
            relinkey_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp_location.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), n_power, Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.host_location_ = HostVector<Data>(rk.relinkey_size_);
            cudaMemcpy(rk.data(), temp_location.data(),
                       rk.relinkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_relin_key_method_II(Relinkey& rk,
                                                     Secretkey& sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            rk.device_location_ = DeviceVector<Data>(rk.relinkey_size_);
            relinkey_kernel_externel_product<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256>>>(
                rk.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), Sk_pair_->data(), n_power, Q_prime_size_, d_,
                Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(rk.relinkey_size_);
            relinkey_kernel_externel_product<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256>>>(
                temp_location.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), Sk_pair_->data(), n_power, Q_prime_size_, d_,
                Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.host_location_ = HostVector<Data>(rk.relinkey_size_);
            cudaMemcpy(rk.data(), temp_location.data(),
                       rk.relinkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_relin_key_method_III(Relinkey& rk,
                                                      Secretkey& sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        DeviceVector<Data> temp_calculation(2 * Q_prime_size_ * d_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relinkey_kernel_externel_product<<<dim3((n >> 8), Q_prime_size_, 1),
                                           256>>>(
            temp_calculation.data(), sk.data(), e_a.data(), modulus_->data(),
            factor_->data(), Sk_pair_->data(), n_power, Q_prime_size_, d_,
            Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp_calculation.data(), intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_ * d_,
                        Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //////////////
        if (rk.store_in_gpu_)
        {
            rk.device_location_ = DeviceVector<Data>(rk.relinkey_size_);
            relinkey_DtoB_kernel<<<dim3((n >> 8), d_tilda_, (d_ << 1)), 256>>>(
                temp_calculation.data(), rk.data(), modulus_->data(),
                B_prime_->data(), base_change_matrix_D_to_B_->data(),
                Mi_inv_D_to_B_->data(), prod_D_to_B_->data(), I_j_->data(),
                I_location_->data(), n_power, Q_prime_size_, d_tilda_, d_,
                r_prime_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            GPU_NTT_Inplace(rk.data(), B_prime_ntt_tables_->data(),
                            B_prime_->data(), cfg_ntt,
                            2 * d_tilda_ * d_ * r_prime_, r_prime_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(rk.relinkey_size_);
            relinkey_DtoB_kernel<<<dim3((n >> 8), d_tilda_, (d_ << 1)), 256>>>(
                temp_calculation.data(), temp_location.data(), modulus_->data(),
                B_prime_->data(), base_change_matrix_D_to_B_->data(),
                Mi_inv_D_to_B_->data(), prod_D_to_B_->data(), I_j_->data(),
                I_location_->data(), n_power, Q_prime_size_, d_tilda_, d_,
                r_prime_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            GPU_NTT_Inplace(temp_location.data(), B_prime_ntt_tables_->data(),
                            B_prime_->data(), cfg_ntt,
                            2 * d_tilda_ * d_ * r_prime_, r_prime_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.host_location_ = HostVector<Data>(rk.relinkey_size_);
            cudaMemcpy(rk.data(), temp_location.data(),
                       rk.relinkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void
    HEKeyGenerator::generate_ckks_relin_key_method_II(Relinkey& rk,
                                                      Secretkey& sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (rk.store_in_gpu_)
        {
            rk.device_location_ = DeviceVector<Data>(rk.relinkey_size_);
            relinkey_kernel_externel_product<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256>>>(
                rk.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), Sk_pair_leveled_->operator[](0).data(),
                n_power, Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(rk.relinkey_size_);
            relinkey_kernel_externel_product<<<dim3((n >> 8), Q_prime_size_, 1),
                                               256>>>(
                temp_location.data(), sk.data(), e_a.data(), modulus_->data(),
                factor_->data(), Sk_pair_leveled_->operator[](0).data(),
                n_power, Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            rk.host_location_ = HostVector<Data>(rk.relinkey_size_);
            cudaMemcpy(rk.data(), temp_location.data(),
                       rk.relinkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void
    HEKeyGenerator::generate_ckks_relin_key_method_III(Relinkey& rk,
                                                       Secretkey& sk)
    {
        int max_depth = Q_size_ - 1;
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        DeviceVector<Data> temp_calculation(2 * Q_prime_size_ *
                                            d_leveled_->operator[](0) * n);

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

            error_kernel_leveled<<<dim3((n >> 8), 2, 1), 256>>>(
                e_a.data(), modulus_->data(), n_power, depth_mod_size,
                prime_location_leveled_->data() + location, seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Modulus_Ordered_Inplace(
                e_a.data(), ntt_table_->data(), modulus_->data(), cfg_ntt,
                2 * depth_mod_size, depth_mod_size,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            relinkey_kernel_externel_product_leveled<<<
                dim3((n >> 8), depth_mod_size, 1), 256>>>(
                temp_calculation.data(), sk.data(), e_a.data(),
                modulus_->data(), factor_->data(),
                Sk_pair_leveled_->operator[](i).data(), n_power, depth_mod_size,
                d, Q_size_, P_size_,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                              .ntt_type = INVERSE,
                                              .reduction_poly =
                                                  ReductionPolynomial::X_N_plus,
                                              .zero_padding = false,
                                              .mod_inverse = n_inverse_->data(),
                                              .stream = 0};

            GPU_NTT_Modulus_Ordered_Inplace(
                temp_calculation.data(), intt_table_->data(), modulus_->data(),
                cfg_intt, 2 * depth_mod_size * d, depth_mod_size,
                prime_location_leveled_->data() + location);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            //////////////
            if (rk.store_in_gpu_)
            {
                rk.device_location_leveled_.push_back(
                    DeviceVector<Data>(rk.relinkey_size_leveled_[i]));
                relinkey_DtoB_kernel_leveled2<<<
                    dim3((n >> 8), d_tilda, (d << 1)), 256>>>(
                    temp_calculation.data(), rk.data(i), modulus_->data(),
                    B_prime_leveled_->data(),
                    base_change_matrix_D_to_B_leveled_->operator[](i).data(),
                    Mi_inv_D_to_B_leveled_->operator[](i).data(),
                    prod_D_to_B_leveled_->operator[](i).data(),
                    I_j_leveled_->operator[](i).data(),
                    I_location_leveled_->operator[](i).data(), n_power,
                    depth_mod_size, d_tilda, d, r_prime,
                    prime_location_leveled_->data() + location);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                GPU_NTT_Inplace(rk.data(i), B_prime_ntt_tables_leveled_->data(),
                                B_prime_leveled_->data(), cfg_ntt,
                                2 * d_tilda * d * r_prime, r_prime);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(rk.relinkey_size_leveled_[i]);
                relinkey_DtoB_kernel_leveled2<<<
                    dim3((n >> 8), d_tilda, (d << 1)), 256>>>(
                    temp_calculation.data(), temp_location.data(),
                    modulus_->data(), B_prime_leveled_->data(),
                    base_change_matrix_D_to_B_leveled_->operator[](i).data(),
                    Mi_inv_D_to_B_leveled_->operator[](i).data(),
                    prod_D_to_B_leveled_->operator[](i).data(),
                    I_j_leveled_->operator[](i).data(),
                    I_location_leveled_->operator[](i).data(), n_power,
                    depth_mod_size, d_tilda, d, r_prime,
                    prime_location_leveled_->data() + location);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                GPU_NTT_Inplace(temp_location.data(),
                                B_prime_ntt_tables_leveled_->data(),
                                B_prime_leveled_->data(), cfg_ntt,
                                2 * d_tilda * d * r_prime, r_prime);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                rk.host_location_leveled_.push_back(
                    HostVector<Data>(rk.relinkey_size_leveled_[i]));

                cudaMemcpy(rk.data(i), temp_location.data(),
                           rk.relinkey_size_leveled_[i] * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    __host__ void HEKeyGenerator::generate_galois_key_method_I(Galoiskey& gk,
                                                               Secretkey& sk)
    {
        DeviceVector<Data> error_a(2 * Q_prime_size_ * n);
        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois.second] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_I_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.data(galois.second), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_I_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois.second] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois.second].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_I_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                            256>>>(
                    gk.c_data(), sk.data(), error_a.data(), modulus_->data(),
                    factor_->data(), gk.galois_elt_zero, n_power,
                    Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_I_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                            256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois_] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_I_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.device_location_[galois_].data(), sk.data(),
                        error_a.data(), modulus_->data(), factor_->data(),
                        galois_, n_power, Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_I_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois_, n_power,
                        Q_prime_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois_] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois_].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_I_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                            256>>>(
                    gk.zero_device_location_.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_I_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                            256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    n_power, Q_prime_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    __host__ void
    HEKeyGenerator::generate_bfv_galois_key_method_II(Galoiskey& gk,
                                                      Secretkey& sk)
    {
        DeviceVector<Data> error_a(2 * Q_prime_size_ * n);
        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois.second] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.data(galois.second), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois.second] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois.second].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    gk.c_data(), sk.data(), error_a.data(), modulus_->data(),
                    factor_->data(), gk.galois_elt_zero, Sk_pair_->data(),
                    n_power, Q_prime_size_, d_, Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                    P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois_] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.device_location_[galois_].data(), sk.data(),
                        error_a.data(), modulus_->data(), factor_->data(),
                        galois_, Sk_pair_->data(), n_power, Q_prime_size_, d_,
                        Q_size_, P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois_,
                        Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois_] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois_].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    gk.zero_device_location_.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                    P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_->data(), n_power, Q_prime_size_, d_, Q_size_,
                    P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    __host__ void
    HEKeyGenerator::generate_ckks_galois_key_method_II(Galoiskey& gk,
                                                       Secretkey& sk)
    {
        DeviceVector<Data> error_a(2 * Q_prime_size_ * n);
        if (!gk.customized)
        {
            // Positive Row Shift
            for (auto& galois : gk.galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois.second] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.data(galois.second), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        Sk_pair_leveled_->operator[](0).data(), n_power,
                        Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois.second,
                        Sk_pair_leveled_->operator[](0).data(), n_power,
                        Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois.second] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois.second].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    gk.c_data(), sk.data(), error_a.data(), modulus_->data(),
                    factor_->data(), gk.galois_elt_zero,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
        else
        {
            for (auto& galois_ : gk.custom_galois_elt)
            {
                error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                    error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                    seed_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                ntt_rns_configuration cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = FORWARD,
                    .reduction_poly = ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = 0};

                GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                                Q_prime_size_);

                if (gk.store_in_gpu_)
                {
                    gk.device_location_[galois_] =
                        DeviceVector<Data>(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        gk.device_location_[galois_].data(), sk.data(),
                        error_a.data(), modulus_->data(), factor_->data(),
                        galois_, Sk_pair_leveled_->operator[](0).data(),
                        n_power, Q_prime_size_, d_leveled_->operator[](0),
                        Q_size_, P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
                else
                {
                    DeviceVector<Data> temp_location(gk.galoiskey_size_);
                    galoiskey_method_II_kernel<<<
                        dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                        temp_location.data(), sk.data(), error_a.data(),
                        modulus_->data(), factor_->data(), galois_,
                        Sk_pair_leveled_->operator[](0).data(), n_power,
                        Q_prime_size_, d_leveled_->operator[](0), Q_size_,
                        P_size_);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    gk.host_location_[galois_] =
                        HostVector<Data>(gk.galoiskey_size_);
                    cudaMemcpy(gk.host_location_[galois_].data(),
                               temp_location.data(),
                               gk.galoiskey_size_ * sizeof(Data),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                }
            }

            // Columns Rotate
            error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
                error_a.data(), modulus_->data(), n_power, Q_prime_size_,
                seed_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            GPU_NTT_Inplace(error_a.data(), ntt_table_->data(),
                            modulus_->data(), cfg_ntt, 2 * Q_prime_size_,
                            Q_prime_size_);

            if (gk.store_in_gpu_)
            {
                gk.zero_device_location_ =
                    DeviceVector<Data>(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    gk.zero_device_location_.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data> temp_location(gk.galoiskey_size_);
                galoiskey_method_II_kernel<<<dim3((n >> 8), Q_prime_size_, 1),
                                             256>>>(
                    temp_location.data(), sk.data(), error_a.data(),
                    modulus_->data(), factor_->data(), gk.galois_elt_zero,
                    Sk_pair_leveled_->operator[](0).data(), n_power,
                    Q_prime_size_, d_leveled_->operator[](0), Q_size_, P_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gk.zero_host_location_ = HostVector<Data>(gk.galoiskey_size_);
                cudaMemcpy(gk.zero_host_location_.data(), temp_location.data(),
                           gk.galoiskey_size_ * sizeof(Data),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////

    __host__ void HEKeyGenerator::generate_switch_key_method_I(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk)
    {
        DeviceVector<Data> error_a(2 * Q_prime_size_ * n);

        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(error_a.data(),
                                                    modulus_->data(), n_power,
                                                    Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(error_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            swk.device_location_ = DeviceVector<Data>(swk.switchkey_size_);
            switchkey_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                swk.device_location_.data(), new_sk.data(), old_sk.data(),
                error_a.data(), modulus_->data(), factor_->data(), n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(swk.switchkey_size_);
            switchkey_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp_location.data(), new_sk.data(), old_sk.data(),
                error_a.data(), modulus_->data(), factor_->data(), n_power,
                Q_prime_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.host_location_ = HostVector<Data>(swk.switchkey_size_);
            cudaMemcpy(swk.host_location_.data(), temp_location.data(),
                       swk.switchkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void HEKeyGenerator::generate_bfv_switch_key_method_II(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            swk.device_location_ = DeviceVector<Data>(swk.switchkey_size_);
            switchkey_kernel_method_II<<<dim3((n >> 8), Q_prime_size_, 1),
                                         256>>>(
                swk.device_location_.data(), new_sk.data(), old_sk.data(),
                e_a.data(), modulus_->data(), factor_->data(), Sk_pair_->data(),
                n_power, Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(swk.switchkey_size_);
            switchkey_kernel_method_II<<<dim3((n >> 8), Q_prime_size_, 1),
                                         256>>>(
                temp_location.data(), new_sk.data(), old_sk.data(), e_a.data(),
                modulus_->data(), factor_->data(), Sk_pair_->data(), n_power,
                Q_prime_size_, d_, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.host_location_ = HostVector<Data>(swk.switchkey_size_);
            cudaMemcpy(swk.host_location_.data(), temp_location.data(),
                       swk.switchkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

    __host__ void HEKeyGenerator::generate_ckks_switch_key_method_II(
        Switchkey& swk, Secretkey& new_sk, Secretkey& old_sk)
    {
        DeviceVector<Data> e_a(2 * Q_prime_size_ * n);
        error_kernel<<<dim3((n >> 8), 2, 1), 256>>>(
            e_a.data(), modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(e_a.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (swk.store_in_gpu_)
        {
            swk.device_location_ = DeviceVector<Data>(swk.switchkey_size_);
            switchkey_kernel_method_II<<<dim3((n >> 8), Q_prime_size_, 1),
                                         256>>>(
                swk.device_location_.data(), new_sk.data(), old_sk.data(),
                e_a.data(), modulus_->data(), factor_->data(),
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> temp_location(swk.switchkey_size_);
            switchkey_kernel_method_II<<<dim3((n >> 8), Q_prime_size_, 1),
                                         256>>>(
                temp_location.data(), new_sk.data(), old_sk.data(), e_a.data(),
                modulus_->data(), factor_->data(),
                Sk_pair_leveled_->operator[](0).data(), n_power, Q_prime_size_,
                d_leveled_->operator[](0), Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            swk.host_location_ = HostVector<Data>(swk.switchkey_size_);
            cudaMemcpy(swk.host_location_.data(), temp_location.data(),
                       swk.switchkey_size_ * sizeof(Data),
                       cudaMemcpyDeviceToHost);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
    }

} // namespace heongpu