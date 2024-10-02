// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "decryptor.cuh"

namespace heongpu
{
    __host__ HEDecryptor::HEDecryptor(Parameters& context,
                                      Secretkey& secret_key)
    {
        scheme = context.scheme_;

        secret_key_ = secret_key.data();

        n = context.n;
        n_power = context.n_power;

        decomp_mod_count_ = context.Q_size;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        if (scheme == scheme_type::bfv)
        {
            plain_modulus_ = context.plain_modulus_;

            gamma_ = context.gamma_;

            Qi_t_ = context.Qi_t_;

            Qi_gamma_ = context.Qi_gamma_;

            Qi_inverse_ = context.Qi_inverse_;

            mulq_inv_t_ = context.mulq_inv_t_;

            mulq_inv_gamma_ = context.mulq_inv_gamma_;

            inv_gamma_ = context.inv_gamma_;

            // Noise budget calculation

            Mi_ = context.Mi_;
            Mi_inv_ = context.Mi_inv_;
            upper_half_threshold_ = context.upper_half_threshold_;
            decryption_modulus_ = context.decryption_modulus_;
            total_bit_count_ = context.total_bit_count_;

            temp_memory_ = DeviceVector<Data>(n * decomp_mod_count_);
            temp_memory2_ = DeviceVector<Data>(2 * n * decomp_mod_count_);

            // max_norm_memory_ =
            //     (Data*) malloc(n * decomp_mod_count_ * sizeof(Data));
            max_norm_memory_.resize(n * decomp_mod_count_);
        }
    }

    __host__ void HEDecryptor::decrypt_bfv(Plaintext& plaintext,
                                           Ciphertext& ciphertext)
    {
        Data* ct0 = ciphertext.data();
        Data* ct1 = ciphertext.data() + (decomp_mod_count_ << n_power);

        Data* ct0_temp = temp_memory2_.data();
        Data* ct1_temp = temp_memory2_.data() + (decomp_mod_count_ << n_power);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};
        if (!ciphertext.in_ntt_domain_)
        {
            GPU_NTT(ct1, ct1_temp, ntt_table_->data(), modulus_->data(),
                    cfg_ntt, decomp_mod_count_, decomp_mod_count_);

            sk_multiplication<<<dim3((n >> 8), decomp_mod_count_, 1), 256>>>(
                ct1_temp, secret_key_, ct1_temp, modulus_->data(), n_power,
                decomp_mod_count_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((n >> 8), decomp_mod_count_, 1), 256>>>(
                ct1, secret_key_, ct1_temp, modulus_->data(), n_power,
                decomp_mod_count_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        if (ciphertext.in_ntt_domain_)
        {
            // TODO: merge these NTTs
            GPU_NTT(ct0, ct0_temp, intt_table_->data(), modulus_->data(),
                    cfg_intt, decomp_mod_count_, decomp_mod_count_);

            GPU_NTT_Inplace(ct1_temp, intt_table_->data(), modulus_->data(),
                            cfg_intt, decomp_mod_count_, decomp_mod_count_);

            ct0 = ct0_temp;
        }
        else
        {
            GPU_NTT_Inplace(ct1_temp, intt_table_->data(), modulus_->data(),
                            cfg_intt, decomp_mod_count_, decomp_mod_count_);
        }

        decryption_kernel<<<dim3((n >> 8), 1, 1), 256>>>(
            ct0, ct1_temp, plaintext.data(), modulus_->data(), plain_modulus_,
            gamma_, Qi_t_->data(), Qi_gamma_->data(), Qi_inverse_->data(),
            mulq_inv_t_, mulq_inv_gamma_, inv_gamma_, n_power,
            decomp_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEDecryptor::decryptx3_bfv(Plaintext& plaintext,
                                             Ciphertext& ciphertext)
    {
        Data* ct0 = ciphertext.data();
        Data* ct1 = ciphertext.data() + (decomp_mod_count_ << n_power);
        Data* ct2 = ciphertext.data() + (decomp_mod_count_ << (n_power + 1));

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(ct1, ntt_table_->data(), modulus_->data(), cfg_ntt,
                        2 * decomp_mod_count_, decomp_mod_count_);

        sk_multiplicationx3<<<dim3((n >> 8), decomp_mod_count_, 1), 256>>>(
            ct1, secret_key_, modulus_->data(), n_power, decomp_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(ct1, intt_table_->data(), modulus_->data(), cfg_intt,
                        2 * decomp_mod_count_, decomp_mod_count_);

        decryption_kernelx3<<<dim3((n >> 8), 1, 1), 256>>>(
            ct0, ct1, ct2, plaintext.data(), modulus_->data(), plain_modulus_,
            gamma_, Qi_t_->data(), Qi_gamma_->data(), Qi_inverse_->data(),
            mulq_inv_t_, mulq_inv_gamma_, inv_gamma_, n_power,
            decomp_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ int HEDecryptor::noise_budget_calculation(Ciphertext& ciphertext)
    {
        Data* ct0 = ciphertext.data();
        Data* ct1 = ciphertext.data() + (decomp_mod_count_ << n_power);

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT(ct1, temp_memory_.data(), ntt_table_->data(), modulus_->data(),
                cfg_ntt, decomp_mod_count_, decomp_mod_count_);

        sk_multiplication<<<dim3((n >> 8), decomp_mod_count_, 1), 256>>>(
            temp_memory_.data(), secret_key_, temp_memory_.data(),
            modulus_->data(), n_power, decomp_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp_memory_.data(), intt_table_->data(),
                        modulus_->data(), cfg_intt, decomp_mod_count_,
                        decomp_mod_count_);

        coeff_multadd<<<dim3((n >> 8), decomp_mod_count_, 1), 256>>>(
            ct0, temp_memory_.data(), temp_memory_.data(), plain_modulus_,
            modulus_->data(), n_power, decomp_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        compose_kernel<<<dim3((n >> 8), 1, 1), 256>>>(
            temp_memory_.data(), temp_memory_.data(), modulus_->data(),
            Mi_inv_->data(), Mi_->data(), decryption_modulus_->data(),
            decomp_mod_count_, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        find_max_norm_kernel<<<1, 512, sizeof(Data) * 512>>>(
            temp_memory_.data(), temp_memory_.data(),
            upper_half_threshold_->data(), decryption_modulus_->data(),
            decomp_mod_count_,
            n_power); // TODO: merge with above kernel if possible
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        cudaMemcpy(max_norm_memory_.data(), temp_memory_.data(),
                   decomp_mod_count_ * sizeof(Data), cudaMemcpyDeviceToHost);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        return total_bit_count_ -
               calculate_big_integer_bit_count(max_norm_memory_.data(),
                                               decomp_mod_count_) -
               1;
    }

    __host__ void HEDecryptor::decrypt_ckks(Plaintext& plaintext,
                                            Ciphertext& ciphertext)
    {
        int current_decomp_count = decomp_mod_count_ - ciphertext.depth_;

        sk_multiplication_ckks<<<dim3((n >> 8), current_decomp_count, 1),
                                 256>>>(ciphertext.data(), plaintext.data(),
                                        secret_key_, modulus_->data(), n_power,
                                        current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.scale_ = ciphertext.scale_;
        plaintext.depth_ = ciphertext.depth_;
    }

} // namespace heongpu