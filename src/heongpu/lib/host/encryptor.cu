// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encryptor.cuh"

namespace heongpu
{

    __host__ HEEncryptor::HEEncryptor(Parameters& context,
                                      Publickey& public_key)
    {
        scheme = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();

        public_key_ = public_key.data();

        n = context.n;
        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        // modulus_ = context.modulus_.data();
        modulus_ = context.modulus_;

        last_q_modinv_ = context.last_q_modinv_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        half_ = context.half_p_;

        half_mod_ = context.half_mod_;

        n = context.n;
        n_power = context.n_power;

        if (scheme == scheme_type::bfv)
        {
            plain_modulus_ = context.plain_modulus_;

            Q_mod_t_ = context.Q_mod_t_;

            upper_threshold_ = context.upper_threshold_;

            coeeff_div_plainmod_ = context.coeeff_div_plainmod_;
        }
        else
        {
        }

        temp_data = DeviceVector<Data>((3 * n * Q_prime_size_) +
                                       (2 * n * Q_prime_size_));
        temp1_enc = temp_data.data();
        temp2_enc = temp1_enc + (3 * n * Q_prime_size_);
    }

    __host__ void HEEncryptor::encrypt_bfv(Ciphertext& ciphertext,
                                           Plaintext& plaintext)
    {
        enc_error_kernel<<<dim3((n >> 8), 3, 1), 256>>>(
            temp1_enc, modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_enc, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256>>>(
            public_key_, temp1_enc, temp2_enc, modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_enc, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        EncDivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_enc, temp1_enc + (Q_prime_size_ << n_power), plaintext.data(),
            ciphertext.data(), modulus_->data(), half_->data(),
            half_mod_->data(), last_q_modinv_->data(), plain_modulus_, Q_mod_t_,
            upper_threshold_, coeeff_div_plainmod_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncryptor::encrypt_bfv(Ciphertext& ciphertext,
                                           Plaintext& plaintext,
                                           HEStream& stream)
    {
        enc_error_kernel<<<dim3((n >> 8), 3, 1), 256, 0, stream.stream>>>(
            stream.temp1_enc, modulus_->data(), n_power, Q_prime_size_,
            seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_enc, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                      stream.stream>>>(public_key_, stream.temp1_enc,
                                       stream.temp2_enc, modulus_->data(),
                                       n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_enc, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        EncDivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                  stream.stream>>>(
            stream.temp2_enc, stream.temp1_enc + (Q_prime_size_ << n_power),
            plaintext.data(), ciphertext.data(), modulus_->data(),
            half_->data(), half_mod_->data(), last_q_modinv_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEEncryptor::encrypt_ckks(Ciphertext& ciphertext,
                                            Plaintext& plaintext)
    {
        enc_error_kernel<<<dim3((n >> 8), 3, 1), 256>>>(
            temp1_enc, modulus_->data(), n_power, Q_prime_size_, seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_enc, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256>>>(
            public_key_, temp1_enc, temp2_enc, modulus_->data(), n_power,
            Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_enc, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        EncDivideRoundLastqNewP_ckks<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_enc, temp1_enc + (Q_prime_size_ << n_power),
            ciphertext.data(), modulus_->data(), half_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power, Q_prime_size_,
            Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(ciphertext.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_size_, Q_size_);

        cipher_message_add<<<dim3((n >> 8), Q_size_, 1), 256>>>(
            ciphertext.data(), plaintext.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ciphertext.scale_ = plaintext.scale_;
    }

    __host__ void HEEncryptor::encrypt_ckks(Ciphertext& ciphertext,
                                            Plaintext& plaintext,
                                            HEStream& stream)
    {
        enc_error_kernel<<<dim3((n >> 8), 3, 1), 256, 0, stream.stream>>>(
            stream.temp1_enc, modulus_->data(), n_power, Q_prime_size_,
            seed_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_enc, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_prime_size_, Q_prime_size_);

        pk_u_kernel<<<dim3((n >> 8), Q_prime_size_, 2), 256, 0,
                      stream.stream>>>(public_key_, stream.temp1_enc,
                                       stream.temp2_enc, modulus_->data(),
                                       n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_enc, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        EncDivideRoundLastqNewP_ckks<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream.stream>>>(
            stream.temp2_enc, stream.temp1_enc + (Q_prime_size_ << n_power),
            ciphertext.data(), modulus_->data(), half_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power, Q_prime_size_,
            Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(ciphertext.data(), ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * Q_size_, Q_size_);

        cipher_message_add<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                             stream.stream>>>(
            ciphertext.data(), plaintext.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ciphertext.scale_ = plaintext.scale_;
    }

} // namespace heongpu