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
        scheme_ = context.scheme_;

        std::random_device rd;
        std::mt19937 gen(rd());
        seed_ = gen();
        offset_ = gen();

        secret_key_ = secret_key.location_;

        n = context.n;
        n_power = context.n_power;

        Q_size_ = context.Q_size;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        if (scheme_ == scheme_type::bfv)
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
        }
    }

    __host__ void HEDecryptor::decrypt_bfv(Plaintext& plaintext,
                                           Ciphertext& ciphertext,
                                           const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(n, stream);

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);

        DeviceVector<Data64> temp_memory(2 * n * Q_size_, stream);
        Data64* ct0_temp = temp_memory.data();
        Data64* ct1_temp = temp_memory.data() + (Q_size_ << n_power);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};
        if (!ciphertext.in_ntt_domain_)
        {
            gpuntt::GPU_NTT(ct1, ct1_temp, ntt_table_->data(), modulus_->data(),
                            cfg_ntt, Q_size_, Q_size_);

            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1_temp, secret_key_.data(), ct1_temp, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                ct1, secret_key_.data(), ct1_temp, modulus_->data(), n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        if (ciphertext.in_ntt_domain_)
        {
            // TODO: merge these NTTs
            gpuntt::GPU_NTT(ct0, ct0_temp, intt_table_->data(),
                            modulus_->data(), cfg_intt, Q_size_, Q_size_);

            gpuntt::GPU_NTT_Inplace(ct1_temp, intt_table_->data(),
                                    modulus_->data(), cfg_intt, Q_size_,
                                    Q_size_);

            ct0 = ct0_temp;
        }
        else
        {
            gpuntt::GPU_NTT_Inplace(ct1_temp, intt_table_->data(),
                                    modulus_->data(), cfg_intt, Q_size_,
                                    Q_size_);
        }

        decryption_kernel<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1_temp, output_memory.data(), modulus_->data(),
            plain_modulus_, gamma_, Qi_t_->data(), Qi_gamma_->data(),
            Qi_inverse_->data(), mulq_inv_t_, mulq_inv_gamma_, inv_gamma_,
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void HEDecryptor::decryptx3_bfv(Plaintext& plaintext,
                                             Ciphertext& ciphertext,
                                             const cudaStream_t stream)
    {
        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);
        Data64* ct2 = ciphertext.data() + (Q_size_ << (n_power + 1));

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(ct1, ntt_table_->data(), modulus_->data(),
                                cfg_ntt, 2 * Q_size_, Q_size_);

        sk_multiplicationx3<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            ct1, secret_key_.data(), modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(ct1, intt_table_->data(), modulus_->data(),
                                cfg_intt, 2 * Q_size_, Q_size_);

        decryption_kernelx3<<<dim3((n >> 8), 1, 1), 256, 0, stream>>>(
            ct0, ct1, ct2, plaintext.data(), modulus_->data(), plain_modulus_,
            gamma_, Qi_t_->data(), Qi_gamma_->data(), Qi_inverse_->data(),
            mulq_inv_t_, mulq_inv_gamma_, inv_gamma_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ int
    HEDecryptor::noise_budget_calculation(Ciphertext& ciphertext,
                                          const ExecutionOptions& options)
    {
        HostVector<Data64> max_norm_memory(n * Q_size_);

        input_storage_manager(
            ciphertext,
            [&](Ciphertext& ciphertext_)
            {
                Data64* ct0 = ciphertext.data();
                Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);

                gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::FORWARD,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .stream = options.stream_};

                DeviceVector<Data64> temp_memory(n * Q_size_, options.stream_);
                gpuntt::GPU_NTT(ct1, temp_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, Q_size_, Q_size_);

                sk_multiplication<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                    options.stream_>>>(
                    temp_memory.data(), secret_key_.data(), temp_memory.data(),
                    modulus_->data(), n_power, Q_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                    .n_power = n_power,
                    .ntt_type = gpuntt::INVERSE,
                    .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                    .zero_padding = false,
                    .mod_inverse = n_inverse_->data(),
                    .stream = options.stream_};

                gpuntt::GPU_NTT_Inplace(temp_memory.data(), intt_table_->data(),
                                        modulus_->data(), cfg_intt, Q_size_,
                                        Q_size_);

                coeff_multadd<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                options.stream_>>>(
                    ct0, temp_memory.data(), temp_memory.data(), plain_modulus_,
                    modulus_->data(), n_power, Q_size_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                compose_kernel<<<dim3((n >> 8), 1, 1), 256, 0,
                                 options.stream_>>>(
                    temp_memory.data(), temp_memory.data(), modulus_->data(),
                    Mi_inv_->data(), Mi_->data(), decryption_modulus_->data(),
                    Q_size_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                find_max_norm_kernel<<<1, 512, sizeof(Data64) * 512,
                                       options.stream_>>>(
                    temp_memory.data(), temp_memory.data(),
                    upper_half_threshold_->data(), decryption_modulus_->data(),
                    Q_size_,
                    n_power); // TODO: merge with above kernel if possible
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                cudaMemcpyAsync(max_norm_memory.data(), temp_memory.data(),
                                Q_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, options.stream_);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            },
            options, false);

        return total_bit_count_ -
               calculate_big_integer_bit_count(max_norm_memory.data(),
                                               Q_size_) -
               1;
    }

    __host__ void HEDecryptor::decrypt_ckks(Plaintext& plaintext,
                                            Ciphertext& ciphertext,
                                            const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - ciphertext.depth_;
        DeviceVector<Data64> output_memory(n * current_decomp_count, stream);

        sk_multiplication_ckks<<<dim3((n >> 8), current_decomp_count, 1), 256,
                                 0, stream>>>(
            ciphertext.data(), output_memory.data(), secret_key_.data(),
            modulus_->data(), n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        plaintext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEDecryptor::partial_decrypt_bfv(Ciphertext& ciphertext, Secretkey& sk,
                                     Ciphertext& partial_ciphertext,
                                     const cudaStream_t stream)
    {
        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (Q_size_ << n_power);

        DeviceVector<Data64> temp_memory(n * Q_size_, stream);
        Data64* ct1_temp = temp_memory.data();

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
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
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        DeviceVector<Data64> output_memory((2 * n * Q_size_), stream);

        gpuntt::GPU_NTT(ct1_temp, output_memory.data() + (Q_size_ * n),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        Q_size_, Q_size_);

        DeviceVector<Data64> error_poly(Q_size_ * n, stream);
        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                           256, 0, stream>>>(
            error_poly.data(), modulus_->data(), n_power, Q_size_, seed_,
            offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

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

    __host__ void
    HEDecryptor::partial_decrypt_ckks(Ciphertext& ciphertext, Secretkey& sk,
                                      Ciphertext& partial_ciphertext,
                                      const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - ciphertext.depth_;

        Data64* ct0 = ciphertext.data();
        Data64* ct1 = ciphertext.data() + (current_decomp_count << n_power);

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        sk_multiplication<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                            stream>>>(ct1, sk.data(),
                                      output_memory.data() +
                                          (current_decomp_count << n_power),
                                      modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DeviceVector<Data64> error_poly(current_decomp_count * n, stream);
        modular_gaussian_random_number_generation_kernel<<<dim3((n >> 8), 1, 1),
                                                           256, 0, stream>>>(
            error_poly.data(), modulus_->data(), n_power, current_decomp_count,
            seed_, offset_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        offset_++;

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(error_poly.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, current_decomp_count,
                                current_decomp_count);

        // TODO: Optimize it!
        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            output_memory.data() + (current_decomp_count * n),
            error_poly.data(),
            output_memory.data() + (current_decomp_count * n), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<dim3((n >> 8), current_decomp_count, 1),
                                       256, 0, stream>>>(
            ct0, output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        partial_ciphertext.memory_set(std::move(output_memory));
    }

    __host__ void
    HEDecryptor::decrypt_fusion_bfv(std::vector<Ciphertext>& ciphertexts,
                                    Plaintext& plaintext,
                                    const cudaStream_t stream)
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
    HEDecryptor::decrypt_fusion_ckks(std::vector<Ciphertext>& ciphertexts,
                                     Plaintext& plaintext,
                                     const cudaStream_t stream)
    {
        int cipher_count = ciphertexts.size();
        int current_detph = ciphertexts[0].depth_;
        int current_decomp_count = Q_size_ - current_detph;

        DeviceVector<Data64> output_memory(n * current_decomp_count, stream);

        Data64* ct0 = ciphertexts[0].data();
        Data64* ct1 = ciphertexts[0].data() + (current_decomp_count << n_power);
        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            ct0, ct1, output_memory.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < cipher_count; i++)
        {
            Data64* ct1_i =
                ciphertexts[i].data() + (current_decomp_count << n_power);

            addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                       stream>>>(ct1_i, output_memory.data(),
                                 output_memory.data(), modulus_->data(),
                                 n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        plaintext.memory_set(std::move(output_memory));
    }

} // namespace heongpu