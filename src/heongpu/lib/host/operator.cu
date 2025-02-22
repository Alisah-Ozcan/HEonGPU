// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "operator.cuh"

namespace heongpu
{

    __host__ HEOperator::HEOperator(Parameters& context, HEEncoder& encoder)
    {
        scheme_ = context.scheme_;

        n = context.n;

        n_power = context.n_power;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;
        P_size_ = context.P_size;

        bsk_mod_count_ = context.bsk_modulus;

        modulus_ = context.modulus_;

        ntt_table_ = context.ntt_table_;

        intt_table_ = context.intt_table_;

        n_inverse_ = context.n_inverse_;

        last_q_modinv_ = context.last_q_modinv_;

        base_Bsk_ = context.base_Bsk_;

        bsk_ntt_tables_ = context.bsk_ntt_tables_;

        bsk_intt_tables_ = context.bsk_intt_tables_;

        bsk_n_inverse_ = context.bsk_n_inverse_;

        m_tilde_ = context.m_tilde_;

        base_change_matrix_Bsk_ = context.base_change_matrix_Bsk_;

        inv_punctured_prod_mod_base_array_ =
            context.inv_punctured_prod_mod_base_array_;

        base_change_matrix_m_tilde_ = context.base_change_matrix_m_tilde_;

        inv_prod_q_mod_m_tilde_ = context.inv_prod_q_mod_m_tilde_;

        inv_m_tilde_mod_Bsk_ = context.inv_m_tilde_mod_Bsk_;

        prod_q_mod_Bsk_ = context.prod_q_mod_Bsk_;

        inv_prod_q_mod_Bsk_ = context.inv_prod_q_mod_Bsk_;

        plain_modulus_ = context.plain_modulus_;

        base_change_matrix_q_ = context.base_change_matrix_q_;

        base_change_matrix_msk_ = context.base_change_matrix_msk_;

        inv_punctured_prod_mod_B_array_ =
            context.inv_punctured_prod_mod_B_array_;

        inv_prod_B_mod_m_sk_ = context.inv_prod_B_mod_m_sk_;

        prod_B_mod_q_ = context.prod_B_mod_q_;

        q_Bsk_merge_modulus_ = context.q_Bsk_merge_modulus_;

        q_Bsk_merge_ntt_tables_ = context.q_Bsk_merge_ntt_tables_;

        q_Bsk_merge_intt_tables_ = context.q_Bsk_merge_intt_tables_;

        q_Bsk_n_inverse_ = context.q_Bsk_n_inverse_;

        half_p_ = context.half_p_;

        half_mod_ = context.half_mod_;

        upper_threshold_ = context.upper_threshold_;

        upper_halfincrement_ = context.upper_halfincrement_;

        Q_mod_t_ = context.Q_mod_t_;

        coeeff_div_plainmod_ = context.coeeff_div_plainmod_;

        //////

        d = context.d;
        d_tilda = context.d_tilda;
        r_prime = context.r_prime;

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

        base_change_matrix_D_to_Q_tilda_ =
            context.base_change_matrix_D_to_Q_tilda_;
        Mi_inv_D_to_Q_tilda_ = context.Mi_inv_D_to_Q_tilda_;
        prod_D_to_Q_tilda_ = context.prod_D_to_Q_tilda_;

        I_j_ = context.I_j_;
        I_location_ = context.I_location_;
        Sk_pair_ = context.Sk_pair_;

        l_leveled_ = context.l_leveled;
        l_tilda_leveled_ = context.l_tilda_leveled;
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

        // Method2
        base_change_matrix_D_to_Qtilda_leveled_ =
            context.base_change_matrix_D_to_Qtilda_leveled;
        Mi_inv_D_to_Qtilda_leveled_ = context.Mi_inv_D_to_Qtilda_leveled;
        prod_D_to_Qtilda_leveled_ = context.prod_D_to_Qtilda_leveled;

        I_j_leveled_ = context.I_j_leveled;
        I_location_leveled_ = context.I_location_leveled;
        Sk_pair_leveled_ = context.Sk_pair_leveled;

        prime_location_leveled_ = context.prime_location_leveled;

        // Leveled Rescale
        rescaled_last_q_modinv_ = context.rescaled_last_q_modinv_;
        rescaled_half_ = context.rescaled_half_;
        rescaled_half_mod_ = context.rescaled_half_mod_;

        prime_vector_ = context.prime_vector;

        std::vector<int> prime_loc;
        std::vector<int> input_loc;

        int counter = Q_size_;
        for (int i = 0; i < Q_size_ - 1; i++)
        {
            for (int j = 0; j < counter; j++)
            {
                prime_loc.push_back(j);
            }
            counter--;
            for (int j = 0; j < P_size_; j++)
            {
                prime_loc.push_back(Q_size_ + j);
            }
        }

        counter = Q_prime_size_;
        for (int i = 0; i < Q_prime_size_ - 1; i++)
        {
            int sum = counter - 1;
            for (int j = 0; j < 2; j++)
            {
                input_loc.push_back(sum);
                sum += counter;
            }
            counter--;
        }

        new_prime_locations_ = DeviceVector<int>(prime_loc);
        new_input_locations_ = DeviceVector<int>(input_loc);
        new_prime_locations = new_prime_locations_.data();
        new_input_locations = new_input_locations_.data();

        // Encode params
        slot_count_ = encoder.slot_count_;
        log_slot_count_ = encoder.log_slot_count_;

        // BFV
        plain_modulus_pointer_ = encoder.plain_modulus_;
        n_plain_inverse_ = encoder.n_plain_inverse_;
        plain_intt_tables_ = encoder.plain_intt_tables_;
        encoding_location_ = encoder.encoding_location_;

        // CKKS
        two_pow_64_ = encoder.two_pow_64;
        reverse_order_ = encoder.reverse_order;
        special_ifft_roots_table_ = encoder.special_ifft_roots_table_;
    }

    __host__ void HEOperator::add(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output,
                                  const ExecutionOptions& options)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count) ||
            input2.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * n * current_decomp_count),
                                    options.stream_);

                                addition<<<dim3((n >> 8), current_decomp_count,
                                                cipher_size),
                                           256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(), modulus_->data(),
                                    n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = scheme_;
                                output_.ring_size_ = n;
                                output_.coeff_modulus_count_ = Q_size_;
                                output_.cipher_size_ = cipher_size;
                                output_.depth_ = input1_.depth_;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.scale_ = input1_.scale_;
                                output_.rescale_required_ =
                                    (input1_.rescale_required_ ||
                                     input2_.rescale_required_);
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator::sub(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output,
                                  const ExecutionOptions& options)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        if (input1.relinearization_required_ !=
            input2.relinearization_required_)
        {
            throw std::invalid_argument("Ciphertexts can not be added because "
                                        "ciphertext sizes have to be equal!");
        }

        if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
        {
            throw std::invalid_argument(
                "Both Ciphertexts should be in same domain");
        }

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count) ||
            input2.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext& input1_)
            {
                input_storage_manager(
                    input2,
                    [&](Ciphertext& input2_)
                    {
                        output_storage_manager(
                            output,
                            [&](Ciphertext& output_)
                            {
                                DeviceVector<Data64> output_memory(
                                    (cipher_size * n * current_decomp_count),
                                    options.stream_);

                                substraction<<<dim3((n >> 8),
                                                    current_decomp_count,
                                                    cipher_size),
                                               256, 0, options.stream_>>>(
                                    input1_.data(), input2_.data(),
                                    output_memory.data(), modulus_->data(),
                                    n_power);
                                HEONGPU_CUDA_CHECK(cudaGetLastError());

                                output_.scheme_ = scheme_;
                                output_.ring_size_ = n;
                                output_.coeff_modulus_count_ = Q_size_;
                                output_.cipher_size_ = cipher_size;
                                output_.depth_ = input1_.depth_;
                                output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                                output_.scale_ = input1_.scale_;
                                output_.rescale_required_ =
                                    (input1_.rescale_required_ ||
                                     input2_.rescale_required_);
                                output_.relinearization_required_ =
                                    input1_.relinearization_required_;

                                output_.memory_set(std::move(output_memory));
                            },
                            options);
                    },
                    options, (&input2 == &output));
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator::negate(Ciphertext& input1, Ciphertext& output,
                                     const ExecutionOptions& options)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        input_storage_manager(
            input1,
            [&](Ciphertext& input1_)
            {
                output_storage_manager(
                    output,
                    [&](Ciphertext& output_)
                    {
                        DeviceVector<Data64> output_memory(
                            (cipher_size * n * current_decomp_count),
                            options.stream_);

                        negation<<<dim3((n >> 8), current_decomp_count,
                                        cipher_size),
                                   256, 0, options.stream_>>>(
                            input1_.data(), output_memory.data(),
                            modulus_->data(), n_power);
                        HEONGPU_CUDA_CHECK(cudaGetLastError());

                        output_.scheme_ = scheme_;
                        output_.ring_size_ = n;
                        output_.coeff_modulus_count_ = Q_size_;
                        output_.cipher_size_ = cipher_size;
                        output_.depth_ = input1_.depth_;
                        output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                        output_.scale_ = input1_.scale_;
                        output_.rescale_required_ = input1_.rescale_required_;
                        output_.relinearization_required_ =
                            input1_.relinearization_required_;

                        output_.memory_set(std::move(output_memory));
                    },
                    options);
            },
            options, (&input1 == &output));
    }

    __host__ void HEOperator::add_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output,
                                            const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * n * current_decomp_count), stream);

        addition_plain_bfv_poly<<<dim3((n >> 8), current_decomp_count,
                                       cipher_size),
                                  256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::add_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2,
                                                    const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::add_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output,
                                             const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * n * current_decomp_count), stream);

        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count,
                                        cipher_size),
                                   256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::add_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2,
                                                     const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output,
                                            const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * n * current_decomp_count), stream);

        substraction_plain_bfv_poly<<<dim3((n >> 8), current_decomp_count,
                                           cipher_size),
                                      256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::sub_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2,
                                                    const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output,
                                             const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        DeviceVector<Data64> output_memory(
            (cipher_size * n * current_decomp_count), stream);

        substraction_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count,
                                            cipher_size),
                                       256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::sub_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2,
                                                     const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.memory_size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                       stream>>>(input1.data(), input2.data(), input1.data(),
                                 modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::multiply_bfv(Ciphertext& input1,
                                           Ciphertext& input2,
                                           Ciphertext& output,
                                           const cudaStream_t stream)
    {
        if ((input1.in_ntt_domain_ != false) ||
            (input2.in_ntt_domain_ != false))
        {
            throw std::invalid_argument("Ciphertexts should be in same domain");
        }

        if (input1.memory_size() < (2 * n * Q_size_) ||
            input2.memory_size() < (2 * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        DeviceVector<Data64> output_memory((3 * n * Q_size_), stream);

        DeviceVector<Data64> temp_mul((4 * n * (bsk_mod_count_ + Q_size_)) +
                                          (3 * n * (bsk_mod_count_ + Q_size_)),
                                      stream);
        Data64* temp1_mul = temp_mul.data();
        Data64* temp2_mul = temp1_mul + (4 * n * (bsk_mod_count_ + Q_size_));

        fast_convertion<<<dim3((n >> 8), 4, 1), 256, 0, stream>>>(
            input1.data(), input2.data(), temp1_mul, modulus_->data(),
            base_Bsk_->data(), m_tilde_, inv_prod_q_mod_m_tilde_,
            inv_m_tilde_mod_Bsk_->data(), prod_q_mod_Bsk_->data(),
            base_change_matrix_Bsk_->data(),
            base_change_matrix_m_tilde_->data(),
            inv_punctured_prod_mod_base_array_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = q_Bsk_n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_mul, q_Bsk_merge_ntt_tables_->data(),
                                q_Bsk_merge_modulus_->data(), cfg_ntt,
                                ((bsk_mod_count_ + Q_size_) * 4),
                                (bsk_mod_count_ + Q_size_));

        cross_multiplication<<<dim3((n >> 8), (bsk_mod_count_ + Q_size_), 1),
                               256, 0, stream>>>(
            temp1_mul, temp1_mul + (((bsk_mod_count_ + Q_size_) * 2) * n),
            temp2_mul, q_Bsk_merge_modulus_->data(), n_power,
            (bsk_mod_count_ + Q_size_));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp2_mul, q_Bsk_merge_intt_tables_->data(),
                                q_Bsk_merge_modulus_->data(), cfg_intt,
                                (3 * (bsk_mod_count_ + Q_size_)),
                                (bsk_mod_count_ + Q_size_));

        fast_floor<<<dim3((n >> 8), 3, 1), 256, 0, stream>>>(
            temp2_mul, output_memory.data(), modulus_->data(),
            base_Bsk_->data(), plain_modulus_,
            inv_punctured_prod_mod_base_array_->data(),
            base_change_matrix_Bsk_->data(), inv_prod_q_mod_Bsk_->data(),
            inv_punctured_prod_mod_B_array_->data(),
            base_change_matrix_q_->data(), base_change_matrix_msk_->data(),
            inv_prod_B_mod_m_sk_, prod_B_mod_q_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::multiply_ckks(Ciphertext& input1,
                                            Ciphertext& input2,
                                            Ciphertext& output,
                                            const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.memory_size() < (2 * n * current_decomp_count) ||
            input2.memory_size() < (2 * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        DeviceVector<Data64> output_memory((3 * n * current_decomp_count),
                                           stream);

        cross_multiplication<<<dim3((n >> 8), (current_decomp_count), 1), 256,
                               0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), n_power, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void HEOperator::multiply_plain_bfv(Ciphertext& input1,
                                                 Plaintext& input2,
                                                 Ciphertext& output,
                                                 const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        if (input1.in_ntt_domain_)
        {
            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
                input1.data(), input2.data(), output_memory.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> temp_plain_mul(n * Q_size_, stream);
            Data64* temp1_plain_mul = temp_plain_mul.data();

            threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
                input2.data(), temp1_plain_mul, modulus_->data(),
                upper_halfincrement_->data(), upper_threshold_, n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
                .n_power = n_power,
                .ntt_type = gpuntt::FORWARD,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .stream = stream};

            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = n_power,
                .ntt_type = gpuntt::INVERSE,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = n_inverse_->data(),
                .stream = stream};

            gpuntt::GPU_NTT_Inplace(temp1_plain_mul, ntt_table_->data(),
                                    modulus_->data(), cfg_ntt, Q_size_,
                                    Q_size_);

            gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                            ntt_table_->data(), modulus_->data(), cfg_ntt,
                            2 * Q_size_, Q_size_);

            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
                output_memory.data(), temp1_plain_mul, output_memory.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT_Inplace(output_memory.data(), intt_table_->data(),
                                    modulus_->data(), cfg_intt, 2 * Q_size_,
                                    Q_size_);
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::multiply_plain_ckks(Ciphertext& input1,
                                                  Plaintext& input2,
                                                  Ciphertext& output,
                                                  const cudaStream_t stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            input1.data(), input2.data(), output_memory.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::relinearize_seal_method_inplace(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        cipher_broadcast_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                    stream>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method_inplace(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin_new((n * d * r_prime) +
                                                (2 * n * d_tilda * r_prime) +
                                                (2 * n * Q_prime_size_),
                                            stream);
        Data64* temp1_relin_new = temp_relin_new.data();
        Data64* temp2_relin_new = temp1_relin_new + (n * d * r_prime);
        Data64* temp3_relin_new = temp2_relin_new + (2 * n * d_tilda * r_prime);

        base_conversion_DtoB_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                            stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin_new,
            modulus_->data(), B_prime_->data(),
            base_change_matrix_D_to_B_->data(), Mi_inv_D_to_B_->data(),
            prod_D_to_B_->data(), I_j_->data(), I_location_->data(), n_power,
            Q_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin_new, B_prime_ntt_tables_->data(),
                                B_prime_->data(), cfg_ntt, d * r_prime,
                                r_prime);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime, d_tilda), 256, 0, stream>>>(
                temp1_relin_new, relin_key.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime, d_tilda), 256, 0, stream>>>(
                temp1_relin_new, key_location.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin_new, B_prime_intt_tables_->data(),
                                B_prime_->data(), cfg_intt,
                                2 * r_prime * d_tilda, r_prime);

        base_conversion_BtoD_relin_kernel<<<dim3((n >> 8), d_tilda, 2), 256, 0,
                                            stream>>>(
            temp2_relin_new, temp3_relin_new, modulus_->data(),
            B_prime_->data(), base_change_matrix_B_to_D_->data(),
            Mi_inv_B_to_D_->data(), prod_B_to_D_->data(), I_j_->data(),
            I_location_->data(), n_power, Q_prime_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        divide_round_lastq_extended_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                             stream>>>(
            temp3_relin_new, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_relin, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_extended_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                             stream>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_seal_method_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(input1.data() +
                                    (current_decomp_count << (n_power + 1)),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                current_decomp_count, current_decomp_count);

        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        cipher_broadcast_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count, 1), 256, 0, stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            temp1_relin, modulus_->data(), first_rns_mod_count,
            current_rns_mod_count, n_power);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = stream};

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            temp2_relin, intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<dim3((n >> 8), 2, 1), 256,
                                                      0, stream>>>(
            temp2_relin, temp1_relin, modulus_->data(), half_p_->data(),
            half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        divide_round_lastq_leveled_stage_two_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp1_relin, temp2_relin, input1.data(), input1.data(),
            modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        DeviceVector<Data64> temp_relin_new(
            (n * d_leveled_->operator[](0) * r_prime_leveled_) +
                (2 * n * d_tilda_leveled_->operator[](0) * r_prime_leveled_) +
                (2 * n * Q_prime_size_),
            stream);
        Data64* temp1_relin_new = temp_relin_new.data();
        Data64* temp2_relin_new =
            temp1_relin_new +
            (n * d_leveled_->operator[](0) * r_prime_leveled_);
        Data64* temp3_relin_new =
            temp2_relin_new +
            (2 * n * d_tilda_leveled_->operator[](0) * r_prime_leveled_);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            input1.data() + (current_decomp_count << (n_power + 1)),
            intt_table_->data(), modulus_->data(), cfg_intt,
            current_decomp_count, current_decomp_count,
            prime_location_leveled_->data() + location);

        base_conversion_DtoB_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            temp1_relin_new, modulus_->data(), B_prime_leveled_->data(),
            base_change_matrix_D_to_B_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_D_to_B_leveled_->operator[](input1.depth_).data(),
            prod_D_to_B_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            d_tilda_leveled_->operator[](input1.depth_),
            d_leveled_->operator[](input1.depth_), r_prime_leveled_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            temp1_relin_new, B_prime_ntt_tables_leveled_->data(),
            B_prime_leveled_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * r_prime_leveled_,
            r_prime_leveled_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime_leveled_,
                     d_tilda_leveled_->operator[](input1.depth_)),
                256, 0, stream>>>(
                temp1_relin_new, relin_key.data(input1.depth_), temp2_relin_new,
                B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                relin_key.host_location_leveled_[input1.depth_], stream);
            multiply_accumulate_extended_kernel<<<
                dim3((n >> 8), r_prime_leveled_,
                     d_tilda_leveled_->operator[](input1.depth_)),
                256, 0, stream>>>(
                temp1_relin_new, key_location.data(), temp2_relin_new,
                B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_leveled_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(
            temp2_relin_new, B_prime_intt_tables_leveled_->data(),
            B_prime_leveled_->data(), cfg_intt2,
            2 * r_prime_leveled_ * d_tilda_leveled_->operator[](input1.depth_),
            r_prime_leveled_);

        base_conversion_BtoD_relin_leveled_kernel<<<
            dim3((n >> 8), d_tilda_leveled_->operator[](input1.depth_), 2), 256,
            0, stream>>>(
            temp2_relin_new, temp3_relin_new, modulus_->data(),
            B_prime_leveled_->data(),
            base_change_matrix_B_to_D_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_B_to_D_leveled_->data(),
            prod_B_to_D_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            current_rns_mod_count, d_tilda_leveled_->operator[](input1.depth_),
            d_leveled_->operator[](input1.depth_), r_prime_leveled_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        divide_round_lastq_extended_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp3_relin_new, temp2_relin_new, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp2_relin_new, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        addition<<<dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp2_relin_new, input1.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        gpuntt::GPU_NTT_Inplace(input1.data() +
                                    (current_decomp_count << (n_power + 1)),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                current_decomp_count, current_decomp_count);

        DeviceVector<Data64> temp_relin(
            (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_relin = temp_relin.data();
        Data64* temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            temp1_relin, modulus_->data(),
            base_change_matrix_D_to_Qtilda_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            prod_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            d_leveled_->operator[](input1.depth_), current_rns_mod_count,
            current_decomp_count, input1.depth_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(relin_key.host_location_, stream);
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_relin, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        divide_round_lastq_extended_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp2_relin, temp1_relin, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp1_relin, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        addition<<<dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp1_relin, input1.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator::rescale_inplace_ckks_leveled(Ciphertext& input1,
                                             const cudaStream_t stream)
    {
        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + (current_decomp_count - 1),
            .stream = stream};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        // int counter = first_rns_mod_count - 2;
        int counter = first_decomp_count - 1;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        DeviceVector<Data64> temp_rescale(
            (2 * n * Q_prime_size_) + (2 * n * Q_prime_size_), stream);
        Data64* temp1_rescale = temp_rescale.data();
        Data64* temp2_rescale = temp1_rescale + (2 * n * Q_prime_size_);

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            input1.data(),
            intt_table_->data() + ((current_decomp_count - 1) << n_power),
            modulus_->data() + (current_decomp_count - 1), cfg_intt, 2, 1,
            new_input_locations + ((input1.depth_ + P_size_) * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<dim3((n >> 8), 2, 1), 256,
                                                      0, stream>>>(
            input1.data(), temp1_rescale, modulus_->data(),
            rescaled_half_->data() + input1.depth_,
            rescaled_half_mod_->data() + location, n_power,
            current_decomp_count - 1, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(
            temp1_rescale, ntt_table_->data(), modulus_->data(), cfg_ntt,
            2 * (current_decomp_count - 1), (current_decomp_count - 1));

        move_cipher_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 2), 256, 0, stream>>>(
            input1.data(), temp2_rescale, n_power, current_decomp_count - 1);

        divide_round_lastq_rescale_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 2), 256, 0, stream>>>(
            temp1_rescale, temp2_rescale, input1.data(), modulus_->data(),
            rescaled_last_q_modinv_->data() + location, n_power,
            current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            input1.scale_ = input1.scale_ /
                            static_cast<double>(
                                prime_vector_[current_decomp_count - 1].value);
        }

        input1.depth_++;
    }

    __host__ void
    HEOperator::mod_drop_ckks_leveled_inplace(Ciphertext& input1,
                                              const cudaStream_t stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int offset1 = current_decomp_count << n_power;
        int offset2 = (current_decomp_count - 1) << n_power;

        DeviceVector<Data64> temp_mod_drop_(n * Q_size_, stream);
        Data64* temp_mod_drop = temp_mod_drop_.data();

        // TODO: do with efficient way!
        global_memory_replace_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 1), 256, 0, stream>>>(
            input1.data() + offset1, temp_mod_drop, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 1), 256, 0, stream>>>(
            temp_mod_drop, input1.data() + offset2, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input1.depth_++;
    }

    __host__ void HEOperator::mod_drop_ckks_leveled(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    const cudaStream_t stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory(
            (current_decomp_count * n * current_decomp_count), stream);

        global_memory_replace_offset_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 2), 256, 0, stream>>>(
            input1.data(), output_memory.data(), current_decomp_count, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::mod_drop_ckks_plaintext(Plaintext& input1,
                                                      Plaintext& output,
                                                      const cudaStream_t stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory(n * (current_decomp_count - 1),
                                           stream);

        global_memory_replace_kernel<<<
            dim3((n >> 8), current_decomp_count - 1, 1), 256, 0, stream>>>(
            input1.data(), output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.depth_ = input1.depth_ + 1;

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::mod_drop_ckks_plaintext_inplace(Plaintext& input1,
                                                const cudaStream_t stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        input1.depth_++;
    }

    __host__ void HEOperator::rotate_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key, int shift,
                                              const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_method_I(input1, output, galois_key, galoiselt,
                                  stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_method_I(in_data, output, galois_key, galois_elt,
                                      stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key, int shift,
                                               const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_method_II(input1, output, galois_key, galoiselt,
                                   stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_method_II(in_data, output, galois_key, galois_elt,
                                       stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_ckks_method_I(Ciphertext& input1,
                                                   Ciphertext& output,
                                                   Galoiskey& galois_key,
                                                   int shift,
                                                   const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_ckks_method_I(input1, output, galois_key, galoiselt,
                                       stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_ckks_method_I(in_data, output, galois_key,
                                           galois_elt, stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_ckks_method_II(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    Galoiskey& galois_key,
                                                    int shift,
                                                    const cudaStream_t stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n, galois_key.group_order_);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        if (key_exist)
        {
            apply_galois_ckks_method_II(input1, output, galois_key, galoiselt,
                                        stream);
        }
        else
        {
            std::vector<int> required_galoiselt;
            int shift_num = abs(shift);
            int negative = (shift < 0) ? (-1) : 1;
            while (shift_num != 0)
            {
                int power = int(log2(shift_num));
                int power_2 = pow(2, power);
                shift_num = shift_num - power_2;

                int index_in = power_2 * negative;

                if (!(galois_key.galois_elt.find(index_in) !=
                      galois_key.galois_elt.end()))
                {
                    throw std::logic_error("Galois key not present!");
                }
                galoiselt = galois_key.galois_elt[index_in];
                required_galoiselt.push_back(galoiselt);
            }

            Ciphertext& in_data = input1;
            for (auto& galois_elt : required_galoiselt)
            {
                apply_galois_ckks_method_II(in_data, output, galois_key,
                                            galois_elt, stream);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::apply_galois_method_I(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    Galoiskey& galois_key,
                                                    int galois_elt,
                                                    const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        bfv_duplicate_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, galois_key.device_location_[galois_elt].data(),
                temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::apply_galois_method_II(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     Galoiskey& galois_key,
                                                     int galois_elt,
                                                     const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        // TODO: make it efficient
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                       stream>>>(input1.data(), temp0_rotation,
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << n_power), temp2_rotation,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::apply_galois_ckks_method_I(
        Ciphertext& input1, Ciphertext& output, Galoiskey& galois_key,
        int galois_elt, const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) +
                (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: make it efficient
        ckks_duplicate_kernel<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                                stream>>>(
            temp0_rotation, temp2_rotation, modulus_->data(), n_power,
            first_rns_mod_count, current_rns_mod_count, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::apply_galois_ckks_method_II(
        Ciphertext& input1, Ciphertext& output, Galoiskey& galois_key,
        int galois_elt, const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * n * d_leveled_->operator[](0) * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream>>>(
            temp0_rotation + (current_decomp_count << n_power), temp3_rotation,
            modulus_->data(),
            base_change_matrix_D_to_Qtilda_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            prod_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            d_leveled_->operator[](input1.depth_), current_rns_mod_count,
            current_decomp_count, input1.depth_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, galois_key.device_location_[galois_elt].data(),
                temp4_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, current_rns_mod_count,
                d_leveled_->operator[](input1.depth_), input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(
                galois_key.host_location_[galois_elt], stream);
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp4_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::rotate_columns_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Galoiskey& galois_key,
                                                      const cudaStream_t stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        bfv_duplicate_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, galois_key.c_data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(galois_key.zero_host_location_,
                                              stream);
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galoiselt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::rotate_columns_method_II(Ciphertext& input1, Ciphertext& output,
                                         Galoiskey& galois_key,
                                         const cudaStream_t stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        // TODO: make it efficient
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                                       stream>>>(input1.data(), temp0_rotation,
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            input1.data() + (Q_size_ << n_power), temp2_rotation,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // MultSum
        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, galois_key.c_data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(galois_key.zero_host_location_,
                                              stream);
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        // ModDown + Permute
        divide_round_lastq_permute_bfv_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galoiselt, n_power, Q_prime_size_, Q_size_,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::switchkey_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Switchkey& switch_key,
                                                 const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) +
                                               (n * Q_size_ * Q_prime_size_) +
                                               (2 * n * Q_prime_size_),
                                           stream);
        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);

        cipher_broadcast_switchkey_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                            stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, switch_key.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            multiply_accumulate_kernel<<<dim3((n >> 8), Q_prime_size_, 1), 256,
                                         0, stream>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_switchkey_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                              0, stream>>>(
            temp2_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::switchkey_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Switchkey& switch_key,
                                                  const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation((2 * n * Q_size_) + (n * Q_size_) +
                                               (2 * n * Q_prime_size_ * d) +
                                               (2 * n * Q_prime_size_),
                                           stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);

        cipher_broadcast_switchkey_method_II_kernel<<<
            dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        base_conversion_DtoQtilde_relin_kernel<<<dim3((n >> 8), d, 1), 256, 0,
                                                 stream>>>(
            temp1_rotation, temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, d * Q_prime_size_,
                                Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, switch_key.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            multiply_accumulate_method_II_kernel<<<
                dim3((n >> 8), Q_prime_size_, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(temp3_rotation, intt_table_->data(),
                                modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                                Q_prime_size_);

        divide_round_lastq_extended_switchkey_kernel<<<
            dim3((n >> 8), Q_size_, 2), 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::switchkey_ckks_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Switchkey& switch_key,
                                                      const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) +
                (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_broadcast_switchkey_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            n_power, first_rns_mod_count, current_rns_mod_count,
            current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, switch_key.data(), temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = stream};

        gpuntt::GPU_NTT_Poly_Ordered_Inplace(
            temp3_rotation,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        divide_round_lastq_leveled_stage_one_kernel<<<dim3((n >> 8), 2, 1), 256,
                                                      0, stream>>>(
            temp3_rotation, temp2_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        // TODO: Merge with previous one
        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, current_decomp_count,
                                current_decomp_count);

        divide_round_lastq_leveled_stage_two_switchkey_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp2_rotation, temp3_rotation, temp1_rotation,
            output_memory.data(), modulus_->data(), last_q_modinv_->data(),
            n_power, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::switchkey_ckks_method_II(Ciphertext& input1, Ciphertext& output,
                                         Switchkey& switch_key,
                                         const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * n * d_leveled_->operator[](0) * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_broadcast_switchkey_method_II_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream>>>(
            temp2_rotation, temp3_rotation, modulus_->data(),
            base_change_matrix_D_to_Qtilda_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            prod_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            d_leveled_->operator[](input1.depth_), current_rns_mod_count,
            current_decomp_count, input1.depth_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, switch_key.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(switch_key.host_location_,
                                              stream);
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        divide_round_lastq_extended_leveled_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp4_rotation, temp3_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(temp3_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        // TODO: Fused the redundant kernels
        // TODO: Merge with previous one
        gpuntt::GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(),
                                modulus_->data(), cfg_ntt, current_decomp_count,
                                current_decomp_count);

        addition_switchkey<<<dim3((n >> 8), current_decomp_count, 2), 256, 0,
                             stream>>>(temp3_rotation, temp1_rotation,
                                       output_memory.data(), modulus_->data(),
                                       n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::conjugate_ckks_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Galoiskey& conjugate_key,
                                                      const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        int galois_elt = conjugate_key.galois_elt_zero;

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) +
                (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: make it efficient
        ckks_duplicate_kernel<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                                stream>>>(
            temp0_rotation, temp2_rotation, modulus_->data(), n_power,
            first_rns_mod_count, current_rns_mod_count, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        if (conjugate_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, conjugate_key.c_data(), temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(conjugate_key.zero_host_location_,
                                              stream);
            multiply_accumulate_leveled_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp3_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::conjugate_ckks_method_II(Ciphertext& input1, Ciphertext& output,
                                         Galoiskey& conjugate_key,
                                         const cudaStream_t stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        int galois_elt = conjugate_key.galois_elt_zero;

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * n * d_leveled_->operator[](0) * Q_prime_size_);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * current_decomp_count,
                        current_decomp_count);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream>>>(
            temp0_rotation + (current_decomp_count << n_power), temp3_rotation,
            modulus_->data(),
            base_change_matrix_D_to_Qtilda_leveled_->operator[](input1.depth_)
                .data(),
            Mi_inv_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            prod_D_to_Qtilda_leveled_->operator[](input1.depth_).data(),
            I_j_leveled_->operator[](input1.depth_).data(),
            I_location_leveled_->operator[](input1.depth_).data(), n_power,
            d_leveled_->operator[](input1.depth_), current_rns_mod_count,
            current_decomp_count, input1.depth_,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // MultSum
        // TODO: make it efficient
        if (conjugate_key.store_in_gpu_)
        {
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, conjugate_key.c_data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data64> key_location(conjugate_key.zero_host_location_,
                                              stream);
            multiply_accumulate_leveled_method_II_kernel<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // ModDown + Permute
        divide_round_lastq_permute_ckks_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
            temp4_rotation, temp0_rotation, output_memory.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), galois_elt, n_power, current_rns_mod_count,
            current_decomp_count, first_rns_mod_count, first_decomp_count,
            P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(output_memory.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt,
                                2 * current_decomp_count, current_decomp_count);

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                               Ciphertext& output, int index,
                                               const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        DeviceVector<Data64> temp(2 * n * Q_size_, stream);

        negacyclic_shift_poly_coeffmod_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream>>>(
            input1.data(), temp.data(), modulus_->data(), index, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: do with efficient way!
        global_memory_replace_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                       stream>>>(temp.data(),
                                                 output_memory.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void
    HEOperator::transform_to_ntt_bfv_plain(Plaintext& input1, Plaintext& output,
                                           const cudaStream_t stream)
    {
        DeviceVector<Data64> output_memory(n * Q_size_, stream);

        DeviceVector<Data64> temp_plain_mul(n * Q_size_, stream);
        Data64* temp1_plain_mul = temp_plain_mul.data();

        threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream>>>(
            input1.data(), temp1_plain_mul, modulus_->data(),
            upper_halfincrement_->data(), upper_threshold_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(temp1_plain_mul, output_memory.data(),
                        ntt_table_->data(), modulus_->data(), cfg_ntt, Q_size_,
                        Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::transform_to_ntt_bfv_cipher(
        Ciphertext& input1, Ciphertext& output, const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), output_memory.data(), ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    __host__ void HEOperator::transform_from_ntt_bfv_cipher(
        Ciphertext& input1, Ciphertext& output, const cudaStream_t stream)
    {
        int current_decomp_count = Q_size_ - input1.depth_;
        DeviceVector<Data64> output_memory((2 * n * current_decomp_count),
                                           stream);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(input1.data(), output_memory.data(),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.memory_set(std::move(output_memory));
    }

    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////
    //                       BOOTSRAPPING                         //
    ////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////

    __host__ Plaintext HEOperator::operator_plaintext(cudaStream_t stream)
    {
        Plaintext plain;

        plain.scheme_ = scheme_;
        switch (static_cast<int>(scheme_))
        {
            case 1: // BFV
                plain.plain_size_ = n;
                plain.depth_ = 0;
                plain.scale_ = 0;
                plain.in_ntt_domain_ = false;
                break;
            case 2: // CKKS
                plain.plain_size_ = n * Q_size_; // n
                plain.depth_ = 0;
                plain.scale_ = 0;
                plain.in_ntt_domain_ = true;
                break;
            default:
                break;
        }

        plain.device_locations_ =
            DeviceVector<Data64>(plain.plain_size_, stream);

        return plain;
    }

    __host__ Plaintext HEOperator::operator_from_plaintext(Plaintext& input,
                                                           cudaStream_t stream)
    {
        Plaintext plain;

        plain.scheme_ = input.scheme_;
        plain.plain_size_ = input.plain_size_;
        plain.depth_ = input.depth_;
        plain.scale_ = input.scale_;
        plain.in_ntt_domain_ = input.in_ntt_domain_;

        plain.device_locations_ =
            DeviceVector<Data64>(plain.plain_size_, stream);

        return plain;
    }

    __host__ Ciphertext HEOperator::operator_ciphertext(double scale,
                                                        cudaStream_t stream)
    {
        Ciphertext cipher;

        cipher.coeff_modulus_count_ = Q_size_;
        cipher.cipher_size_ = 2; // default
        cipher.ring_size_ = n; // n
        cipher.depth_ = 0;

        cipher.scheme_ = scheme_;
        cipher.in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        cipher.storage_type_ = storage_type::DEVICE;

        cipher.rescale_required_ = false;
        cipher.relinearization_required_ = false;
        cipher.scale_ = scale;

        int cipher_memory_size = 2 * (Q_size_ - cipher.depth_) * n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ Ciphertext
    HEOperator::operator_from_ciphertext(Ciphertext& input, cudaStream_t stream)
    {
        Ciphertext cipher;

        cipher.coeff_modulus_count_ = input.coeff_modulus_count_;
        cipher.cipher_size_ = input.cipher_size_;
        cipher.ring_size_ = input.ring_size_;
        cipher.depth_ = input.depth_;

        cipher.scheme_ = input.scheme_;
        cipher.in_ntt_domain_ = input.in_ntt_domain_;

        cipher.storage_type_ = storage_type::DEVICE;

        cipher.rescale_required_ = input.rescale_required_;
        cipher.relinearization_required_ = input.relinearization_required_;
        cipher.scale_ = input.scale_;

        int cipher_memory_size = 2 * (Q_size_ - cipher.depth_) * n;

        cipher.device_locations_ =
            DeviceVector<Data64>(cipher_memory_size, stream);

        return cipher;
    }

    __host__ void HEOperator::quick_ckks_encoder_vec_complex(Complex64* input,
                                                             Data64* output,
                                                             const double scale,
                                                             bool use_all_bases)
    {
        int rns_count = use_all_bases ? Q_prime_size_ : Q_size_;

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft = {
            .n_power = log_slot_count_,
            .fft_type = gpufft::type::INVERSE,
            .mod_inverse = Complex64(fix, 0.0),
            .stream = 0};

        gpufft::GPU_Special_FFT(input, special_ifft_roots_table_->data(),
                                cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(output, input, modulus_->data(),
                                               rns_count, two_pow_64_,
                                               reverse_order_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(output, ntt_table_->data(), modulus_->data(),
                                cfg_ntt, rns_count, rns_count);
    }

    __host__ void HEOperator::quick_ckks_encoder_constant_complex(
        Complex64 input, Data64* output, const double scale, bool use_all_bases)
    {
        // std::vector<Complex64> in = {input};
        std::vector<Complex64> in;
        for (int i = 0; i < slot_count_; i++)
        {
            in.push_back(input);
        }
        DeviceVector<Complex64> message_gpu(slot_count_);
        cudaMemcpy(message_gpu.data(), in.data(), in.size() * sizeof(Complex64),
                   cudaMemcpyHostToDevice);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        double fix = scale / static_cast<double>(slot_count_);

        gpufft::fft_configuration<Float64> cfg_ifft = {
            .n_power = log_slot_count_,
            .fft_type = gpufft::type::INVERSE,
            .mod_inverse = Complex64(fix, 0.0),
            .stream = 0};

        gpufft::GPU_Special_FFT(message_gpu.data(),
                                special_ifft_roots_table_->data(), cfg_ifft, 1);

        encode_kernel_ckks_conversion<<<dim3(((slot_count_) >> 8), 1, 1),
                                        256>>>(
            output, message_gpu.data(), modulus_->data(), Q_size_, two_pow_64_,
            reverse_order_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = 0};

        gpuntt::GPU_NTT_Inplace(output, ntt_table_->data(), modulus_->data(),
                                cfg_ntt, Q_size_, Q_size_);
    }

    __host__ void HEOperator::quick_ckks_encoder_constant_double(
        double input, Data64* output, const double scale, bool use_all_bases)
    {
        double value = input * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256>>>(
            output, value, modulus_->data(), Q_size_, two_pow_64_, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::quick_ckks_encoder_constant_integer(
        std::int64_t input, Data64* output, const double scale,
        bool use_all_bases)
    {
        double value = static_cast<double>(input) * scale;

        encode_kernel_double_ckks_conversion<<<dim3((n >> 8), 1, 1), 256>>>(
            output, value, modulus_->data(), Q_size_, two_pow_64_, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ std::vector<heongpu::DeviceVector<Data64>>
    HEOperator::encode_V_matrixs(Vandermonde& vandermonde, const double scale,
                                 bool use_all_bases)
    {
        std::vector<heongpu::DeviceVector<Data64>> result;

        int rns_count = use_all_bases ? Q_prime_size_ : Q_size_;

        for (int m = 0; m < vandermonde.StoC_piece_; m++)
        {
            heongpu::DeviceVector<Data64> temp_encoded(
                (vandermonde.V_matrixs_index_[m].size() * rns_count)
                << (vandermonde.log_num_slots_ + 1));

            for (int i = 0; i < vandermonde.V_matrixs_index_[m].size(); i++)
            {
                int matrix_location = (i << vandermonde.log_num_slots_);
                int plaintext_location =
                    ((i * rns_count) << (vandermonde.log_num_slots_ + 1));

                quick_ckks_encoder_vec_complex(
                    vandermonde.V_matrixs_rotated_[m].data() + matrix_location,
                    temp_encoded.data() + plaintext_location, scale,
                    use_all_bases);
            }

            result.push_back(std::move(temp_encoded));
        }

        return result;
    }

    __host__ std::vector<heongpu::DeviceVector<Data64>>
    HEOperator::encode_V_inv_matrixs(Vandermonde& vandermonde,
                                     const double scale, bool use_all_bases)
    {
        std::vector<heongpu::DeviceVector<Data64>> result;

        int rns_count = use_all_bases ? Q_prime_size_ : Q_size_;

        for (int m = 0; m < vandermonde.CtoS_piece_; m++)
        {
            heongpu::DeviceVector<Data64> temp_encoded(
                (vandermonde.V_inv_matrixs_index_[m].size() * rns_count)
                << (vandermonde.log_num_slots_ + 1));

            for (int i = 0; i < vandermonde.V_inv_matrixs_index_[m].size(); i++)
            {
                int matrix_location = (i << vandermonde.log_num_slots_);
                int plaintext_location =
                    ((i * rns_count) << (vandermonde.log_num_slots_ + 1));

                quick_ckks_encoder_vec_complex(
                    vandermonde.V_inv_matrixs_rotated_[m].data() +
                        matrix_location,
                    temp_encoded.data() + plaintext_location, scale,
                    use_all_bases);
            }

            result.push_back(std::move(temp_encoded));
        }

        return result;
    }

    __host__ Ciphertext HEOperator::multiply_matrix(
        Ciphertext& cipher, std::vector<heongpu::DeviceVector<Data64>>& matrix,
        std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
        Galoiskey& galois_key, const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher.stream();
        cipher.switch_stream(
            options.stream_); // TODO: Change copy and assign structure!
        Ciphertext result;
        result = cipher;
        cipher.switch_stream(
            old_stream); // TODO: Change copy and assign structure!

        int matrix_count = diags_matrices_bsgs_.size();
        for (int m = (matrix_count - 1); - 1 < m; m--)
        {
            int n1 = diags_matrices_bsgs_[m][0].size();
            int current_level = result.depth_;
            int current_decomp_count = (Q_size_ - current_level);

            DeviceVector<Data64> rotated_result =
                fast_single_hoisting_rotation_ckks(
                    result, diags_matrices_bsgs_[m][0], n1, galois_key,
                    options.stream_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                int real_shift = diags_matrices_bsgs_[m][j][0];

                Ciphertext inner_sum = operator_ciphertext(0, options.stream_);

                int matrix_plaintext_location = (counter * Q_size_) << n_power;
                int inner_n1 = diags_matrices_bsgs_[m][j].size();

                cipherplain_multiply_accumulate_kernel<<<
                    dim3((n >> 8), current_decomp_count, 2), 256, 0,
                    options.stream_>>>(
                    rotated_result.data(),
                    matrix[m].data() + matrix_plaintext_location,
                    inner_sum.data(), modulus_->data(), inner_n1,
                    current_decomp_count, Q_size_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                counter = counter + inner_n1;

                inner_sum.scheme_ = scheme_;
                inner_sum.ring_size_ = n;
                inner_sum.coeff_modulus_count_ = Q_size_;
                inner_sum.cipher_size_ = 2;
                inner_sum.depth_ = result.depth_;
                inner_sum.scale_ = result.scale_;
                inner_sum.in_ntt_domain_ = result.in_ntt_domain_;
                inner_sum.rescale_required_ = result.rescale_required_;
                inner_sum.relinearization_required_ =
                    result.relinearization_required_;

                rotate_rows_inplace(inner_sum, galois_key, real_shift, options);

                if (j == 0)
                {
                    cudaStream_t old_stream2 = inner_sum.stream();
                    inner_sum.switch_stream(
                        options.stream_); // TODO: Change copy and assign
                                          // structure!
                    result = inner_sum;
                    inner_sum.switch_stream(
                        old_stream2); // TODO: Change copy and assign structure!
                }
                else
                {
                    add(result, inner_sum, result, options);
                }
            }

            result.scale_ = result.scale_ * scale_boot_;
            result.rescale_required_ = true;
            rescale_inplace(result, options);
        }

        return result;
    }

    __host__ Ciphertext HEOperator::multiply_matrix_less_memory(
        Ciphertext& cipher, std::vector<heongpu::DeviceVector<Data64>>& matrix,
        std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
        std::vector<std::vector<std::vector<int>>>& real_shift,
        Galoiskey& galois_key, const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher.stream();
        cipher.switch_stream(
            options.stream_); // TODO: Change copy and assign structure!
        Ciphertext result;
        result = cipher;
        cipher.switch_stream(
            old_stream); // TODO: Change copy and assign structure!

        int matrix_count = diags_matrices_bsgs_.size();
        for (int m = (matrix_count - 1); - 1 < m; m--)
        {
            int n1 = diags_matrices_bsgs_[m][0].size();
            int current_level = result.depth_;
            int current_decomp_count = (Q_size_ - current_level);

            DeviceVector<Data64> rotated_result =
                fast_single_hoisting_rotation_ckks(
                    result, diags_matrices_bsgs_[m][0], n1, galois_key,
                    options.stream_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                Ciphertext inner_sum = operator_ciphertext(0, options.stream_);

                int matrix_plaintext_location = (counter * Q_size_) << n_power;
                int inner_n1 = diags_matrices_bsgs_[m][j].size();

                cipherplain_multiply_accumulate_kernel<<<
                    dim3((n >> 8), current_decomp_count, 2), 256, 0,
                    options.stream_>>>(
                    rotated_result.data(),
                    matrix[m].data() + matrix_plaintext_location,
                    inner_sum.data(), modulus_->data(), inner_n1,
                    current_decomp_count, Q_size_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                counter = counter + inner_n1;

                inner_sum.scheme_ = scheme_;
                inner_sum.ring_size_ = n;
                inner_sum.coeff_modulus_count_ = Q_size_;
                inner_sum.cipher_size_ = 2;
                inner_sum.depth_ = result.depth_;
                inner_sum.scale_ = result.scale_;
                inner_sum.in_ntt_domain_ = result.in_ntt_domain_;
                inner_sum.storage_type_ = result.storage_type_;
                inner_sum.rescale_required_ = result.rescale_required_;
                inner_sum.relinearization_required_ =
                    result.relinearization_required_;

                int real_shift_size = real_shift[m][j].size();
                for (int ss = 0; ss < real_shift_size; ss++)
                {
                    int shift_amount = real_shift[m][j][ss];
                    rotate_rows_inplace(inner_sum, galois_key, shift_amount,
                                        options);
                }

                if (j == 0)
                {
                    cudaStream_t old_stream2 = inner_sum.stream();
                    inner_sum.switch_stream(
                        options.stream_); // TODO: Change copy and assign
                                          // structure!
                    result = inner_sum;
                    inner_sum.switch_stream(
                        old_stream2); // TODO: Change copy and assign structure!
                }
                else
                {
                    add(result, inner_sum, result, options);
                }
            }
            result.scale_ = result.scale_ * scale_boot_;
            result.rescale_required_ = true;
            rescale_inplace(result, options);
        }

        return result;
    }

    __host__ std::vector<Ciphertext>
    HEOperator::coeff_to_slot(Ciphertext& cipher, Galoiskey& galois_key,
                              const ExecutionOptions& options)
    {
        Ciphertext c1;
        if (less_key_mode_)
        {
            c1 = multiply_matrix_less_memory(
                cipher, V_inv_matrixs_rotated_encoded_,
                diags_matrices_inv_bsgs_, real_shift_n2_inv_bsgs_, galois_key,
                options);
        }
        else
        {
            c1 = multiply_matrix(cipher, V_inv_matrixs_rotated_encoded_,
                                 diags_matrices_inv_bsgs_, galois_key, options);
        }

        Ciphertext c2 = operator_ciphertext(0, options.stream_);
        conjugate(c1, c2, galois_key, options); // conjugate

        Ciphertext result0 = operator_ciphertext(0, options.stream_);
        add(c1, c2, result0, options);

        int current_decomp_count = Q_size_ - result0.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            result0.data(), encoded_constant_1over2_.data(), result0.data(),
            modulus_->data(), n_power);
        result0.scale_ = result0.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result0.rescale_required_ = true;
        rescale_inplace(result0, options);

        Ciphertext result1 = operator_ciphertext(0, options.stream_);
        sub(c1, c2, result1, options);

        current_decomp_count = Q_size_ - result1.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            result1.data(), encoded_complex_minus_iover2_.data(),
            result1.data(), modulus_->data(), n_power);
        result1.scale_ = result1.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result1.rescale_required_ = true;
        rescale_inplace(result1, options);

        std::vector<Ciphertext> result;
        result.push_back(std::move(result0));
        result.push_back(std::move(result1));

        return result;
    }

    __host__ Ciphertext
    HEOperator::solo_coeff_to_slot(Ciphertext& cipher, Galoiskey& galois_key,
                                   const ExecutionOptions& options)
    {
        Ciphertext c1;
        if (less_key_mode_)
        {
            c1 = multiply_matrix_less_memory(
                cipher, V_inv_matrixs_rotated_encoded_,
                diags_matrices_inv_bsgs_, real_shift_n2_inv_bsgs_, galois_key,
                options);
        }
        else
        {
            c1 = multiply_matrix(cipher, V_inv_matrixs_rotated_encoded_,
                                 diags_matrices_inv_bsgs_, galois_key, options);
        }

        Ciphertext c2 = operator_ciphertext(0, options.stream_);
        conjugate(c1, c2, galois_key, options); // conjugate

        Ciphertext result = operator_ciphertext(0, options.stream_);
        add(c1, c2, result, options);

        int current_decomp_count = Q_size_ - result.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            result.data(), encoded_constant_1over2_.data(), result.data(),
            modulus_->data(), n_power);
        result.scale_ = result.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result.rescale_required_ = true;
        rescale_inplace(result, options);

        return result;
    }

    __host__ Ciphertext HEOperator::slot_to_coeff(
        Ciphertext& cipher0, Ciphertext& cipher1, Galoiskey& galois_key,
        const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher1.stream();
        cipher1.switch_stream(
            options.stream_); // TODO: Change copy and assign structure!
        Ciphertext result;
        result = cipher1;
        cipher1.switch_stream(
            old_stream); // TODO: Change copy and assign structure!

        int current_decomp_count = Q_size_ - cipher1.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            result.data(), encoded_complex_i_.data(), result.data(),
            modulus_->data(), n_power);
        result.scale_ = result.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        result.rescale_required_ = true;
        rescale_inplace(result, options);

        mod_drop_inplace(cipher0, options);

        add(result, cipher0, result, options);

        Ciphertext c1;
        if (less_key_mode_)
        {
            c1 = multiply_matrix_less_memory(
                result, V_matrixs_rotated_encoded_, diags_matrices_bsgs_,
                real_shift_n2_bsgs_, galois_key, options);
        }
        else
        {
            c1 = multiply_matrix(result, V_matrixs_rotated_encoded_,
                                 diags_matrices_bsgs_, galois_key, options);
        }

        return c1;
    }

    __host__ Ciphertext
    HEOperator::solo_slot_to_coeff(Ciphertext& cipher, Galoiskey& galois_key,
                                   const ExecutionOptions& options)
    {
        Ciphertext result;
        if (less_key_mode_)
        {
            result = multiply_matrix_less_memory(
                cipher, V_matrixs_rotated_encoded_, diags_matrices_bsgs_,
                real_shift_n2_bsgs_, galois_key, options);
        }
        else
        {
            result = multiply_matrix(cipher, V_matrixs_rotated_encoded_,
                                     diags_matrices_bsgs_, galois_key, options);
        }

        return result;
    }

    __host__ Ciphertext HEOperator::exp_scaled(Ciphertext& cipher,
                                               Relinkey& relin_key,
                                               const ExecutionOptions& options)
    {
        int current_decomp_count = Q_size_ - cipher.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            cipher.data(), encoded_complex_iscaleoverr_.data(), cipher.data(),
            modulus_->data(), n_power);
        cipher.scale_ = cipher.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        cipher.rescale_required_ = true;
        rescale_inplace(cipher, options);

        Ciphertext cipher_taylor =
            exp_taylor_approximation(cipher, relin_key, options);

        for (int i = 0; i < taylor_number_; i++)
        {
            multiply_inplace(cipher_taylor, cipher_taylor, options);
            relinearize_inplace(cipher_taylor, relin_key, options);
            rescale_inplace(cipher_taylor, options);
        }

        return cipher_taylor;
    }

    __host__ Ciphertext HEOperator::exp_taylor_approximation(
        Ciphertext& cipher, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        cudaStream_t old_stream = cipher.stream();
        cipher.switch_stream(
            options.stream_); // TODO: Change copy and assign structure!
        Ciphertext second;
        second = cipher; // 1 - c^1

        Ciphertext third = operator_ciphertext(0, options.stream_);
        multiply(second, second, third, options);
        relinearize_inplace(third, relin_key, options);
        rescale_inplace(third, options); // 2 - c^2

        mod_drop_inplace(second, options); // 2
        Ciphertext forth = operator_ciphertext(0, options.stream_);
        multiply(third, second, forth, options);
        relinearize_inplace(forth, relin_key, options);
        rescale_inplace(forth, options); // 3 - c^3

        Ciphertext fifth = operator_ciphertext(0, options.stream_);
        multiply(third, third, fifth, options);
        relinearize_inplace(fifth, relin_key, options);
        rescale_inplace(fifth, options); // 3 - c^4

        mod_drop_inplace(second, options); // 3
        Ciphertext sixth = operator_ciphertext(0, options.stream_);
        multiply(fifth, second, sixth, options);
        relinearize_inplace(sixth, relin_key, options);
        rescale_inplace(sixth, options); // 4 - c^5

        Ciphertext seventh = operator_ciphertext(0, options.stream_);
        multiply(forth, forth, seventh, options);
        relinearize_inplace(seventh, relin_key, options);
        rescale_inplace(seventh, options); // 4 - c^6

        Ciphertext eighth = operator_ciphertext(0, options.stream_);
        multiply(fifth, forth, eighth, options);
        relinearize_inplace(eighth, relin_key, options);
        rescale_inplace(eighth, options); // 4 - c^7

        //

        int current_decomp_count = Q_size_ - third.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            third.data(), encoded_constant_1over2_.data(), third.data(),
            modulus_->data(), n_power);
        third.scale_ = third.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        third.rescale_required_ = true;
        rescale_inplace(third, options); // 3

        //

        current_decomp_count = Q_size_ - forth.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            forth.data(), encoded_constant_1over6_.data(), forth.data(),
            modulus_->data(), n_power);
        forth.scale_ = forth.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        forth.rescale_required_ = true;
        rescale_inplace(forth, options); // 4

        //

        current_decomp_count = Q_size_ - fifth.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            fifth.data(), encoded_constant_1over24_.data(), fifth.data(),
            modulus_->data(), n_power);
        fifth.scale_ = fifth.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        fifth.rescale_required_ = true;
        rescale_inplace(fifth, options); // 4

        //

        current_decomp_count = Q_size_ - sixth.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            sixth.data(), encoded_constant_1over120_.data(), sixth.data(),
            modulus_->data(), n_power);
        sixth.scale_ = sixth.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        sixth.rescale_required_ = true;
        rescale_inplace(sixth, options); // 5

        //

        current_decomp_count = Q_size_ - seventh.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            seventh.data(), encoded_constant_1over720_.data(), seventh.data(),
            modulus_->data(), n_power);
        seventh.scale_ = seventh.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        seventh.rescale_required_ = true;
        rescale_inplace(seventh, options); // 5

        //

        current_decomp_count = Q_size_ - eighth.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            eighth.data(), encoded_constant_1over5040_.data(), eighth.data(),
            modulus_->data(), n_power);
        eighth.scale_ = eighth.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        eighth.rescale_required_ = true;
        rescale_inplace(eighth, options); // 5

        //

        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - second.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            second.data(), encoded_constant_1_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = second.depth_;
        result.scale_ = second.scale_;
        result.in_ntt_domain_ = second.in_ntt_domain_;
        result.rescale_required_ = second.rescale_required_;
        result.relinearization_required_ = second.relinearization_required_;

        //

        add_inplace(result, third, options); // 3

        //

        mod_drop_inplace(result, options); // 4

        //

        add_inplace(result, forth, options); // 4
        add_inplace(result, fifth, options); // 4

        //

        mod_drop_inplace(result, options); // 5

        //

        add_inplace(result, sixth, options); // 5
        add_inplace(result, seventh, options); // 5
        add_inplace(result, eighth, options); // 5

        return result;
    }

    __host__ DeviceVector<Data64>
    HEOperator::fast_single_hoisting_rotation_ckks_method_I(
        Ciphertext& first_cipher, std::vector<int>& bsgs_shift, int n1,
        Galoiskey& galois_key, const cudaStream_t stream)
    {
        int current_level = first_cipher.depth_;
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - current_level;
        int current_decomp_count = Q_size_ - current_level;

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) +
                (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);

        DeviceVector<Data64> result((2 * current_decomp_count * n1) << n_power,
                                    stream); // store n1 ciphertext

        // decompose and mult P
        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(first_cipher.data(), temp0_rotation,
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        2 * current_decomp_count, current_decomp_count);

        // TODO: make it efficient
        ckks_duplicate_kernel<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                                stream>>>(
            temp0_rotation, temp2_rotation, modulus_->data(), n_power,
            first_rns_mod_count, current_rns_mod_count, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < current_level; i++)
        {
            location += counter;
            counter--;
        }

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        //

        global_memory_replace_kernel<<<dim3((n >> 8), current_decomp_count, 2),
                                       256, 0, stream>>>(
            first_cipher.data(), result.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        //

        for (int i = 1; i < n1; i++)
        {
            int shift_n1 = bsgs_shift[i];
            int galoiselt =
                steps_to_galois_elt(shift_n1, n, galois_key.group_order_);
            int offset = ((2 * current_decomp_count) << n_power) * i;

            // MultSum
            // TODO: make it efficient
            if (galois_key.store_in_gpu_)
            {
                multiply_accumulate_leveled_kernel<<<
                    dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                    temp2_rotation,
                    galois_key.device_location_[galoiselt].data(),
                    temp3_rotation, modulus_->data(), first_rns_mod_count,
                    current_decomp_count, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data64> key_location(
                    galois_key.host_location_[galoiselt], stream);
                multiply_accumulate_leveled_kernel<<<
                    dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                    temp2_rotation, key_location.data(), temp3_rotation,
                    modulus_->data(), first_rns_mod_count, current_decomp_count,
                    n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                temp3_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
                2 * current_rns_mod_count, current_rns_mod_count,
                new_prime_locations + location);

            // ModDown + Permute
            divide_round_lastq_permute_ckks_kernel<<<
                dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
                temp3_rotation, temp0_rotation, result.data() + offset,
                modulus_->data(), half_p_->data(), half_mod_->data(),
                last_q_modinv_->data(), galoiselt, n_power,
                current_rns_mod_count, current_decomp_count,
                first_rns_mod_count, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT_Inplace(
                result.data() + offset, ntt_table_->data(), modulus_->data(),
                cfg_ntt, 2 * current_decomp_count, current_decomp_count);
        }

        return result;
    }

    __host__ DeviceVector<Data64>
    HEOperator::fast_single_hoisting_rotation_ckks_method_II(
        Ciphertext& first_cipher, std::vector<int>& bsgs_shift, int n1,
        Galoiskey& galois_key, const cudaStream_t stream)
    {
        int current_level = first_cipher.depth_;
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - current_level;
        int current_decomp_count = Q_size_ - current_level;

        DeviceVector<Data64> temp_rotation(
            (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                (2 * n * Q_prime_size_),
            stream);

        Data64* temp0_rotation = temp_rotation.data();
        Data64* temp1_rotation = temp0_rotation + (2 * n * Q_size_);
        Data64* temp2_rotation = temp1_rotation + (2 * n * Q_size_);
        Data64* temp3_rotation = temp2_rotation + (n * Q_size_);
        Data64* temp4_rotation =
            temp3_rotation +
            (2 * n * d_leveled_->operator[](0) * Q_prime_size_);

        DeviceVector<Data64> result((2 * current_decomp_count * n1) << n_power,
                                    stream); // store n1 ciphertext

        // decompose and mult P
        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = stream};

        gpuntt::GPU_NTT(first_cipher.data(), temp0_rotation,
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        2 * current_decomp_count, current_decomp_count);

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < current_level; i++)
        {
            location += counter;
            counter--;
        }

        base_conversion_DtoQtilde_relin_leveled_kernel<<<
            dim3((n >> 8), d_leveled_->operator[](current_level), 1), 256, 0,
            stream>>>(
            temp0_rotation + (current_decomp_count << n_power), temp3_rotation,
            modulus_->data(),
            base_change_matrix_D_to_Qtilda_leveled_->operator[](current_level)
                .data(),
            Mi_inv_D_to_Qtilda_leveled_->operator[](current_level).data(),
            prod_D_to_Qtilda_leveled_->operator[](current_level).data(),
            I_j_leveled_->operator[](current_level).data(),
            I_location_leveled_->operator[](current_level).data(), n_power,
            d_leveled_->operator[](current_level), current_rns_mod_count,
            current_decomp_count, current_level,
            prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](current_level) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        global_memory_replace_kernel<<<dim3((n >> 8), current_decomp_count, 2),
                                       256, 0, stream>>>(
            first_cipher.data(), result.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < n1; i++)
        {
            int shift_n1 = bsgs_shift[i];
            int galoiselt =
                steps_to_galois_elt(shift_n1, n, galois_key.group_order_);
            int offset = ((2 * current_decomp_count) << n_power) * i;

            // MultSum
            // TODO: make it efficient
            if (galois_key.store_in_gpu_)
            {
                multiply_accumulate_leveled_method_II_kernel<<<
                    dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                    temp3_rotation,
                    galois_key.device_location_[galoiselt].data(),
                    temp4_rotation, modulus_->data(), first_rns_mod_count,
                    current_decomp_count, current_rns_mod_count,
                    d_leveled_->operator[](first_cipher.depth_),
                    first_cipher.depth_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }
            else
            {
                DeviceVector<Data64> key_location(
                    galois_key.host_location_[galoiselt], stream);
                multiply_accumulate_leveled_method_II_kernel<<<
                    dim3((n >> 8), current_rns_mod_count, 1), 256, 0, stream>>>(
                    temp3_rotation, key_location.data(), temp4_rotation,
                    modulus_->data(), first_rns_mod_count, current_decomp_count,
                    current_rns_mod_count,
                    d_leveled_->operator[](first_cipher.depth_),
                    first_cipher.depth_, n_power);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
            }

            gpuntt::GPU_NTT_Modulus_Ordered_Inplace(
                temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
                2 * current_rns_mod_count, current_rns_mod_count,
                new_prime_locations + location);

            // ModDown + Permute
            divide_round_lastq_permute_ckks_kernel<<<
                dim3((n >> 8), current_decomp_count, 2), 256, 0, stream>>>(
                temp4_rotation, temp0_rotation, result.data() + offset,
                modulus_->data(), half_p_->data(), half_mod_->data(),
                last_q_modinv_->data(), galoiselt, n_power,
                current_rns_mod_count, current_decomp_count,
                first_rns_mod_count, Q_size_, P_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT_Inplace(
                result.data() + offset, ntt_table_->data(), modulus_->data(),
                cfg_ntt, 2 * current_decomp_count, current_decomp_count);
        }

        return result;
    }

    __host__ HEOperator::Vandermonde::Vandermonde(const int poly_degree,
                                                  const int CtoS_piece,
                                                  const int StoC_piece,
                                                  const bool less_key_mode)
    {
        poly_degree_ = poly_degree;
        num_slots_ = poly_degree_ >> 1;
        log_num_slots_ = int(log2l(num_slots_));

        CtoS_piece_ = CtoS_piece;
        StoC_piece_ = StoC_piece;

        generate_E_diagonals_index();
        generate_E_inv_diagonals_index();
        split_E();
        split_E_inv();

        generate_E_diagonals();
        generate_E_inv_diagonals();

        generate_V_n_lists();

        generate_pre_comp_V();
        generate_pre_comp_V_inv();

        generate_key_indexs(less_key_mode);
        key_indexs_ = unique_sort(key_indexs_);
    }

    __host__ void HEOperator::Vandermonde::generate_E_diagonals_index()
    {
        bool first = true;
        for (int i = 1; i < (log_num_slots_ + 1); i++)
        {
            if (first)
            {
                int block_size = num_slots_ >> i;
                E_index_.push_back(0);
                E_index_.push_back(block_size);
                first = false;

                E_size_.push_back(2);
            }
            else
            {
                int block_size = num_slots_ >> i;
                E_index_.push_back(0);
                E_index_.push_back(block_size);
                E_index_.push_back(num_slots_ - block_size);

                E_size_.push_back(3);
            }
        }
    }

    __host__ void HEOperator::Vandermonde::generate_E_inv_diagonals_index()
    {
        for (int i = log_num_slots_; 0 < i; i--)
        {
            if (i == 1)
            {
                int block_size = num_slots_ >> i;
                E_inv_index_.push_back(0);
                E_inv_index_.push_back(block_size);

                E_inv_size_.push_back(2);
            }
            else
            {
                int block_size = num_slots_ >> i;
                E_inv_index_.push_back(0);
                E_inv_index_.push_back(block_size);
                E_inv_index_.push_back(num_slots_ - block_size);

                E_inv_size_.push_back(3);
            }
        }
    }

    __host__ void HEOperator::Vandermonde::split_E()
    {
        // E_splitted
        int k = log_num_slots_ / StoC_piece_;
        int m = log_num_slots_ % StoC_piece_;

        for (int i = 0; i < StoC_piece_; i++)
        {
            E_splitted_.push_back(k);
        }

        for (int i = 0; i < m; i++)
        {
            E_splitted_[i]++;
        }

        int counter = 0;
        for (int i = 0; i < StoC_piece_; i++)
        {
            std::vector<int> temp;
            for (int j = 0; j < E_splitted_[i]; j++)
            {
                int size = (counter == 0) ? 2 : 3;
                for (int k = 0; k < size; k++)
                {
                    temp.push_back(E_index_[counter]);
                    counter++;
                }
            }
            E_splitted_index_.push_back(temp);
        }

        int num_slots_mask = num_slots_ - 1;
        counter = 0;
        for (int k = 0; k < StoC_piece_; k++)
        {
            int matrix_count = E_splitted_[k];
            int L_m_loc = (k == 0) ? 2 : 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;
            std::vector<int> diag_index_temp;
            std::vector<int> iteration_temp;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    iteration_temp.push_back(E_size_[counter]);
                    for (int i = 0; i < E_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_splitted_index_[k][i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    iteration_temp.push_back(index_mul_sorted.size());
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            V_matrixs_index_.push_back(index_mul_sorted);
            E_splitted_diag_index_gpu_.push_back(diag_index_temp);
            E_splitted_iteration_gpu_.push_back(iteration_temp);
            counter += matrix_count;
        }

        std::vector<std::unordered_map<int, int>> dict_output_index;
        for (int k = 0; k < StoC_piece_; k++)
        {
            std::unordered_map<int, int> temp;
            for (int i = 0; i < V_matrixs_index_[k].size(); i++)
            {
                temp[V_matrixs_index_[k][i]] = i;
            }
            dict_output_index.push_back(temp);
        }

        counter = 0;
        for (int k = 0; k < StoC_piece_; k++)
        {
            int matrix_count = E_splitted_[k];
            int L_m_loc = (k == 0) ? 2 : 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;

            std::vector<int> temp_in_index;
            std::vector<int> temp_out_index;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    for (int i = 0; i < E_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_splitted_index_[k][i];
                        for (int j = 0; j < E_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        temp_in_index.push_back(
                            dict_output_index[k][R_m_İNDEX]);
                        for (int j = 0; j < E_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX = E_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            counter += matrix_count;
            E_splitted_input_index_gpu_.push_back(temp_in_index);
            E_splitted_output_index_gpu_.push_back(temp_out_index);
        }
    }

    __host__ void HEOperator::Vandermonde::split_E_inv()
    {
        // E_inv_splitted
        int k = log_num_slots_ / CtoS_piece_;
        int m = log_num_slots_ % CtoS_piece_;

        for (int i = 0; i < CtoS_piece_; i++)
        {
            E_inv_splitted_.push_back(k);
        }

        for (int i = 0; i < m; i++)
        {
            E_inv_splitted_[i]++;
        }

        int counter = 0;
        for (int i = 0; i < CtoS_piece_; i++)
        {
            std::vector<int> temp;
            for (int j = 0; j < E_inv_splitted_[i]; j++)
            {
                int size = (counter == (E_inv_index_.size() - 2)) ? 2 : 3;
                for (int k = 0; k < size; k++)
                {
                    temp.push_back(E_inv_index_[counter]);
                    counter++;
                }
            }
            E_inv_splitted_index_.push_back(temp);
        }

        int num_slots_mask = num_slots_ - 1;
        counter = 0;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            int matrix_count = E_inv_splitted_[k];

            int L_m_loc = 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;
            std::vector<int> diag_index_temp;
            std::vector<int> iteration_temp;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    iteration_temp.push_back(E_inv_size_[counter]);
                    for (int i = 0; i < E_inv_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_inv_splitted_index_[k][i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_inv_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    iteration_temp.push_back(index_mul_sorted.size());
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        diag_index_temp.push_back(R_m_İNDEX);
                        for (int j = 0; j < E_inv_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            index_mul.push_back((L_m_İNDEX + R_m_İNDEX) &
                                                num_slots_mask);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            V_inv_matrixs_index_.push_back(index_mul_sorted);
            E_inv_splitted_diag_index_gpu_.push_back(diag_index_temp);
            E_inv_splitted_iteration_gpu_.push_back(iteration_temp);
            counter += matrix_count;
        }

        std::vector<std::unordered_map<int, int>> dict_output_index;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            std::unordered_map<int, int> temp;
            for (int i = 0; i < V_inv_matrixs_index_[k].size(); i++)
            {
                temp[V_inv_matrixs_index_[k][i]] = i;
            }
            dict_output_index.push_back(temp);
        }

        counter = 0;
        for (int k = 0; k < CtoS_piece_; k++)
        {
            int matrix_count = E_inv_splitted_[k];
            int L_m_loc = 3;
            std::vector<int> index_mul;
            std::vector<int> index_mul_sorted;

            std::vector<int> temp_in_index;
            std::vector<int> temp_out_index;
            for (int m = 0; m < matrix_count - 1; m++)
            {
                if (m == 0)
                {
                    for (int i = 0; i < E_inv_size_[counter]; i++)
                    {
                        int R_m_İNDEX = E_inv_splitted_index_[k][i];
                        for (int j = 0; j < E_inv_size_[counter + 1]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
                else
                {
                    for (int i = 0; i < index_mul_sorted.size(); i++)
                    {
                        int R_m_İNDEX = index_mul_sorted[i];
                        temp_in_index.push_back(
                            dict_output_index[k][R_m_İNDEX]);
                        for (int j = 0; j < E_inv_size_[counter + 1 + m]; j++)
                        {
                            int L_m_İNDEX =
                                E_inv_splitted_index_[k][L_m_loc + j];
                            int indexs =
                                (L_m_İNDEX + R_m_İNDEX) & num_slots_mask;
                            index_mul.push_back(indexs);
                            temp_out_index.push_back(
                                dict_output_index[k][indexs]);
                        }
                    }
                    index_mul_sorted = unique_sort(index_mul);
                    index_mul.clear();
                    L_m_loc += 3;
                }
            }
            counter += matrix_count;
            E_inv_splitted_input_index_gpu_.push_back(temp_in_index);
            E_inv_splitted_output_index_gpu_.push_back(temp_out_index);
        }
    }

    __host__ void HEOperator::Vandermonde::generate_E_diagonals()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        heongpu::DeviceVector<Complex64> V_logn_diagnal(
            ((3 * log_num_slots_) - 1) << log_num_slots_);
        E_diagonal_generate_kernel<<<dim3(blokcount, log_num_slots_, 1),
                                     bloksize>>>(V_logn_diagnal.data(),
                                                 log_num_slots_);

        int matrix_counter = 0;
        for (int i = 0; i < StoC_piece_; i++)
        {
            heongpu::DeviceVector<int> diag_index_gpu(
                E_splitted_diag_index_gpu_[i]);
            heongpu::DeviceVector<int> input_index_gpu(
                E_splitted_input_index_gpu_[i]);
            heongpu::DeviceVector<int> output_index_gpu(
                E_splitted_output_index_gpu_[i]);

            heongpu::DeviceVector<Complex64> V_mul((V_matrixs_index_[i].size())
                                                   << log_num_slots_);
            cudaMemset(V_mul.data(), 0, V_mul.size() * sizeof(Complex64));

            int input_loc;
            if (i == 0)
            {
                input_loc = 0;
            }
            else
            {
                input_loc = ((3 * matrix_counter) - 1) << log_num_slots_;
            }

            int R_matrix_counter = 0;
            int output_index_counter = 0;

            for (int j = 0; j < (E_splitted_[i] - 1); j++)
            {
                heongpu::DeviceVector<Complex64> temp_result(
                    (V_matrixs_index_[i].size()) << log_num_slots_);
                cudaMemset(temp_result.data(), 0,
                           temp_result.size() * sizeof(Complex64));

                bool first_check1 = (i == 0) ? true : false;
                bool first_check2 = (j == 0) ? true : false;

                E_diagonal_matrix_mult_kernel<<<blokcount, bloksize>>>(
                    V_logn_diagnal.data() + input_loc, temp_result.data(),
                    V_mul.data(), diag_index_gpu.data(), input_index_gpu.data(),
                    output_index_gpu.data(), E_splitted_iteration_gpu_[i][j],
                    R_matrix_counter, output_index_counter, j, first_check1,
                    first_check2, log_num_slots_);

                V_mul = std::move(temp_result);

                R_matrix_counter += E_splitted_iteration_gpu_[i][j];
                output_index_counter += (E_splitted_iteration_gpu_[i][j] * 3);
            }

            V_matrixs_.push_back(std::move(V_mul));
            matrix_counter += E_splitted_[i];
        }
    }

    __host__ void HEOperator::Vandermonde::generate_E_inv_diagonals()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        heongpu::DeviceVector<Complex64> V_inv_logn_diagnal(
            ((3 * log_num_slots_) - 1) << log_num_slots_);
        E_diagonal_inverse_generate_kernel<<<dim3(blokcount, log_num_slots_, 1),
                                             bloksize>>>(
            V_inv_logn_diagnal.data(), log_num_slots_);

        int matrix_counter = 0;
        for (int i = 0; i < CtoS_piece_; i++)
        {
            heongpu::DeviceVector<int> diag_index_gpu(
                E_inv_splitted_diag_index_gpu_[i]);
            heongpu::DeviceVector<int> input_index_gpu(
                E_inv_splitted_input_index_gpu_[i]);
            heongpu::DeviceVector<int> output_index_gpu(
                E_inv_splitted_output_index_gpu_[i]);

            heongpu::DeviceVector<Complex64> V_mul(
                (V_inv_matrixs_index_[i].size()) << log_num_slots_);
            cudaMemset(V_mul.data(), 0, V_mul.size() * sizeof(Complex64));

            int input_loc = (3 * matrix_counter) << log_num_slots_;
            int R_matrix_counter = 0;
            int output_index_counter = 0;

            for (int j = 0; j < (E_inv_splitted_[i] - 1); j++)
            {
                heongpu::DeviceVector<Complex64> temp_result(
                    (V_inv_matrixs_index_[i].size()) << log_num_slots_);
                cudaMemset(temp_result.data(), 0,
                           temp_result.size() * sizeof(Complex64));
                bool first_check = (j == 0) ? true : false;
                bool last_check = ((i == (CtoS_piece_ - 1)) &&
                                   (j == (E_inv_splitted_[i] - 2)))
                                      ? true
                                      : false;

                E_diagonal_inverse_matrix_mult_kernel<<<blokcount, bloksize>>>(
                    V_inv_logn_diagnal.data() + input_loc, temp_result.data(),
                    V_mul.data(), diag_index_gpu.data(), input_index_gpu.data(),
                    output_index_gpu.data(),
                    E_inv_splitted_iteration_gpu_[i][j], R_matrix_counter,
                    output_index_counter, j, first_check, last_check,
                    log_num_slots_);

                V_mul = std::move(temp_result);
                R_matrix_counter += E_inv_splitted_iteration_gpu_[i][j];
                output_index_counter +=
                    (E_inv_splitted_iteration_gpu_[i][j] * 3);
            }
            V_inv_matrixs_.push_back(std::move(V_mul));
            matrix_counter += E_inv_splitted_[i];
        }
    }

    __host__ void HEOperator::Vandermonde::generate_V_n_lists()
    {
        for (int i = 0; i < StoC_piece_; i++)
        {
            std::vector<std::vector<int>> result =
                heongpu::seperate_func(V_matrixs_index_[i]);

            int sizex = result.size();
            int sizex_2 = (sizex >> 1);

            std::vector<std::vector<int>> real_shift_n2;
            for (size_t l1 = 0; l1 < sizex_2; l1++)
            {
                std::vector<int> temp = {result[l1][0]};
                real_shift_n2.push_back(std::move(temp));
            }

            for (size_t l1 = sizex_2; l1 < sizex; l1++)
            {
                std::vector<int> temp;
                int fisrt_ = result[sizex_2][0];
                int second_ = result[l1][0] - result[sizex_2][0];

                if (second_ == 0)
                {
                    temp.push_back(fisrt_);
                }
                else
                {
                    temp.push_back(fisrt_);
                    temp.push_back(second_);
                }

                real_shift_n2.push_back(std::move(temp));
            }

            diags_matrices_bsgs_.push_back(std::move(result));
            real_shift_n2_bsgs_.push_back(std::move(real_shift_n2));
        }

        for (int i = 0; i < CtoS_piece_; i++)
        {
            std::vector<std::vector<int>> result =
                heongpu::seperate_func(V_inv_matrixs_index_[i]);

            int sizex = result.size();
            int sizex_2 = (sizex >> 1);

            std::vector<std::vector<int>> real_shift_n2;
            for (size_t l1 = 0; l1 < sizex_2; l1++)
            {
                std::vector<int> temp = {result[l1][0]};
                real_shift_n2.push_back(std::move(temp));
            }

            for (size_t l1 = sizex_2; l1 < sizex; l1++)
            {
                std::vector<int> temp;
                int fisrt_ = result[sizex_2][0];
                int second_ = result[l1][0] - result[sizex_2][0];

                if (second_ == 0)
                {
                    temp.push_back(fisrt_);
                }
                else
                {
                    temp.push_back(fisrt_);
                    temp.push_back(second_);
                }

                real_shift_n2.push_back(std::move(temp));
            }

            diags_matrices_inv_bsgs_.push_back(std::move(result));
            real_shift_n2_inv_bsgs_.push_back(std::move(real_shift_n2));
        }
    }

    __host__ void HEOperator::Vandermonde::generate_pre_comp_V()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        for (int m = 0; m < StoC_piece_; m++)
        {
            heongpu::DeviceVector<Complex64> temp_rotated(
                (V_matrixs_index_[m].size()) << log_num_slots_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
            {
                int real_shift = -(diags_matrices_bsgs_[m][j][0]);
                for (int i = 0; i < diags_matrices_bsgs_[m][j].size(); i++)
                {
                    int location = (counter << log_num_slots_);

                    vector_rotate_kernel<<<blokcount, bloksize>>>(
                        V_matrixs_[m].data() + location,
                        temp_rotated.data() + location, real_shift,
                        log_num_slots_);

                    counter++;
                }
            }

            V_matrixs_rotated_.push_back(std::move(temp_rotated));
        }
    }

    __host__ void HEOperator::Vandermonde::generate_pre_comp_V_inv()
    {
        int bloksize = (num_slots_ <= 1024) ? num_slots_ : 1024;
        int blokcount = (num_slots_ + (1023)) / 1024;

        for (int m = 0; m < CtoS_piece_; m++)
        {
            heongpu::DeviceVector<Complex64> temp_rotated(
                (V_inv_matrixs_index_[m].size()) << log_num_slots_);

            int counter = 0;
            for (int j = 0; j < diags_matrices_inv_bsgs_[m].size(); j++)
            {
                int real_shift = -(diags_matrices_inv_bsgs_[m][j][0]);
                for (int i = 0; i < diags_matrices_inv_bsgs_[m][j].size(); i++)
                {
                    int location = (counter << log_num_slots_);

                    vector_rotate_kernel<<<blokcount, bloksize>>>(
                        V_inv_matrixs_[m].data() + location,
                        temp_rotated.data() + location, real_shift,
                        log_num_slots_);

                    counter++;
                }
            }

            V_inv_matrixs_rotated_.push_back(std::move(temp_rotated));
        }
    }

    __host__ void
    HEOperator::Vandermonde::generate_key_indexs(const bool less_key_mode)
    {
        if (less_key_mode)
        {
            for (int m = 0; m < CtoS_piece_; m++)
            {
                key_indexs_.insert(key_indexs_.end(),
                                   diags_matrices_inv_bsgs_[m][0].begin(),
                                   diags_matrices_inv_bsgs_[m][0].end());
                for (int j = 0; j < diags_matrices_inv_bsgs_[m].size(); j++)
                {
                    key_indexs_.push_back(real_shift_n2_inv_bsgs_[m][j][0]);
                }
            }

            for (int m = 0; m < StoC_piece_; m++)
            {
                key_indexs_.insert(key_indexs_.end(),
                                   diags_matrices_bsgs_[m][0].begin(),
                                   diags_matrices_bsgs_[m][0].end());
                for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
                {
                    key_indexs_.push_back(real_shift_n2_bsgs_[m][j][0]);
                }
            }
        }
        else
        {
            for (int m = 0; m < CtoS_piece_; m++)
            {
                key_indexs_.insert(key_indexs_.end(),
                                   diags_matrices_inv_bsgs_[m][0].begin(),
                                   diags_matrices_inv_bsgs_[m][0].end());
                for (int j = 0; j < diags_matrices_inv_bsgs_[m].size(); j++)
                {
                    key_indexs_.push_back(diags_matrices_inv_bsgs_[m][j][0]);
                }
            }

            for (int m = 0; m < StoC_piece_; m++)
            {
                key_indexs_.insert(key_indexs_.end(),
                                   diags_matrices_bsgs_[m][0].begin(),
                                   diags_matrices_bsgs_[m][0].end());
                for (int j = 0; j < diags_matrices_bsgs_[m].size(); j++)
                {
                    key_indexs_.push_back(diags_matrices_bsgs_[m][j][0]);
                }
            }
        }
    }

    HEArithmeticOperator::HEArithmeticOperator(Parameters& context,
                                               HEEncoder& encoder)
        : HEOperator(context, encoder)
    {
    }

    __host__ void HEArithmeticOperator::generate_bootstrapping_params(
        const double scale, const BootstrappingConfig& config)
    {
        if (!boot_context_generated_)
        {
            if (!(scheme_ == scheme_type::ckks))
            {
                throw std::invalid_argument(
                    "HEonGPU currently does not support bootstrapping "
                    "operation except for CKKS scheme!");
            }

            scale_boot_ = scale;
            CtoS_piece_ = config.CtoS_piece_;
            StoC_piece_ = config.StoC_piece_;
            taylor_number_ = config.taylor_number_;
            less_key_mode_ = config.less_key_mode_;

            // TODO: remove it!
            bool use_all_bases = false; // Do not change it!

            Vandermonde matrix_gen(n, CtoS_piece_, StoC_piece_, less_key_mode_);

            V_matrixs_rotated_encoded_ =
                encode_V_matrixs(matrix_gen, scale_boot_, use_all_bases);
            V_inv_matrixs_rotated_encoded_ =
                encode_V_inv_matrixs(matrix_gen, scale_boot_, use_all_bases);

            V_matrixs_index_ = matrix_gen.V_matrixs_index_;
            V_inv_matrixs_index_ = matrix_gen.V_inv_matrixs_index_;

            diags_matrices_bsgs_ = matrix_gen.diags_matrices_bsgs_;
            diags_matrices_inv_bsgs_ = matrix_gen.diags_matrices_inv_bsgs_;

            if (less_key_mode_)
            {
                real_shift_n2_bsgs_ = matrix_gen.real_shift_n2_bsgs_;
                real_shift_n2_inv_bsgs_ = matrix_gen.real_shift_n2_inv_bsgs_;
            }

            key_indexs_ = matrix_gen.key_indexs_;

            // Pre-computed encoded parameters
            // CtoS
            double constant_1over2 = 0.5;
            encoded_constant_1over2_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over2, encoded_constant_1over2_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            Complex64 complex_minus_iover2(0.0, -0.5);
            encoded_complex_minus_iover2_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_minus_iover2, encoded_complex_minus_iover2_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // StoC
            Complex64 complex_i(0, 1);
            encoded_complex_i_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_i, encoded_complex_i_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Scale part
            Complex64 complex_minus_iscale(
                0.0, -(((static_cast<double>(prime_vector_[0].value) * 0.25) /
                        (scale_boot_ * M_PI))));
            encoded_complex_minus_iscale_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_minus_iscale, encoded_complex_minus_iscale_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Exponentiate
            Complex64 complex_iscaleoverr(
                0.0, (((2 * M_PI * scale_boot_) /
                       static_cast<double>(prime_vector_[0].value))) /
                         static_cast<double>(1 << taylor_number_));
            encoded_complex_iscaleoverr_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_iscaleoverr, encoded_complex_iscaleoverr_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Sinus taylor
            double constant_1 = 1.0;
            encoded_constant_1_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1, encoded_constant_1_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over6 = 1.0 / 6.0;
            encoded_constant_1over6_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over6, encoded_constant_1over6_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over24 = 1.0 / 24.0;
            encoded_constant_1over24_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(constant_1over24,
                                               encoded_constant_1over24_.data(),
                                               scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over120 = 1.0 / 120.0;
            encoded_constant_1over120_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over120, encoded_constant_1over120_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over720 = 1.0 / 720.0;
            encoded_constant_1over720_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over720, encoded_constant_1over720_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over5040 = 1.0 / 5040.0;
            encoded_constant_1over5040_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over5040, encoded_constant_1over5040_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            boot_context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Bootstrapping parameters is locked after "
                                     "generation and cannot be modified.");
        }

        cudaDeviceSynchronize();
    }

    __host__ Ciphertext HEArithmeticOperator::regular_bootstrapping(
        Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // Raise modulus
        int current_decomp_count = Q_size_ - input1.depth_;
        if (current_decomp_count != 1)
        {
            throw std::logic_error("Ciphertexts leveled should be at max!");
        }

        ExecutionOptions options_inner =
            ExecutionOptions()
                .set_stream(options.stream_)
                .set_storage_type(storage_type::DEVICE)
                .set_initial_location(true);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        DeviceVector<Data64> input_intt_poly(2 * n, options.stream_);
        input_storage_manager(
            input1,
            [&](Ciphertext& input1_)
            {
                gpuntt::GPU_NTT(input1.data(), input_intt_poly.data(),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                2, 1);
            },
            options, false);

        Ciphertext c_raised =
            operator_ciphertext(scale_boot_, options_inner.stream_);
        mod_raise_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                           options_inner.stream_>>>(
            input_intt_poly.data(), c_raised.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(c_raised.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_size_,
                                Q_size_);

        // Coeff to slot
        std::vector<heongpu::Ciphertext> enc_results =
            coeff_to_slot(c_raised, galois_key, options_inner); // c_raised

        // Exponentiate
        Ciphertext ciph_neg_exp0 =
            operator_ciphertext(0, options_inner.stream_);
        Ciphertext ciph_exp0 =
            exp_scaled(enc_results[0], relin_key, options_inner);

        Ciphertext ciph_neg_exp1 =
            operator_ciphertext(0, options_inner.stream_);
        Ciphertext ciph_exp1 =
            exp_scaled(enc_results[1], relin_key, options_inner);

        // Compute sine
        Ciphertext ciph_sin0 = operator_ciphertext(0, options_inner.stream_);
        conjugate(ciph_exp0, ciph_neg_exp0, galois_key,
                  options_inner); // conjugate
        sub(ciph_exp0, ciph_neg_exp0, ciph_sin0, options_inner);

        Ciphertext ciph_sin1 = operator_ciphertext(0, options_inner.stream_);
        conjugate(ciph_exp1, ciph_neg_exp1, galois_key,
                  options_inner); // conjugate
        sub(ciph_exp1, ciph_neg_exp1, ciph_sin1, options_inner);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin0.depth_;
        cipherplain_multiplication_kernel<<<dim3((n >> 8), current_decomp_count,
                                                 2),
                                            256, 0, options_inner.stream_>>>(
            ciph_sin0.data(), encoded_complex_minus_iscale_.data(),
            ciph_sin0.data(), modulus_->data(), n_power);
        ciph_sin0.scale_ = ciph_sin0.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin0.rescale_required_ = true;
        rescale_inplace(ciph_sin0, options_inner);

        current_decomp_count = Q_size_ - ciph_sin1.depth_;
        cipherplain_multiplication_kernel<<<dim3((n >> 8), current_decomp_count,
                                                 2),
                                            256, 0, options_inner.stream_>>>(
            ciph_sin1.data(), encoded_complex_minus_iscale_.data(),
            ciph_sin1.data(), modulus_->data(), n_power);
        ciph_sin1.scale_ = ciph_sin1.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin1.rescale_required_ = true;
        rescale_inplace(ciph_sin1, options_inner);

        // Slot to coeff
        Ciphertext StoC_results =
            slot_to_coeff(ciph_sin0, ciph_sin1, galois_key, options_inner);
        StoC_results.scale_ = scale_boot_;

        return StoC_results;
    }

    __host__ Ciphertext HEArithmeticOperator::slim_bootstrapping(
        Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // Raise modulus
        int current_decomp_count = Q_size_ - input1.depth_;
        if (current_decomp_count != (1 + StoC_piece_))
        {
            throw std::logic_error("Ciphertexts leveled should be at max!");
        }

        ExecutionOptions options_inner =
            ExecutionOptions()
                .set_stream(options.stream_)
                .set_storage_type(storage_type::DEVICE)
                .set_initial_location(true);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        // Slot to coeff
        Ciphertext StoC_results =
            solo_slot_to_coeff(input1, galois_key, options_inner);

        DeviceVector<Data64> input_intt_poly(2 * n, options.stream_);
        input_storage_manager(
            StoC_results,
            [&](Ciphertext& StoC_results_)
            {
                gpuntt::GPU_NTT(StoC_results.data(), input_intt_poly.data(),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                2, 1);
            },
            options, false);

        Ciphertext c_raised =
            operator_ciphertext(scale_boot_, options_inner.stream_);
        mod_raise_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                           options_inner.stream_>>>(
            input_intt_poly.data(), c_raised.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(c_raised.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_size_,
                                Q_size_);

        // Coeff to slot
        Ciphertext CtoS_results =
            solo_coeff_to_slot(c_raised, galois_key, options_inner);

        // Exponentiate
        Ciphertext ciph_neg_exp = operator_ciphertext(0, options_inner.stream_);
        Ciphertext ciph_exp =
            exp_scaled(CtoS_results, relin_key, options_inner);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options_inner.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options_inner); // conjugate
        sub(ciph_exp, ciph_neg_exp, ciph_sin, options_inner);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<dim3((n >> 8), current_decomp_count,
                                                 2),
                                            256, 0, options_inner.stream_>>>(
            ciph_sin.data(), encoded_complex_minus_iscale_.data(),
            ciph_sin.data(), modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options_inner);
        ciph_sin.scale_ = scale_boot_;

        return ciph_sin;
    }

    HELogicOperator::HELogicOperator(Parameters& context, HEEncoder& encoder,
                                     double scale)
        : HEOperator(context, encoder)
    {
        if (scheme_ == scheme_type::bfv)
        {
            // TODO: make it efficinet
            Data64 constant_1 = 1ULL;
            encoded_constant_one_ = DeviceVector<Data64>(slot_count_);
            fill_device_vector<<<dim3((n >> 8), 1, 1), 256>>>(
                encoded_constant_one_.data(), constant_1, slot_count_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
                .n_power = n_power,
                .ntt_type = gpuntt::INVERSE,
                .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
                .zero_padding = false,
                .mod_inverse = n_plain_inverse_->data(),
                .stream = 0};

            gpuntt::GPU_NTT_Inplace(
                encoded_constant_one_.data(), plain_intt_tables_->data(),
                plain_modulus_pointer_->data(), cfg_intt, 1, 1);
        }
        else if (scheme_ == scheme_type::ckks)
        {
            if (scale == 0.0)
            {
                throw std::invalid_argument(
                    "Scale can not be zero for CKKS Scheme");
            }

            double constant_1 = 1.0;
            encoded_constant_one_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1, encoded_constant_one_.data(), scale);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            throw std::invalid_argument("Invalid Scheme Type");
        }
    }

    __host__ void HELogicOperator::generate_bootstrapping_params(
        const double scale, const BootstrappingConfig& config,
        const logic_bootstrapping_type& boot_type)
    {
        if (!boot_context_generated_)
        {
            if (!(scheme_ == scheme_type::ckks))
            {
                throw std::invalid_argument(
                    "HEonGPU currently does not support bootstrapping "
                    "operation except for CKKS scheme!");
            }

            int division = static_cast<int>(
                round(static_cast<double>(prime_vector_[0].value) / scale));

            switch (static_cast<int>(boot_type))
            {
                case 1: // BIT_BOOTSTRAPPING
                    if ((division != 2))
                    {
                        throw std::invalid_argument(
                            "Bootstrapping parameters can not be generated, "
                            "because of context is not suitable for Bit "
                            "Bootstrapping. Last modulus should be 2*scale!");
                    }
                    break;
                case 2: // GATE_BOOTSTRAPPING
                    if ((division != 3))
                    {
                        throw std::invalid_argument(
                            "Bootstrapping parameters can not be generated, "
                            "because of context is not suitable for Gate "
                            "Bootstrapping. Last modulus should be 3*scale!");
                    }

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            /*
            enum class logic_bootstrapping_type : std::uint8_t
            {
                NONE = 0x0,
                BIT_BOOTSTRAPPING = 0x1, // scale = q0 / 2. More detail:
                                        // https://eprint.iacr.org/2024/767.pdf
                GATE_BOOTSTRAPPING = 0x2, // scale = q0 / 3. More detail:
                                        // https://eprint.iacr.org/2024/767.pdf
            };
            */

            scale_boot_ = scale;
            CtoS_piece_ = config.CtoS_piece_;
            StoC_piece_ = config.StoC_piece_;
            taylor_number_ = config.taylor_number_;
            less_key_mode_ = config.less_key_mode_;

            // TODO: remove it!
            bool use_all_bases = false; // Do not change it!

            Vandermonde matrix_gen(n, CtoS_piece_, StoC_piece_, less_key_mode_);

            V_matrixs_rotated_encoded_ =
                encode_V_matrixs(matrix_gen, scale_boot_, use_all_bases);
            V_inv_matrixs_rotated_encoded_ =
                encode_V_inv_matrixs(matrix_gen, scale_boot_, use_all_bases);

            V_matrixs_index_ = matrix_gen.V_matrixs_index_;
            V_inv_matrixs_index_ = matrix_gen.V_inv_matrixs_index_;

            diags_matrices_bsgs_ = matrix_gen.diags_matrices_bsgs_;
            diags_matrices_inv_bsgs_ = matrix_gen.diags_matrices_inv_bsgs_;

            if (less_key_mode_)
            {
                real_shift_n2_bsgs_ = matrix_gen.real_shift_n2_bsgs_;
                real_shift_n2_inv_bsgs_ = matrix_gen.real_shift_n2_inv_bsgs_;
            }

            key_indexs_ = matrix_gen.key_indexs_;

            // Pre-computed encoded parameters
            // CtoS
            double constant_1over2 = 0.5;
            encoded_constant_1over2_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over2, encoded_constant_1over2_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            Complex64 complex_minus_iover2(0.0, -0.5);
            encoded_complex_minus_iover2_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_minus_iover2, encoded_complex_minus_iover2_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // StoC
            Complex64 complex_i(0.0, 1.0);
            encoded_complex_i_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_i, encoded_complex_i_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Scale part
            Complex64 complex_minus_iscale(
                0.0, -(((static_cast<double>(prime_vector_[0].value) * 0.25) /
                        (scale_boot_ * M_PI))));
            encoded_complex_minus_iscale_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_minus_iscale, encoded_complex_minus_iscale_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Exponentiate
            Complex64 complex_iscaleoverr(
                0.0, (((2 * M_PI * scale_boot_) /
                       static_cast<double>(prime_vector_[0].value))) /
                         static_cast<double>(1 << taylor_number_));
            encoded_complex_iscaleoverr_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_iscaleoverr, encoded_complex_iscaleoverr_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Sinus taylor
            double constant_1 = 1.0;
            encoded_constant_1_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1, encoded_constant_1_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over6 = 1.0 / 6.0;
            encoded_constant_1over6_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over6, encoded_constant_1over6_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over24 = 1.0 / 24.0;
            encoded_constant_1over24_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(constant_1over24,
                                               encoded_constant_1over24_.data(),
                                               scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over120 = 1.0 / 120.0;
            encoded_constant_1over120_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over120, encoded_constant_1over120_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over720 = 1.0 / 720.0;
            encoded_constant_1over720_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over720, encoded_constant_1over720_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_1over5040 = 1.0 / 5040.0;
            encoded_constant_1over5040_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over5040, encoded_constant_1over5040_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Bit bootstrapping
            double constant_minus_1over4 = -0.25;
            encoded_constant_minus_1over4_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_minus_1over4, encoded_constant_minus_1over4_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            // Gate bootstrapping
            double constant_1over3_ = 1.0 / 3.0;
            encoded_constant_1over3_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_1over3_, encoded_constant_1over3_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_2over3_ = 2.0 / 3.0;
            encoded_constant_2over3_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_2over3_, encoded_constant_2over3_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            Complex64 complex_minus_2over6j_(0.0, (1.0 / 3.0));
            encoded_complex_minus_2over6j_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_minus_2over6j_, encoded_complex_minus_2over6j_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_minus_2over6_ = -(1.0 / 3.0);
            encoded_constant_minus_2over6_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_minus_2over6_, encoded_constant_minus_2over6_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            Complex64 complex_2over6j_(0.0, (-1.0 / 3.0));
            encoded_complex_2over6j_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_complex(
                complex_2over6j_, encoded_complex_2over6j_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_2over6_ = 1.0 / 3.0;
            encoded_constant_2over6_ = DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_2over6_, encoded_constant_2over6_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_pioversome_ =
                prime_vector_[0].value / (12.0 * scale_boot_);
            encoded_constant_pioversome_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_pioversome_, encoded_constant_pioversome_.data(),
                scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            double constant_minus_pioversome_ =
                -((prime_vector_[0].value) / (12.0 * scale_boot_));
            encoded_constant_minus_pioversome_ =
                DeviceVector<Data64>(Q_size_ << n_power);
            quick_ckks_encoder_constant_double(
                constant_minus_pioversome_,
                encoded_constant_minus_pioversome_.data(), scale_boot_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            boot_context_generated_ = true;
        }
        else
        {
            throw std::runtime_error("Bootstrapping parameters is locked after "
                                     "generation and cannot be modified.");
        }

        cudaDeviceSynchronize();
    }

    __host__ Ciphertext HELogicOperator::bit_bootstrapping(
        Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // Raise modulus
        int current_decomp_count = Q_size_ - input1.depth_;
        if (current_decomp_count != (1 + StoC_piece_))
        {
            throw std::logic_error("Ciphertexts leveled should be at max!");
        }

        ExecutionOptions options_inner =
            ExecutionOptions()
                .set_stream(options.stream_)
                .set_storage_type(storage_type::DEVICE)
                .set_initial_location(true);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        // Slot to coeff
        Ciphertext StoC_results =
            solo_slot_to_coeff(input1, galois_key, options_inner);

        DeviceVector<Data64> input_intt_poly(2 * n, options.stream_);
        input_storage_manager(
            StoC_results,
            [&](Ciphertext& StoC_results_)
            {
                gpuntt::GPU_NTT(StoC_results.data(), input_intt_poly.data(),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                2, 1);
            },
            options, false);

        Ciphertext c_raised =
            operator_ciphertext(scale_boot_, options_inner.stream_);
        mod_raise_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                           options_inner.stream_>>>(
            input_intt_poly.data(), c_raised.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(c_raised.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_size_,
                                Q_size_);

        // Coeff to slot
        Ciphertext CtoS_results =
            solo_coeff_to_slot(c_raised, galois_key, options_inner);

        // Exponentiate
        Ciphertext ciph_neg_exp = operator_ciphertext(0, options_inner.stream_);
        Ciphertext ciph_exp =
            exp_scaled(CtoS_results, relin_key, options_inner);

        // Compute cosine
        Ciphertext ciph_cos = operator_ciphertext(0, options_inner.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options_inner); // conjugate
        add(ciph_exp, ciph_neg_exp, ciph_cos, options_inner);

        // Scale
        current_decomp_count = Q_size_ - ciph_cos.depth_;
        cipherplain_multiplication_kernel<<<dim3((n >> 8), current_decomp_count,
                                                 2),
                                            256, 0, options_inner.stream_>>>(
            ciph_cos.data(), encoded_constant_minus_1over4_.data(),
            ciph_cos.data(), modulus_->data(), n_power);
        ciph_cos.scale_ = ciph_cos.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_cos.rescale_required_ = true;
        rescale_inplace(ciph_cos, options_inner);

        //

        Ciphertext result = operator_ciphertext(0, options_inner.stream_);
        current_decomp_count = Q_size_ - ciph_cos.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options_inner.stream_>>>(
            ciph_cos.data(), encoded_constant_1over2_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_cos.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_cos.in_ntt_domain_;
        result.rescale_required_ = ciph_cos.rescale_required_;
        result.relinearization_required_ = ciph_cos.relinearization_required_;

        //

        return result;
    }

    __host__ Ciphertext HELogicOperator::gate_bootstrapping(
        logic_gate gate_type, Ciphertext& input1, Ciphertext& input2,
        Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        if (!boot_context_generated_)
        {
            throw std::invalid_argument(
                "Bootstrapping operation can not be performed before "
                "generating Bootstrapping parameters!");
        }

        // Raise modulus
        int current_decomp_count = Q_size_ - input1.depth_;
        if (current_decomp_count != (1 + StoC_piece_))
        {
            throw std::logic_error("Ciphertexts leveled should be at max!");
        }

        current_decomp_count = Q_size_ - input2.depth_;
        if (current_decomp_count != (1 + StoC_piece_))
        {
            throw std::logic_error("Ciphertexts leveled should be at max!");
        }

        ExecutionOptions options_inner =
            ExecutionOptions()
                .set_stream(options.stream_)
                .set_storage_type(storage_type::DEVICE)
                .set_initial_location(true);

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power,
            .ntt_type = gpuntt::INVERSE,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power,
            .ntt_type = gpuntt::FORWARD,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        Ciphertext input_ = operator_ciphertext(0, options_inner.stream_);
        add(input1, input2, input_);

        // Slot to coeff
        Ciphertext StoC_results =
            solo_slot_to_coeff(input_, galois_key, options_inner);

        DeviceVector<Data64> input_intt_poly(2 * n, options.stream_);
        input_storage_manager(
            StoC_results,
            [&](Ciphertext& StoC_results_)
            {
                gpuntt::GPU_NTT(StoC_results.data(), input_intt_poly.data(),
                                intt_table_->data(), modulus_->data(), cfg_intt,
                                2, 1);
            },
            options, false);

        Ciphertext c_raised =
            operator_ciphertext(scale_boot_, options_inner.stream_);
        mod_raise_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                           options_inner.stream_>>>(
            input_intt_poly.data(), c_raised.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::GPU_NTT_Inplace(c_raised.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, 2 * Q_size_,
                                Q_size_);

        // Coeff to slot
        Ciphertext CtoS_results =
            solo_coeff_to_slot(c_raised, galois_key, options_inner);

        Ciphertext result = operator_ciphertext(0, options_inner.stream_);

        switch (gate_type)
        {
            case logic_gate::AND:
                result = AND_approximation(CtoS_results, galois_key, relin_key,
                                           options_inner);
                break;
            case logic_gate::OR:
                result = OR_approximation(CtoS_results, galois_key, relin_key,
                                          options_inner);
                break;
            case logic_gate::XOR:
                result = XOR_approximation(CtoS_results, galois_key, relin_key,
                                           options_inner);
                break;
            case logic_gate::NAND:
                result = NAND_approximation(CtoS_results, galois_key, relin_key,
                                            options_inner);
                break;
            case logic_gate::NOR:
                result = NOR_approximation(CtoS_results, galois_key, relin_key,
                                           options_inner);
                break;
            case logic_gate::XNOR:
                result = XNOR_approximation(CtoS_results, galois_key, relin_key,
                                            options_inner);
                break;
            default:
                throw std::invalid_argument("Unknown Gate Type!");
        }

        return result;
    }

    __host__ Ciphertext HELogicOperator::AND_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        //////////////////////////////
        // plain add
        Ciphertext cipher_add = operator_ciphertext(0, options.stream_);
        int current_decomp_count = Q_size_ - cipher.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            cipher.data(), encoded_constant_pioversome_.data(),
            cipher_add.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        cipher_add.scheme_ = scheme_;
        cipher_add.ring_size_ = n;
        cipher_add.coeff_modulus_count_ = Q_size_;
        cipher_add.cipher_size_ = 2;
        cipher_add.depth_ = cipher.depth_;
        cipher_add.scale_ = cipher.scale_;
        cipher_add.in_ntt_domain_ = cipher.in_ntt_domain_;
        cipher_add.rescale_required_ = cipher.rescale_required_;
        cipher_add.relinearization_required_ = cipher.relinearization_required_;
        //////////////////////////////

        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher_add, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        sub(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_complex_minus_2over6j_.data(),
            ciph_sin.data(), modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_1over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ Ciphertext HELogicOperator::OR_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        add(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        int current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_minus_2over6_.data(),
            ciph_sin.data(), modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_2over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ Ciphertext HELogicOperator::XOR_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        //////////////////////////////
        // plain add
        Ciphertext cipher_add = operator_ciphertext(0, options.stream_);
        int current_decomp_count = Q_size_ - cipher.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            cipher.data(), encoded_constant_minus_pioversome_.data(),
            cipher_add.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        cipher_add.scheme_ = scheme_;
        cipher_add.ring_size_ = n;
        cipher_add.coeff_modulus_count_ = Q_size_;
        cipher_add.cipher_size_ = 2;
        cipher_add.depth_ = cipher.depth_;
        cipher_add.scale_ = cipher.scale_;
        cipher_add.in_ntt_domain_ = cipher.in_ntt_domain_;
        cipher_add.rescale_required_ = cipher.rescale_required_;
        cipher_add.relinearization_required_ = cipher.relinearization_required_;
        //////////////////////////////

        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher_add, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        sub(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_complex_2over6j_.data(), ciph_sin.data(),
            modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_1over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ Ciphertext HELogicOperator::NAND_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        //////////////////////////////
        // plain add
        Ciphertext cipher_add = operator_ciphertext(0, options.stream_);
        int current_decomp_count = Q_size_ - cipher.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            cipher.data(), encoded_constant_pioversome_.data(),
            cipher_add.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        cipher_add.scheme_ = scheme_;
        cipher_add.ring_size_ = n;
        cipher_add.coeff_modulus_count_ = Q_size_;
        cipher_add.cipher_size_ = 2;
        cipher_add.depth_ = cipher.depth_;
        cipher_add.scale_ = cipher.scale_;
        cipher_add.in_ntt_domain_ = cipher.in_ntt_domain_;
        cipher_add.rescale_required_ = cipher.rescale_required_;
        cipher_add.relinearization_required_ = cipher.relinearization_required_;
        //////////////////////////////

        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher_add, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        sub(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_complex_2over6j_.data(), ciph_sin.data(),
            modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_2over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ Ciphertext HELogicOperator::NOR_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        add(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        int current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_2over6_.data(), ciph_sin.data(),
            modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_1over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ Ciphertext HELogicOperator::XNOR_approximation(
        Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
        const ExecutionOptions& options)
    {
        //////////////////////////////
        // plain add
        Ciphertext cipher_add = operator_ciphertext(0, options.stream_);
        int current_decomp_count = Q_size_ - cipher.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            cipher.data(), encoded_constant_minus_pioversome_.data(),
            cipher_add.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        cipher_add.scheme_ = scheme_;
        cipher_add.ring_size_ = n;
        cipher_add.coeff_modulus_count_ = Q_size_;
        cipher_add.cipher_size_ = 2;
        cipher_add.depth_ = cipher.depth_;
        cipher_add.scale_ = cipher.scale_;
        cipher_add.in_ntt_domain_ = cipher.in_ntt_domain_;
        cipher_add.rescale_required_ = cipher.rescale_required_;
        cipher_add.relinearization_required_ = cipher.relinearization_required_;
        //////////////////////////////

        Ciphertext ciph_neg_exp = operator_ciphertext(0, options.stream_);
        Ciphertext ciph_exp = exp_scaled(cipher_add, relin_key, options);

        // Compute sine
        Ciphertext ciph_sin = operator_ciphertext(0, options.stream_);
        conjugate(ciph_exp, ciph_neg_exp, galois_key,
                  options); // conjugate
        sub(ciph_exp, ciph_neg_exp, ciph_sin, options);

        // Scale
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, options.stream_>>>(
            ciph_sin.data(), encoded_complex_minus_2over6j_.data(),
            ciph_sin.data(), modulus_->data(), n_power);
        ciph_sin.scale_ = ciph_sin.scale_ * scale_boot_;
        HEONGPU_CUDA_CHECK(cudaGetLastError());
        ciph_sin.rescale_required_ = true;
        rescale_inplace(ciph_sin, options);

        //////////////////////////////
        // plain add
        Ciphertext result = operator_ciphertext(0, options.stream_);
        current_decomp_count = Q_size_ - ciph_sin.depth_;
        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, options.stream_>>>(
            ciph_sin.data(), encoded_constant_2over3_.data(), result.data(),
            modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        result.scheme_ = scheme_;
        result.ring_size_ = n;
        result.coeff_modulus_count_ = Q_size_;
        result.cipher_size_ = 2;
        result.depth_ = ciph_sin.depth_;
        result.scale_ = scale_boot_;
        result.in_ntt_domain_ = ciph_sin.in_ntt_domain_;
        result.rescale_required_ = ciph_sin.rescale_required_;
        result.relinearization_required_ = ciph_sin.relinearization_required_;
        //////////////////////////////

        return result;
    }

    __host__ void
    HELogicOperator::one_minus_cipher(Ciphertext& input1, Ciphertext& output,
                                      const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        int current_decomp_count = Q_size_ - input1.depth_;

        if (scheme_ == scheme_type::bfv)
        {
            addition_plain_bfv_poly<<<dim3((n >> 8), current_decomp_count, 2),
                                      256, 0, options.stream_>>>(
                input1.data(), encoded_constant_one_.data(), output.data(),
                modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
                coeeff_div_plainmod_->data(), n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else if (scheme_ == scheme_type::ckks)
        {
            addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count, 2),
                                       256, 0, options.stream_>>>(
                input1.data(), encoded_constant_one_.data(), output.data(),
                modulus_->data(), n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            throw std::invalid_argument("Invalid Scheme Type");
        }
    }

    __host__ void
    HELogicOperator::one_minus_cipher_inplace(Ciphertext& input1,
                                              const ExecutionOptions& options)
    {
        // TODO: make it efficient
        negate_inplace(input1, options);

        int current_decomp_count = Q_size_ - input1.depth_;

        if (scheme_ == scheme_type::bfv)
        {
            addition_plain_bfv_poly_inplace<<<dim3((n >> 8),
                                                   current_decomp_count, 1),
                                              256, 0, options.stream_>>>(
                input1.data(), encoded_constant_one_.data(), input1.data(),
                modulus_->data(), plain_modulus_, Q_mod_t_, upper_threshold_,
                coeeff_div_plainmod_->data(), n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else if (scheme_ == scheme_type::ckks)
        {
            addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                       options.stream_>>>(
                input1.data(), encoded_constant_one_.data(), input1.data(),
                modulus_->data(), n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            throw std::invalid_argument("Invalid Scheme Type");
        }
    }

} // namespace heongpu
