// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "operator.cuh"

namespace heongpu
{

    __host__ HEOperator::HEOperator(Parameters& context)
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

        // Temp
        if (scheme_ == scheme_type::bfv)
        {
            temp_mul = DeviceVector<Data>((4 * n * (bsk_mod_count_ + Q_size_)) +
                                          (3 * n * (bsk_mod_count_ + Q_size_)));
            temp1_mul = temp_mul.data();
            temp2_mul = temp1_mul + (4 * n * (bsk_mod_count_ + Q_size_));

            if (context.keyswitching_type_ ==
                keyswitching_type::KEYSWITCHING_METHOD_I)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (n * Q_size_ * Q_prime_size_) +
                    (2 * n * Q_prime_size_));
                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (n * Q_size_ * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_II)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (n * Q_size_) +
                    (2 * n * Q_prime_size_ * d) + (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (n * Q_size_);
                temp3_rotation = temp2_rotation + (2 * n * Q_prime_size_ * d);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_III)
            {
                temp_relin_new = DeviceVector<Data>(
                    (n * d * r_prime) + (2 * n * d_tilda * r_prime) +
                    (2 * n * Q_prime_size_));
                temp1_relin_new = temp_relin_new.data();
                temp2_relin_new = temp1_relin_new + (n * d * r_prime);
                temp3_relin_new = temp2_relin_new + (2 * n * d_tilda * r_prime);
            }
            else
            {
            }

            temp_rescale = DeviceVector<Data>((2 * n * Q_prime_size_) +
                                              (2 * n * Q_prime_size_));
            temp1_rescale = temp_rescale.data();
            temp2_rescale = temp1_rescale + (2 * n * Q_prime_size_);

            temp_plain_mul = DeviceVector<Data>(n * Q_size_);
            temp1_plain_mul = temp_plain_mul.data();
        }
        else if (scheme_ == scheme_type::ckks)
        {
            if (context.keyswitching_type_ ==
                keyswitching_type::KEYSWITCHING_METHOD_I)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (2 * n * Q_size_) +
                    (n * Q_size_ * Q_prime_size_) + (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (2 * n * Q_size_);
                temp3_rotation = temp2_rotation + (n * Q_size_ * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_II)
            {
                temp_relin = DeviceVector<Data>((n * Q_size_ * Q_prime_size_) +
                                                (2 * n * Q_prime_size_));
                temp1_relin = temp_relin.data();
                temp2_relin = temp1_relin + (n * Q_size_ * Q_prime_size_);

                temp_rotation = DeviceVector<Data>(
                    (2 * n * Q_size_) + (2 * n * Q_size_) + (n * Q_size_) +
                    (2 * n * d_leveled_->operator[](0) * Q_prime_size_) +
                    (2 * n * Q_prime_size_));

                temp0_rotation = temp_rotation.data();
                temp1_rotation = temp0_rotation + (2 * n * Q_size_);
                temp2_rotation = temp1_rotation + (2 * n * Q_size_);
                temp3_rotation = temp2_rotation + (n * Q_size_);
                temp4_rotation =
                    temp3_rotation +
                    (2 * n * d_leveled_->operator[](0) * Q_prime_size_);
            }
            else if (context.keyswitching_type_ ==
                     keyswitching_type::KEYSWITCHING_METHOD_III)
            {
                temp_relin_new = DeviceVector<Data>(
                    (n * d_leveled_->operator[](0) * r_prime_leveled_) +
                    (2 * n * d_tilda_leveled_->operator[](0) *
                     r_prime_leveled_) +
                    (2 * n * Q_prime_size_));
                temp1_relin_new = temp_relin_new.data();
                temp2_relin_new =
                    temp1_relin_new +
                    (n * d_leveled_->operator[](0) * r_prime_leveled_);
                temp3_relin_new =
                    temp2_relin_new + (2 * n * d_tilda_leveled_->operator[](0) *
                                       r_prime_leveled_);
            }
            else
            {
            }

            temp_rescale = DeviceVector<Data>((2 * n * Q_prime_size_) +
                                              (2 * n * Q_prime_size_));
            temp1_rescale = temp_rescale.data();
            temp2_rescale = temp1_rescale + (2 * n * Q_prime_size_);

            // cudaMalloc(&temp_mod_drop, n * Q_size_ * sizeof(Data));
            temp_mod_drop_ = DeviceVector<Data>(n * Q_size_);
            temp_mod_drop = temp_mod_drop_.data();
        }
        else
        {
            throw std::invalid_argument("Invalid Scheme Type");
        }

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
    }

    __host__ void HEOperator::add(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output)
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

        if (input1.locations_.size() <
                (cipher_size * n * current_decomp_count) ||
            input2.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        addition<<<dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ =
            (input1.rescale_required_ || input2.rescale_required_);
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::add(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output, HEStream& stream)
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

        if (input1.locations_.size() <
                (cipher_size * n * current_decomp_count) ||
            input2.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        addition<<<dim3((n >> 8), current_decomp_count, cipher_size), 256, 0,
                   stream.stream>>>(input1.data(), input2.data(), output.data(),
                                    modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ =
            (input1.rescale_required_ || input2.rescale_required_);
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::sub(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output)
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

        if (input1.locations_.size() <
                (cipher_size * n * current_decomp_count) ||
            input2.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        substraction<<<dim3((n >> 8), current_decomp_count, cipher_size),
                       256>>>(input1.data(), input2.data(), output.data(),
                              modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ =
            (input1.rescale_required_ || input2.rescale_required_);
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::sub(Ciphertext& input1, Ciphertext& input2,
                                  Ciphertext& output, HEStream& stream)
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

        if (input1.locations_.size() <
                (cipher_size * n * current_decomp_count) ||
            input2.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        substraction<<<dim3((n >> 8), current_decomp_count, cipher_size), 256,
                       0, stream.stream>>>(input1.data(), input2.data(),
                                           output.data(), modulus_->data(),
                                           n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ =
            (input1.rescale_required_ || input2.rescale_required_);
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::negate(Ciphertext& input1, Ciphertext& output)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        negation<<<dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), output.data(), modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ = input1.rescale_required_;
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::negate(Ciphertext& input1, Ciphertext& output,
                                     HEStream& stream)
    {
        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        negation<<<dim3((n >> 8), current_decomp_count, cipher_size), 256, 0,
                   stream.stream>>>(input1.data(), output.data(),
                                    modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.scheme_ = scheme_;
        output.ring_size_ = n;
        output.coeff_modulus_count_ = Q_size_;
        output.cipher_size_ = cipher_size;
        output.depth_ = input1.depth_;
        output.in_ntt_domain_ = input1.in_ntt_domain_;
        output.scale_ = input1.scale_;
        output.rescale_required_ = input1.rescale_required_;
        output.relinearization_required_ = input1.relinearization_required_;
    }

    __host__ void HEOperator::add_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        addition_plain_bfv_poly<<<
            dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::add_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output,
                                            HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        addition_plain_bfv_poly<<<dim3((n >> 8), current_decomp_count,
                                       cipher_size),
                                  256, 0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::add_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::add_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2,
                                                    HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256, 0, stream.stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::add_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        addition_plain_ckks_poly<<<
            dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::add_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output,
                                             HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        addition_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count,
                                        cipher_size),
                                   256, 0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::add_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition<<<dim3((n >> 8), current_decomp_count, 1), 256>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::add_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2,
                                                     HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        addition<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                   stream.stream>>>(input1.data(), input2.data(), input1.data(),
                                    modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        substraction_plain_bfv_poly<<<
            dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::sub_plain_bfv(Ciphertext& input1,
                                            Plaintext& input2,
                                            Ciphertext& output,
                                            HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        substraction_plain_bfv_poly<<<dim3((n >> 8), current_decomp_count,
                                           cipher_size),
                                      256, 0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::sub_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_bfv_inplace(Ciphertext& input1,
                                                    Plaintext& input2,
                                                    HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < n)
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction_plain_bfv_poly_inplace<<<
            dim3((n >> 8), current_decomp_count, 1), 256, 0, stream.stream>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            plain_modulus_, Q_mod_t_, upper_threshold_,
            coeeff_div_plainmod_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count));
        }

        substraction_plain_ckks_poly<<<
            dim3((n >> 8), current_decomp_count, cipher_size), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::sub_plain_ckks(Ciphertext& input1,
                                             Plaintext& input2,
                                             Ciphertext& output,
                                             HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        if (output.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            output.resize((cipher_size * n * current_decomp_count), stream);
        }

        substraction_plain_ckks_poly<<<dim3((n >> 8), current_decomp_count,
                                            cipher_size),
                                       256, 0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        output.cipher_size_ = cipher_size;
    }

    __host__ void HEOperator::sub_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction<<<dim3((n >> 8), current_decomp_count, 1), 256>>>(
            input1.data(), input2.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::sub_plain_ckks_inplace(Ciphertext& input1,
                                                     Plaintext& input2,
                                                     HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int cipher_size = input1.relinearization_required_ ? 3 : 2;

        if (input1.locations_.size() < (cipher_size * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (input2.size() < (n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Plaintext size!");
        }

        substraction<<<dim3((n >> 8), current_decomp_count, 1), 256, 0,
                       stream.stream>>>(input1.data(), input2.data(),
                                        input1.data(), modulus_->data(),
                                        n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::multiply_bfv(Ciphertext& input1,
                                           Ciphertext& input2,
                                           Ciphertext& output)
    {
        if ((input1.in_ntt_domain_ != false) ||
            (input2.in_ntt_domain_ != false))
        {
            throw std::invalid_argument("Ciphertexts should be in same domain");
        }

        if (input1.locations_.size() < (2 * n * Q_size_) ||
            input2.locations_.size() < (2 * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (3 * n * Q_size_))
        {
            output.resize((3 * n * Q_size_));
        }

        fast_convertion<<<dim3((n >> 8), 4, 1), 256>>>(
            input1.data(), input2.data(), temp1_mul, modulus_->data(),
            base_Bsk_->data(), m_tilde_, inv_prod_q_mod_m_tilde_,
            inv_m_tilde_mod_Bsk_->data(), prod_q_mod_Bsk_->data(),
            base_change_matrix_Bsk_->data(),
            base_change_matrix_m_tilde_->data(),
            inv_punctured_prod_mod_base_array_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = q_Bsk_n_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(temp1_mul, q_Bsk_merge_ntt_tables_->data(),
                        q_Bsk_merge_modulus_->data(), cfg_ntt,
                        ((bsk_mod_count_ + Q_size_) * 4),
                        (bsk_mod_count_ + Q_size_));

        cross_multiplication<<<dim3((n >> 8), (bsk_mod_count_ + Q_size_), 1),
                               256>>>(
            temp1_mul, temp1_mul + (((bsk_mod_count_ + Q_size_) * 2) * n),
            temp2_mul, q_Bsk_merge_modulus_->data(), n_power,
            (bsk_mod_count_ + Q_size_));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp2_mul, q_Bsk_merge_intt_tables_->data(),
                        q_Bsk_merge_modulus_->data(), cfg_intt,
                        (3 * (bsk_mod_count_ + Q_size_)),
                        (bsk_mod_count_ + Q_size_));

        fast_floor<<<dim3((n >> 8), 3, 1), 256>>>(
            temp2_mul, output.data(), modulus_->data(), base_Bsk_->data(),
            plain_modulus_, inv_punctured_prod_mod_base_array_->data(),
            base_change_matrix_Bsk_->data(), inv_prod_q_mod_Bsk_->data(),
            inv_punctured_prod_mod_B_array_->data(),
            base_change_matrix_q_->data(), base_change_matrix_msk_->data(),
            inv_prod_B_mod_m_sk_, prod_B_mod_q_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::multiply_bfv(Ciphertext& input1,
                                           Ciphertext& input2,
                                           Ciphertext& output, HEStream& stream)
    {
        if ((input1.in_ntt_domain_ != false) ||
            (input2.in_ntt_domain_ != false))
        {
            throw std::invalid_argument("Ciphertexts should be in same domain");
        }

        if (input1.locations_.size() < (2 * n * Q_size_) ||
            input2.locations_.size() < (2 * n * Q_size_))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (3 * n * Q_size_))
        {
            output.resize((3 * n * Q_size_), stream);
        }

        fast_convertion<<<dim3((n >> 8), 4, 1), 256, 0, stream.stream>>>(
            input1.data(), input2.data(), stream.temp1_mul, modulus_->data(),
            base_Bsk_->data(), m_tilde_, inv_prod_q_mod_m_tilde_,
            inv_m_tilde_mod_Bsk_->data(), prod_q_mod_Bsk_->data(),
            base_change_matrix_Bsk_->data(),
            base_change_matrix_m_tilde_->data(),
            inv_punctured_prod_mod_base_array_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = q_Bsk_n_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_mul, q_Bsk_merge_ntt_tables_->data(),
                        q_Bsk_merge_modulus_->data(), cfg_ntt,
                        ((bsk_mod_count_ + Q_size_) * 4),
                        (bsk_mod_count_ + Q_size_));

        cross_multiplication<<<dim3((n >> 8), (bsk_mod_count_ + Q_size_), 1),
                               256, 0, stream.stream>>>(
            stream.temp1_mul,
            stream.temp1_mul + (((bsk_mod_count_ + Q_size_) * 2) * n),
            stream.temp2_mul, q_Bsk_merge_modulus_->data(), n_power,
            (bsk_mod_count_ + Q_size_));
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp2_mul, q_Bsk_merge_intt_tables_->data(),
                        q_Bsk_merge_modulus_->data(), cfg_intt,
                        (3 * (bsk_mod_count_ + Q_size_)),
                        (bsk_mod_count_ + Q_size_));

        fast_floor<<<dim3((n >> 8), 3, 1), 256, 0, stream.stream>>>(
            stream.temp2_mul, output.data(), modulus_->data(),
            base_Bsk_->data(), plain_modulus_,
            inv_punctured_prod_mod_base_array_->data(),
            base_change_matrix_Bsk_->data(), inv_prod_q_mod_Bsk_->data(),
            inv_punctured_prod_mod_B_array_->data(),
            base_change_matrix_q_->data(), base_change_matrix_msk_->data(),
            inv_prod_B_mod_m_sk_, prod_B_mod_q_->data(), n_power, Q_size_,
            bsk_mod_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::multiply_ckks(Ciphertext& input1,
                                            Ciphertext& input2,
                                            Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.locations_.size() < (2 * n * current_decomp_count) ||
            input2.locations_.size() < (2 * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (3 * n * current_decomp_count))
        {
            output.resize((3 * n * current_decomp_count));
        }

        cross_multiplication<<<dim3((n >> 8), (current_decomp_count), 1),
                               256>>>(input1.data(), input2.data(),
                                      output.data(), modulus_->data(), n_power,
                                      current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void HEOperator::multiply_ckks(Ciphertext& input1,
                                            Ciphertext& input2,
                                            Ciphertext& output,
                                            HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        if (input1.locations_.size() < (2 * n * current_decomp_count) ||
            input2.locations_.size() < (2 * n * current_decomp_count))
        {
            throw std::invalid_argument("Invalid Ciphertexts size!");
        }

        if (output.locations_.size() < (3 * n * current_decomp_count))
        {
            output.resize((3 * n * current_decomp_count), stream);
        }

        cross_multiplication<<<dim3((n >> 8), (current_decomp_count), 1), 256,
                               0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void HEOperator::multiply_plain_bfv(Ciphertext& input1,
                                                 Plaintext& input2,
                                                 Ciphertext& output)
    {
        if (input1.in_ntt_domain_)
        {
            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
                input1.data(), input2.data(), output.data(), modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256>>>(
                input2.data(), temp1_plain_mul, modulus_->data(),
                upper_halfincrement_->data(), upper_threshold_, n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = 0};

            ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                              .ntt_type = INVERSE,
                                              .reduction_poly =
                                                  ReductionPolynomial::X_N_plus,
                                              .zero_padding = false,
                                              .mod_inverse = n_inverse_->data(),
                                              .stream = 0};

            GPU_NTT_Inplace(temp1_plain_mul, ntt_table_->data(),
                            modulus_->data(), cfg_ntt, Q_size_, Q_size_);

            GPU_NTT(input1.data(), output.data(), ntt_table_->data(),
                    modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);

            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
                output.data(), temp1_plain_mul, output.data(), modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            GPU_NTT_Inplace(output.data(), intt_table_->data(),
                            modulus_->data(), cfg_intt, 2 * Q_size_, Q_size_);
        }
    }

    __host__ void HEOperator::multiply_plain_bfv(Ciphertext& input1,
                                                 Plaintext& input2,
                                                 Ciphertext& output,
                                                 HEStream& stream)
    {
        if (input1.in_ntt_domain_)
        {
            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                 stream.stream>>>(
                input1.data(), input2.data(), output.data(), modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0,
                               stream.stream>>>(
                input2.data(), stream.temp1_plain_mul, modulus_->data(),
                upper_halfincrement_->data(), upper_threshold_, n_power,
                Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                             .ntt_type = FORWARD,
                                             .reduction_poly =
                                                 ReductionPolynomial::X_N_plus,
                                             .zero_padding = false,
                                             .stream = stream.stream};

            ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                              .ntt_type = INVERSE,
                                              .reduction_poly =
                                                  ReductionPolynomial::X_N_plus,
                                              .zero_padding = false,
                                              .mod_inverse = n_inverse_->data(),
                                              .stream = stream.stream};

            GPU_NTT_Inplace(stream.temp1_plain_mul, ntt_table_->data(),
                            modulus_->data(), cfg_ntt, Q_size_, Q_size_);

            GPU_NTT(input1.data(), output.data(), ntt_table_->data(),
                    modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);

            cipherplain_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                 stream.stream>>>(
                output.data(), stream.temp1_plain_mul, output.data(),
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            GPU_NTT_Inplace(output.data(), intt_table_->data(),
                            modulus_->data(), cfg_intt, 2 * Q_size_, Q_size_);
        }
    }

    __host__ void HEOperator::multiply_plain_ckks(Ciphertext& input1,
                                                  Plaintext& input2,
                                                  Ciphertext& output)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void HEOperator::multiply_plain_ckks(Ciphertext& input1,
                                                  Plaintext& input2,
                                                  Ciphertext& output,
                                                  HEStream& stream)
    {
        if (input1.depth_ != input2.depth_)
        {
            throw std::logic_error("Ciphertexts leveled are not equal");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        cipherplain_multiplication_kernel<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            input1.data(), input2.data(), output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            output.scale_ = input1.scale_ * input2.scale_;
        }
    }

    __host__ void
    HEOperator::relinearize_seal_method_inplace(Ciphertext& input1,
                                                Relinkey& relin_key)
    {
        CipherBroadcast2<<<dim3((n >> 8), Q_size_, 1), 256>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_relin, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_relin, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_seal_method_inplace(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        CipherBroadcast2<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream.stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), stream.temp1_relin,
            modulus_->data(), n_power, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_relin, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, Q_size_ * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(stream.temp1_relin, relin_key.data(),
                                           stream.temp2_relin, modulus_->data(),
                                           n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_,
                                            stream.stream);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_relin, key_location.data(), stream.temp2_relin,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_relin, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                            stream.stream>>>(
            stream.temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator::relinearize_external_product_method_inplace(Ciphertext& input1,
                                                            Relinkey& relin_key)
    {
        relin_DtoB_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin_new,
            modulus_->data(), B_prime_->data(),
            base_change_matrix_D_to_B_->data(), Mi_inv_D_to_B_->data(),
            prod_D_to_B_->data(), I_j_->data(), I_location_->data(), n_power,
            Q_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_relin_new, B_prime_ntt_tables_->data(),
                        B_prime_->data(), cfg_ntt, d * r_prime, r_prime);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_new<<<dim3((n >> 8), r_prime, d_tilda), 256>>>(
                temp1_relin_new, relin_key.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_);
            MultiplyAcc_new<<<dim3((n >> 8), r_prime, d_tilda), 256>>>(
                temp1_relin_new, key_location.data(), temp2_relin_new,
                B_prime_->data(), n_power, d_tilda, d, r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_->data(),
            .stream = 0};

        GPU_NTT_Inplace(temp2_relin_new, B_prime_intt_tables_->data(),
                        B_prime_->data(), cfg_intt, 2 * r_prime * d_tilda,
                        r_prime);

        relin_BtoD_kernelNewP<<<dim3((n >> 8), d_tilda, 2), 256>>>(
            temp2_relin_new, temp3_relin_new, modulus_->data(),
            B_prime_->data(), base_change_matrix_B_to_D_->data(),
            Mi_inv_B_to_D_->data(), prod_B_to_D_->data(), I_j_->data(),
            I_location_->data(), n_power, Q_prime_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp3_relin_new, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method_inplace(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        relin_DtoB_kernel<<<dim3((n >> 8), d, 1), 256, 0, stream.stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), stream.temp1_relin_new,
            modulus_->data(), B_prime_->data(),
            base_change_matrix_D_to_B_->data(), Mi_inv_D_to_B_->data(),
            prod_D_to_B_->data(), I_j_->data(), I_location_->data(), n_power,
            Q_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_relin_new, B_prime_ntt_tables_->data(),
                        B_prime_->data(), cfg_ntt, d * r_prime, r_prime);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_new<<<dim3((n >> 8), r_prime, d_tilda), 256, 0,
                              stream.stream>>>(
                stream.temp1_relin_new, relin_key.data(),
                stream.temp2_relin_new, B_prime_->data(), n_power, d_tilda, d,
                r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_,
                                            stream.stream);
            MultiplyAcc_new<<<dim3((n >> 8), r_prime, d_tilda), 256, 0,
                              stream.stream>>>(
                stream.temp1_relin_new, key_location.data(),
                stream.temp2_relin_new, B_prime_->data(), n_power, d_tilda, d,
                r_prime);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_relin_new, B_prime_intt_tables_->data(),
                        B_prime_->data(), cfg_intt, 2 * r_prime * d_tilda,
                        r_prime);

        relin_BtoD_kernelNewP<<<dim3((n >> 8), d_tilda, 2), 256, 0,
                                stream.stream>>>(
            stream.temp2_relin_new, stream.temp3_relin_new, modulus_->data(),
            B_prime_->data(), base_change_matrix_B_to_D_->data(),
            Mi_inv_B_to_D_->data(), prod_B_to_D_->data(), I_j_->data(),
            I_location_->data(), n_power, Q_prime_size_, d_tilda, d, r_prime);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                               stream.stream>>>(
            stream.temp3_relin_new, input1.data(), input1.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace(
        Ciphertext& input1, Relinkey& relin_key)
    {
        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            input1.data() + (Q_size_ << (n_power + 1)), temp1_relin,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_relin, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_relin, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256, 0, stream.stream>>>(
            input1.data() + (Q_size_ << (n_power + 1)), stream.temp1_relin,
            modulus_->data(), base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_relin, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, d * Q_prime_size_,
                        Q_prime_size_);

        MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                              stream.stream>>>(
            stream.temp1_relin, relin_key.data(), stream.temp2_relin,
            modulus_->data(), n_power, Q_prime_size_, d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp1_relin, relin_key.data(), stream.temp2_relin,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_,
                                            stream.stream);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp1_relin, key_location.data(), stream.temp2_relin,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_relin, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                               stream.stream>>>(
            stream.temp2_relin, input1.data(), input1.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void
    HEOperator::relinearize_seal_method_inplace_ckks(Ciphertext& input1,
                                                     Relinkey& relin_key)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(input1.data() + (current_decomp_count << (n_power + 1)),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        current_decomp_count, current_decomp_count);

        CipherBroadcast2_leveled<<<dim3((n >> 8), current_decomp_count, 1),
                                   256>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            temp1_relin, modulus_->data(), first_rns_mod_count,
            current_rns_mod_count, n_power);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = 0};

        GPU_NTT_Poly_Ordered_Inplace(
            temp2_relin, intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256>>>(
            temp2_relin, temp1_relin, modulus_->data(), half_p_->data(),
            half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp1_relin, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp1_relin, temp2_relin, input1.data(), input1.data(),
            modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_seal_method_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(input1.data() + (current_decomp_count << (n_power + 1)),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        current_decomp_count, current_decomp_count);

        CipherBroadcast2_leveled<<<dim3((n >> 8), current_decomp_count, 1), 256,
                                   0, stream.stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            stream.temp1_relin, modulus_->data(), first_rns_mod_count,
            current_rns_mod_count, n_power);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp1_relin, relin_key.data(), stream.temp2_relin,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_,
                                            stream.stream);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp1_relin, key_location.data(), stream.temp2_relin,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = stream.stream};

        GPU_NTT_Poly_Ordered_Inplace(
            stream.temp2_relin,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256, 0,
                                         stream.stream>>>(
            stream.temp2_relin, stream.temp1_relin, modulus_->data(),
            half_p_->data(), half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp1_relin, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp1_relin, stream.temp2_relin, input1.data(),
            input1.data(), modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Modulus_Ordered_Inplace(
            input1.data() + (current_decomp_count << (n_power + 1)),
            intt_table_->data(), modulus_->data(), cfg_intt,
            current_decomp_count, current_decomp_count,
            prime_location_leveled_->data() + location);

        relin_DtoB_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256>>>(
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

        ////////////////////////////////////////////////////////////////////

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_relin_new, B_prime_ntt_tables_leveled_->data(),
                        B_prime_leveled_->data(), cfg_ntt,
                        d_leveled_->operator[](input1.depth_) *
                            r_prime_leveled_,
                        r_prime_leveled_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_new<<<dim3((n >> 8), r_prime_leveled_,
                                   d_tilda_leveled_->operator[](input1.depth_)),
                              256>>>(
                temp1_relin_new, relin_key.data(input1.depth_), temp2_relin_new,
                B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                relin_key.host_location_leveled_[input1.depth_]);
            MultiplyAcc_new<<<dim3((n >> 8), r_prime_leveled_,
                                   d_tilda_leveled_->operator[](input1.depth_)),
                              256>>>(
                temp1_relin_new, key_location.data(), temp2_relin_new,
                B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_leveled_->data(),
            .stream = 0};

        GPU_NTT_Inplace(temp2_relin_new, B_prime_intt_tables_leveled_->data(),
                        B_prime_leveled_->data(), cfg_intt2,
                        2 * r_prime_leveled_ *
                            d_tilda_leveled_->operator[](input1.depth_),
                        r_prime_leveled_);

        relin_BtoD_kernelNewP_leveled2<<<
            dim3((n >> 8), d_tilda_leveled_->operator[](input1.depth_), 2),
            256>>>(temp2_relin_new, temp3_relin_new, modulus_->data(),
                   B_prime_leveled_->data(),
                   base_change_matrix_B_to_D_leveled_->operator[](input1.depth_)
                       .data(),
                   Mi_inv_B_to_D_leveled_->data(),
                   prod_B_to_D_leveled_->operator[](input1.depth_).data(),
                   I_j_leveled_->operator[](input1.depth_).data(),
                   I_location_leveled_->operator[](input1.depth_).data(),
                   n_power, current_rns_mod_count,
                   d_tilda_leveled_->operator[](input1.depth_),
                   d_leveled_->operator[](input1.depth_), r_prime_leveled_,
                   prime_location_leveled_->data() + location);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ////////////////////////////////////////////////////////////////////

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp3_relin_new, temp2_relin_new, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp2_relin_new, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp2_relin_new, input1.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Modulus_Ordered_Inplace(
            input1.data() + (current_decomp_count << (n_power + 1)),
            intt_table_->data(), modulus_->data(), cfg_intt,
            current_decomp_count, current_decomp_count,
            prime_location_leveled_->data() + location);

        relin_DtoB_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream.stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            stream.temp1_relin_new, modulus_->data(), B_prime_leveled_->data(),
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

        ////////////////////////////////////////////////////////////////////

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(
            stream.temp1_relin_new, B_prime_ntt_tables_leveled_->data(),
            B_prime_leveled_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * r_prime_leveled_,
            r_prime_leveled_);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc_new<<<dim3((n >> 8), r_prime_leveled_,
                                   d_tilda_leveled_->operator[](input1.depth_)),
                              256, 0, stream.stream>>>(
                stream.temp1_relin_new, relin_key.data(input1.depth_),
                stream.temp2_relin_new, B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                relin_key.host_location_leveled_[input1.depth_], stream.stream);
            MultiplyAcc_new<<<dim3((n >> 8), r_prime_leveled_,
                                   d_tilda_leveled_->operator[](input1.depth_)),
                              256, 0, stream.stream>>>(
                stream.temp1_relin_new, key_location.data(),
                stream.temp2_relin_new, B_prime_leveled_->data(), n_power,
                d_tilda_leveled_->operator[](input1.depth_),
                d_leveled_->operator[](input1.depth_), r_prime_leveled_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = B_prime_n_inverse_leveled_->data(),
            .stream = stream.stream};

        GPU_NTT_Inplace(
            stream.temp2_relin_new, B_prime_intt_tables_leveled_->data(),
            B_prime_leveled_->data(), cfg_intt2,
            2 * r_prime_leveled_ * d_tilda_leveled_->operator[](input1.depth_),
            r_prime_leveled_);

        relin_BtoD_kernelNewP_leveled2<<<
            dim3((n >> 8), d_tilda_leveled_->operator[](input1.depth_), 2), 256,
            0, stream.stream>>>(
            stream.temp2_relin_new, stream.temp3_relin_new, modulus_->data(),
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

        ////////////////////////////////////////////////////////////////////

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp3_relin_new, stream.temp2_relin_new, modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp2_relin_new, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256, 0,
                          stream.stream>>>(stream.temp2_relin_new,
                                           input1.data(), input1.data(),
                                           modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        GPU_NTT_Inplace(input1.data() + (current_decomp_count << (n_power + 1)),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        current_decomp_count, current_decomp_count);

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256>>>(
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

        ////////////////////////////////////////////////////////////////////

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Modulus_Ordered_Inplace(
            temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp1_relin, relin_key.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_);
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp1_relin, key_location.data(), temp2_relin, modulus_->data(),
                first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp2_relin, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp2_relin, temp1_relin, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp1_relin, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp1_relin, input1.data(), input1.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::relinearize_external_product_method2_inplace_ckks(
        Ciphertext& input1, Relinkey& relin_key, HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int j = 0; j < input1.depth_; j++)
        {
            location += counter;
            counter--;
        }

        GPU_NTT_Inplace(input1.data() + (current_decomp_count << (n_power + 1)),
                        intt_table_->data(), modulus_->data(), cfg_intt,
                        current_decomp_count, current_decomp_count);

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream.stream>>>(
            input1.data() + (current_decomp_count << (n_power + 1)),
            stream.temp1_relin, modulus_->data(),
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

        ////////////////////////////////////////////////////////////////////

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp1_relin, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (relin_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                stream.temp1_relin, relin_key.data(), stream.temp2_relin,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(relin_key.host_location_,
                                            stream.stream);
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                stream.temp1_relin, key_location.data(), stream.temp2_relin,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp2_relin, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp2_relin, stream.temp1_relin, modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp1_relin, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256, 0,
                          stream.stream>>>(stream.temp1_relin, input1.data(),
                                           input1.data(), modulus_->data(),
                                           n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::rescale_inplace_ckks_leveled(Ciphertext& input1)
    {
        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + (current_decomp_count - 1),
            .stream = 0};

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        // int counter = first_rns_mod_count - 2;
        int counter = first_decomp_count - 1;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        GPU_NTT_Poly_Ordered_Inplace(
            input1.data(),
            intt_table_->data() + ((current_decomp_count - 1) << n_power),
            modulus_->data() + (current_decomp_count - 1), cfg_intt, 2, 1,
            new_input_locations + ((input1.depth_ + P_size_) * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256>>>(
            input1.data(), temp1_rescale, modulus_->data(),
            rescaled_half_->data() + input1.depth_,
            rescaled_half_mod_->data() + location, n_power,
            current_decomp_count - 1, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp1_rescale, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * (current_decomp_count - 1),
                        (current_decomp_count - 1));

        move_cipher_ckks_leveled<<<dim3((n >> 8), current_decomp_count - 1, 2),
                                   256>>>(input1.data(), temp2_rescale, n_power,
                                          current_decomp_count - 1);

        DivideRoundLastq_rescale_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count - 1, 2), 256>>>(
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

    __host__ void HEOperator::rescale_inplace_ckks_leveled(Ciphertext& input1,
                                                           HEStream& stream)
    {
        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + (current_decomp_count - 1),
            .stream = stream.stream};

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        // int counter = first_rns_mod_count - 2;
        int counter = first_decomp_count - 1;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        GPU_NTT_Poly_Ordered_Inplace(
            input1.data(),
            intt_table_->data() + ((current_decomp_count - 1) << n_power),
            modulus_->data() + (current_decomp_count - 1), cfg_intt, 2, 1,
            new_input_locations + ((input1.depth_ + P_size_) * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256, 0,
                                         stream.stream>>>(
            input1.data(), stream.temp1_rescale, modulus_->data(),
            rescaled_half_->data() + input1.depth_,
            rescaled_half_mod_->data() + location, n_power,
            current_decomp_count - 1, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(
            stream.temp1_rescale, ntt_table_->data(), modulus_->data(), cfg_ntt,
            2 * (current_decomp_count - 1), (current_decomp_count - 1));

        move_cipher_ckks_leveled<<<dim3((n >> 8), current_decomp_count - 1, 2),
                                   256, 0, stream.stream>>>(
            input1.data(), stream.temp2_rescale, n_power,
            current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DivideRoundLastq_rescale_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count - 1, 2), 256, 0,
            stream.stream>>>(stream.temp1_rescale, stream.temp2_rescale,
                             input1.data(), modulus_->data(),
                             rescaled_last_q_modinv_->data() + location,
                             n_power, current_decomp_count - 1);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (scheme_ == scheme_type::ckks)
        {
            input1.scale_ = input1.scale_ /
                            static_cast<double>(
                                prime_vector_[current_decomp_count - 1].value);
        }

        input1.depth_++;
    }

    __host__ void HEOperator::mod_drop_ckks_leveled_inplace(Ciphertext& input1)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int offset1 = current_decomp_count << n_power;
        int offset2 = (current_decomp_count - 1) << n_power;

        // TODO: do with efficient way!
        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256>>>(input1.data() + offset1, temp_mod_drop,
                                       n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256>>>(temp_mod_drop, input1.data() + offset2,
                                       n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input1.depth_++;
    }

    __host__ void HEOperator::mod_drop_ckks_leveled_inplace(Ciphertext& input1,
                                                            HEStream& stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        int offset1 = current_decomp_count << n_power;
        int offset2 = (current_decomp_count - 1) << n_power;

        // TODO: do with efficient way!
        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256, 0, stream.stream>>>(
            input1.data() + offset1, temp_mod_drop, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256, 0, stream.stream>>>(
            temp_mod_drop, input1.data() + offset2, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input1.depth_++;
    }

    __host__ void HEOperator::mod_drop_ckks_leveled(Ciphertext& input1,
                                                    Ciphertext& input2)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        global_memory_replace_2<<<dim3((n >> 8), current_decomp_count - 1, 2),
                                  256>>>(input1.data(), input2.data(),
                                         current_decomp_count, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::mod_drop_ckks_leveled(Ciphertext& input1,
                                                    Ciphertext& input2,
                                                    HEStream& stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Ciphertext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        global_memory_replace_2<<<dim3((n >> 8), current_decomp_count - 1, 2),
                                  256, 0, stream.stream>>>(
            input1.data(), input2.data(), current_decomp_count, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input2.depth_ = input1.depth_ + 1;
    }

    __host__ void HEOperator::mod_drop_ckks_plaintext(Plaintext& input1,
                                                      Plaintext& input2)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256>>>(input1.data(), input2.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        input2.depth_ = input1.depth_ + 1;
    }

    __host__ void HEOperator::mod_drop_ckks_plaintext(Plaintext& input1,
                                                      Plaintext& input2,
                                                      HEStream& stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        int current_decomp_count = Q_size_ - input1.depth_;

        global_memory_replace<<<dim3((n >> 8), current_decomp_count - 1, 1),
                                256, 0, stream.stream>>>(
            input1.data(), input2.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::mod_drop_ckks_plaintext_inplace(Plaintext& input1)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        input1.depth_++;
    }

    __host__ void HEOperator::mod_drop_ckks_plaintext_inplace(Plaintext& input1,
                                                              HEStream& stream)
    {
        if (input1.depth_ >= (Q_size_ - 1))
        {
            throw std::logic_error("Plaintext modulus can not be dropped!");
        }

        input1.depth_++;
    }

    __host__ void HEOperator::rotate_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key, int shift)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
        if (key_exist)
        {
            apply_galois_method_I(input1, output, galois_key, galoiselt);
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
                apply_galois_method_I(in_data, output, galois_key, galois_elt);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key, int shift,
                                              HEStream& stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
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

    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////

    __host__ void HEOperator::rotate_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key, int shift)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
        if (key_exist)
        {
            apply_galois_method_II(input1, output, galois_key, galoiselt);
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
                apply_galois_method_II(in_data, output, galois_key, galois_elt);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key, int shift,
                                               HEStream& stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
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
                                                   int shift)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
        if (key_exist)
        {
            apply_galois_ckks_method_I(input1, output, galois_key, galoiselt);
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
                                           galois_elt);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_ckks_method_I(Ciphertext& input1,
                                                   Ciphertext& output,
                                                   Galoiskey& galois_key,
                                                   int shift, HEStream& stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
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
                                                    int shift)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
        if (key_exist)
        {
            apply_galois_ckks_method_II(input1, output, galois_key, galoiselt);
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
                                            galois_elt);
                in_data = output;
            }
        }
    }

    __host__ void HEOperator::rotate_ckks_method_II(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    Galoiskey& galois_key,
                                                    int shift, HEStream& stream)
    {
        int galoiselt = steps_to_galois_elt(shift, n);
        bool key_exist = galois_key.store_in_gpu_
                             ? (galois_key.device_location_.find(galoiselt) !=
                                galois_key.device_location_.end())
                             : (galois_key.host_location_.find(galoiselt) !=
                                galois_key.host_location_.end());
        // if ((galois_key.device_location_.find(galoiselt) !=
        // galois_key.device_location_.end()))
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
                                                    int galois_elt)
    {
        apply_galois_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            galois_elt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, galois_key.device_location_[galois_elt].data(),
                temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt]);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_method_I(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    Galoiskey& galois_key,
                                                    int galois_elt,
                                                    HEStream& stream)
    {
        apply_galois_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                              stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), galois_elt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, Q_size_ * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation,
                galois_key.device_location_[galois_elt].data(),
                stream.temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt], stream.stream);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation, key_location.data(),
                stream.temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                            stream.stream>>>(
            stream.temp2_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_method_II(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     Galoiskey& galois_key,
                                                     int galois_elt)
    {
        apply_galois_method_II_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            galois_elt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            temp1_rotation, temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt]);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp3_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp3_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_method_II(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     Galoiskey& galois_key,
                                                     int galois_elt,
                                                     HEStream& stream)
    {
        apply_galois_method_II_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                        stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), galois_elt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256, 0, stream.stream>>>(
            stream.temp1_rotation, stream.temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, d * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation,
                galois_key.device_location_[galois_elt].data(),
                stream.temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt], stream.stream);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation, key_location.data(),
                stream.temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp3_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                               stream.stream>>>(
            stream.temp3_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_ckks_method_I(Ciphertext& input1,
                                                         Ciphertext& output,
                                                         Galoiskey& galois_key,
                                                         int galois_elt)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        apply_galois_ckks_kernel<<<dim3((n >> 8), current_decomp_count, 2),
                                   256>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            galois_elt, n_power, first_rns_mod_count, current_rns_mod_count,
            current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(
                temp2_rotation, galois_key.device_location_[galois_elt].data(),
                temp3_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt]);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(temp2_rotation, key_location.data(),
                                          temp3_rotation, modulus_->data(),
                                          first_rns_mod_count,
                                          current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = 0};

        GPU_NTT_Poly_Ordered_Inplace(
            temp3_rotation,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256>>>(
            temp3_rotation, temp2_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, current_decomp_count, current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp2_rotation, temp3_rotation, temp1_rotation, output.data(),
            modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_ckks_method_I(Ciphertext& input1,
                                                         Ciphertext& output,
                                                         Galoiskey& galois_key,
                                                         int galois_elt,
                                                         HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT(input1.data(), stream.temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        apply_galois_ckks_kernel<<<dim3((n >> 8), current_decomp_count, 2), 256,
                                   0, stream.stream>>>(
            stream.temp0_rotation, stream.temp1_rotation, stream.temp2_rotation,
            modulus_->data(), galois_elt, n_power, first_rns_mod_count,
            current_rns_mod_count, current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp2_rotation, ntt_table_->data(), modulus_->data(),
            cfg_ntt, current_decomp_count * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp2_rotation,
                galois_key.device_location_[galois_elt].data(),
                stream.temp3_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt], stream.stream);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp2_rotation, key_location.data(),
                stream.temp3_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = stream.stream};

        GPU_NTT_Poly_Ordered_Inplace(
            stream.temp3_rotation,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256, 0,
                                         stream.stream>>>(
            stream.temp3_rotation, stream.temp2_rotation, modulus_->data(),
            half_p_->data(), half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp2_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, current_decomp_count,
                        current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp2_rotation, stream.temp3_rotation, stream.temp1_rotation,
            output.data(), modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_ckks_method_II(Ciphertext& input1,
                                                          Ciphertext& output,
                                                          Galoiskey& galois_key,
                                                          int galois_elt)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        apply_galois_method_II_kernel<<<dim3((n >> 8), current_decomp_count, 2),
                                        256>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            galois_elt, n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256>>>(
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

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp3_rotation, galois_key.device_location_[galois_elt].data(),
                temp4_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, current_rns_mod_count,
                d_leveled_->operator[](input1.depth_), input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt]);
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp4_rotation, temp3_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp3_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, current_decomp_count, current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp3_rotation, temp1_rotation, output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::apply_galois_ckks_method_II(Ciphertext& input1,
                                                          Ciphertext& output,
                                                          Galoiskey& galois_key,
                                                          int galois_elt,
                                                          HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT(input1.data(), stream.temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        apply_galois_method_II_kernel<<<dim3((n >> 8), current_decomp_count, 2),
                                        256, 0, stream.stream>>>(
            stream.temp0_rotation, stream.temp1_rotation, stream.temp2_rotation,
            modulus_->data(), galois_elt, n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream.stream>>>(
            stream.temp2_rotation, stream.temp3_rotation, modulus_->data(),
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

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp3_rotation, ntt_table_->data(), modulus_->data(),
            cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                stream.temp3_rotation,
                galois_key.device_location_[galois_elt].data(),
                stream.temp4_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, current_rns_mod_count,
                d_leveled_->operator[](input1.depth_), input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(
                galois_key.host_location_[galois_elt], stream.stream);
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                stream.temp3_rotation, key_location.data(),
                stream.temp4_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, current_rns_mod_count,
                d_leveled_->operator[](input1.depth_), input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp4_rotation, intt_table_->data(), modulus_->data(),
            cfg_intt, 2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp4_rotation, stream.temp3_rotation, modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp3_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, current_decomp_count,
                        current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256, 0,
                          stream.stream>>>(stream.temp3_rotation,
                                           stream.temp1_rotation, output.data(),
                                           modulus_->data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::rotate_columns_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Galoiskey& galois_key)
    {
        int galoiselt = galois_key.galois_elt_zero;

        apply_galois_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            galoiselt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, galois_key.c_data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(galois_key.zero_host_location_);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::rotate_columns_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Galoiskey& galois_key,
                                                      HEStream& stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        apply_galois_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                              stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), galoiselt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, Q_size_ * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation, galois_key.c_data(),
                stream.temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(galois_key.zero_host_location_,
                                            stream.stream);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation, key_location.data(),
                stream.temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                            stream.stream>>>(
            stream.temp2_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::rotate_columns_method_II(Ciphertext& input1,
                                                       Ciphertext& output,
                                                       Galoiskey& galois_key)
    {
        int galoiselt = galois_key.galois_elt_zero;

        apply_galois_method_II_kernel<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            galoiselt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            temp1_rotation, temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, galois_key.c_data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(galois_key.zero_host_location_);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp3_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp3_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::rotate_columns_method_II(Ciphertext& input1,
                                                       Ciphertext& output,
                                                       Galoiskey& galois_key,
                                                       HEStream& stream)
    {
        int galoiselt = galois_key.galois_elt_zero;

        apply_galois_method_II_kernel<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                        stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), galoiselt, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256, 0, stream.stream>>>(
            stream.temp1_rotation, stream.temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, d * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (galois_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation, galois_key.c_data(),
                stream.temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(galois_key.zero_host_location_,
                                            stream.stream);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation, key_location.data(),
                stream.temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp3_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                               stream.stream>>>(
            stream.temp3_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Switchkey& switch_key)
    {
        CipherBroadcast_switchkey_bfv_method_I<<<dim3((n >> 8), Q_size_, 2),
                                                 256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, Q_size_ * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, switch_key.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp1_rotation, key_location.data(), temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp2_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Switchkey& switch_key,
                                                 HEStream& stream)
    {
        CipherBroadcast_switchkey_bfv_method_I<<<dim3((n >> 8), Q_size_, 2),
                                                 256, 0, stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, Q_size_ * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation, switch_key.data(), stream.temp2_rotation,
                modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_,
                                            stream.stream);
            MultiplyAcc<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                          stream.stream>>>(
                stream.temp1_rotation, key_location.data(),
                stream.temp2_rotation, modulus_->data(), n_power, Q_size_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastq_<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                            stream.stream>>>(
            stream.temp2_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Switchkey& switch_key)
    {
        CipherBroadcast_switchkey_bfv_method_II<<<dim3((n >> 8), Q_size_, 2),
                                                  256>>>(
            input1.data(), temp0_rotation, temp1_rotation, modulus_->data(),
            n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            temp1_rotation, temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, d * Q_prime_size_, Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, switch_key.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256>>>(
                temp2_rotation, key_location.data(), temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT_Inplace(temp3_rotation, intt_table_->data(), modulus_->data(),
                        cfg_intt, 2 * Q_prime_size_, Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp3_rotation, temp0_rotation, output.data(), modulus_->data(),
            half_p_->data(), half_mod_->data(), last_q_modinv_->data(), n_power,
            Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Switchkey& switch_key,
                                                  HEStream& stream)
    {
        CipherBroadcast_switchkey_bfv_method_II<<<dim3((n >> 8), Q_size_, 2),
                                                  256, 0, stream.stream>>>(
            input1.data(), stream.temp0_rotation, stream.temp1_rotation,
            modulus_->data(), n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        relin_DtoQtilde_kernel<<<dim3((n >> 8), d, 1), 256>>>(
            stream.temp1_rotation, stream.temp2_rotation, modulus_->data(),
            base_change_matrix_D_to_Q_tilda_->data(),
            Mi_inv_D_to_Q_tilda_->data(), prod_D_to_Q_tilda_->data(),
            I_j_->data(), I_location_->data(), n_power, Q_size_, Q_prime_size_,
            d);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp2_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, d * Q_prime_size_,
                        Q_prime_size_);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation, switch_key.data(), stream.temp3_rotation,
                modulus_->data(), n_power, Q_prime_size_, d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_,
                                            stream.stream);
            MultiplyAcc_method2<<<dim3((n >> 8), Q_prime_size_, 1), 256, 0,
                                  stream.stream>>>(
                stream.temp2_rotation, key_location.data(),
                stream.temp3_rotation, modulus_->data(), n_power, Q_prime_size_,
                d);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT_Inplace(stream.temp3_rotation, intt_table_->data(),
                        modulus_->data(), cfg_intt, 2 * Q_prime_size_,
                        Q_prime_size_);

        DivideRoundLastqNewP<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                               stream.stream>>>(
            stream.temp3_rotation, stream.temp0_rotation, output.data(),
            modulus_->data(), half_p_->data(), half_mod_->data(),
            last_q_modinv_->data(), n_power, Q_prime_size_, Q_size_, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_ckks_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Switchkey& switch_key)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        CipherBroadcast_switchkey_ckks_method_I<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            n_power, first_rns_mod_count, current_rns_mod_count,
            current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            temp2_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            current_decomp_count * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(temp2_rotation, switch_key.data(),
                                          temp3_rotation, modulus_->data(),
                                          first_rns_mod_count,
                                          current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256>>>(temp2_rotation, key_location.data(),
                                          temp3_rotation, modulus_->data(),
                                          first_rns_mod_count,
                                          current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = 0};

        GPU_NTT_Poly_Ordered_Inplace(
            temp3_rotation,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256>>>(
            temp3_rotation, temp2_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp2_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, current_decomp_count, current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp2_rotation, temp3_rotation, temp1_rotation, output.data(),
            modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_ckks_method_I(Ciphertext& input1,
                                                      Ciphertext& output,
                                                      Switchkey& switch_key,
                                                      HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT(input1.data(), stream.temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        CipherBroadcast_switchkey_ckks_method_I<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp0_rotation, stream.temp1_rotation, stream.temp2_rotation,
            modulus_->data(), n_power, first_rns_mod_count,
            current_rns_mod_count, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }
        GPU_NTT_Modulus_Ordered_Inplace(
            stream.temp2_rotation, ntt_table_->data(), modulus_->data(),
            cfg_ntt, current_decomp_count * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp2_rotation, switch_key.data(), stream.temp3_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_,
                                            stream.stream);
            MultiplyAcc2_leveled<<<dim3((n >> 8), current_rns_mod_count, 1),
                                   256, 0, stream.stream>>>(
                stream.temp2_rotation, key_location.data(),
                stream.temp3_rotation, modulus_->data(), first_rns_mod_count,
                current_decomp_count, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ntt_rns_configuration cfg_intt2 = {
            .n_power = n_power,
            .ntt_type = INVERSE,
            .reduction_poly = ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data() + first_decomp_count,
            .stream = stream.stream};

        GPU_NTT_Poly_Ordered_Inplace(
            stream.temp3_rotation,
            intt_table_->data() + (first_decomp_count << n_power),
            modulus_->data() + first_decomp_count, cfg_intt2, 2, 1,
            new_input_locations + (input1.depth_ * 2));

        DivideRoundLastq_ckks1_leveled<<<dim3((n >> 8), 2, 1), 256, 0,
                                         stream.stream>>>(
            stream.temp3_rotation, stream.temp2_rotation, modulus_->data(),
            half_p_->data(), half_mod_->data(), n_power, first_decomp_count,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(stream.temp2_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(stream.temp1_rotation, ntt_table_->data(),
                        modulus_->data(), cfg_ntt, current_decomp_count,
                        current_decomp_count);

        DivideRoundLastq_ckks2_leveled<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            stream.temp2_rotation, stream.temp3_rotation, stream.temp1_rotation,
            output.data(), modulus_->data(), last_q_modinv_->data(), n_power,
            current_decomp_count);

        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_ckks_method_II(Ciphertext& input1,
                                                       Ciphertext& output,
                                                       Switchkey& switch_key)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        CipherBroadcast_switchkey_bfv_method_II<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256>>>(
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

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp3_rotation, switch_key.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_);
            MultiplyAcc2_leveled_method2<<<
                dim3((n >> 8), current_rns_mod_count, 1), 256>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp4_rotation, temp3_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp3_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, current_decomp_count, current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256>>>(
            temp3_rotation, temp1_rotation, output.data(), modulus_->data(),
            n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::switchkey_ckks_method_II(Ciphertext& input1,
                                                       Ciphertext& output,
                                                       Switchkey& switch_key,
                                                       HEStream& stream)
    {
        int first_rns_mod_count = Q_prime_size_;
        int current_rns_mod_count = Q_prime_size_ - input1.depth_;

        int first_decomp_count = Q_size_;
        int current_decomp_count = Q_size_ - input1.depth_;

        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT(input1.data(), temp0_rotation, intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * current_decomp_count,
                current_decomp_count);

        CipherBroadcast_switchkey_bfv_method_II<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            temp0_rotation, temp1_rotation, temp2_rotation, modulus_->data(),
            n_power, current_decomp_count);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        int counter = first_rns_mod_count;
        int location = 0;
        for (int i = 0; i < input1.depth_; i++)
        {
            location += counter;
            counter--;
        }

        relin_DtoQtilda_kernel_leveled2<<<
            dim3((n >> 8), d_leveled_->operator[](input1.depth_), 1), 256, 0,
            stream.stream>>>(
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

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp3_rotation, ntt_table_->data(), modulus_->data(), cfg_ntt,
            d_leveled_->operator[](input1.depth_) * current_rns_mod_count,
            current_rns_mod_count, new_prime_locations + location);

        // TODO: make it efficient
        if (switch_key.store_in_gpu_)
        {
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                temp3_rotation, switch_key.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            DeviceVector<Data> key_location(switch_key.host_location_,
                                            stream.stream);
            MultiplyAcc2_leveled_method2<<<dim3((n >> 8), current_rns_mod_count,
                                                1),
                                           256, 0, stream.stream>>>(
                temp3_rotation, key_location.data(), temp4_rotation,
                modulus_->data(), first_rns_mod_count, current_decomp_count,
                current_rns_mod_count, d_leveled_->operator[](input1.depth_),
                input1.depth_, n_power);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        ////////////////////////////////////////////////////////////////////

        GPU_NTT_Modulus_Ordered_Inplace(
            temp4_rotation, intt_table_->data(), modulus_->data(), cfg_intt,
            2 * current_rns_mod_count, current_rns_mod_count,
            new_prime_locations + location);

        DivideRoundLastqNewP_external_ckks<<<
            dim3((n >> 8), current_decomp_count, 2), 256, 0, stream.stream>>>(
            temp4_rotation, temp3_rotation, modulus_->data(), half_p_->data(),
            half_mod_->data(), last_q_modinv_->data(), n_power,
            current_rns_mod_count, current_decomp_count, first_rns_mod_count,
            first_decomp_count, P_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        GPU_NTT_Inplace(temp3_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, 2 * current_decomp_count,
                        current_decomp_count);

        // TODO: Merge with previous one
        GPU_NTT_Inplace(temp1_rotation, ntt_table_->data(), modulus_->data(),
                        cfg_ntt, current_decomp_count, current_decomp_count);

        cipher_temp_add<<<dim3((n >> 8), current_decomp_count, 2), 256, 0,
                          stream.stream>>>(temp3_rotation, temp1_rotation,
                                           output.data(), modulus_->data(),
                                           n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                                             Ciphertext& output,
                                                             int index)
    {
        DeviceVector<Data> temp(2 * n * Q_size_);

        negacyclic_shift_poly_coeffmod_kernel<<<dim3((n >> 8), Q_size_, 2),
                                                256>>>(
            input1.data(), temp.data(), modulus_->data(), index, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: do with efficient way!
        global_memory_replace<<<dim3((n >> 8), Q_size_, 2), 256>>>(
            temp.data(), output.data(), n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                                             Ciphertext& output,
                                                             int index,
                                                             HEStream& stream)
    {
        DeviceVector<Data> temp(2 * n * Q_size_, stream.stream);

        negacyclic_shift_poly_coeffmod_kernel<<<dim3((n >> 8), Q_size_, 2), 256,
                                                0, stream.stream>>>(
            input1.data(), temp.data(), modulus_->data(), index, n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        // TODO: do with efficient way!
        global_memory_replace<<<dim3((n >> 8), Q_size_, 2), 256, 0,
                                stream.stream>>>(temp.data(), output.data(),
                                                 n_power);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_to_ntt_bfv_plain(Plaintext& input1,
                                                         Plaintext& output)
    {
        threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256>>>(
            input1.data(), temp1_plain_mul, modulus_->data(),
            upper_halfincrement_->data(), upper_threshold_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT(temp1_plain_mul, output.data(), ntt_table_->data(),
                modulus_->data(), cfg_ntt, Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_to_ntt_bfv_plain(Plaintext& input1,
                                                         Plaintext& output,
                                                         HEStream& stream)
    {
        threshold_kernel<<<dim3((n >> 8), Q_size_, 1), 256, 0, stream.stream>>>(
            input1.data(), temp1_plain_mul, modulus_->data(),
            upper_halfincrement_->data(), upper_threshold_, n_power, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT(temp1_plain_mul, output.data(), ntt_table_->data(),
                modulus_->data(), cfg_ntt, Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_to_ntt_bfv_cipher(Ciphertext& input1,
                                                          Ciphertext& output)
    {
        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = 0};

        GPU_NTT(input1.data(), output.data(), ntt_table_->data(),
                modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_to_ntt_bfv_cipher(Ciphertext& input1,
                                                          Ciphertext& output,
                                                          HEStream& stream)
    {
        ntt_rns_configuration cfg_ntt = {.n_power = n_power,
                                         .ntt_type = FORWARD,
                                         .reduction_poly =
                                             ReductionPolynomial::X_N_plus,
                                         .zero_padding = false,
                                         .stream = stream.stream};

        GPU_NTT(input1.data(), output.data(), ntt_table_->data(),
                modulus_->data(), cfg_ntt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_from_ntt_bfv_cipher(Ciphertext& input1,
                                                            Ciphertext& output)
    {
        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = 0};

        GPU_NTT(input1.data(), output.data(), intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HEOperator::transform_from_ntt_bfv_cipher(Ciphertext& input1,
                                                            Ciphertext& output,
                                                            HEStream& stream)
    {
        ntt_rns_configuration cfg_intt = {.n_power = n_power,
                                          .ntt_type = INVERSE,
                                          .reduction_poly =
                                              ReductionPolynomial::X_N_plus,
                                          .zero_padding = false,
                                          .mod_inverse = n_inverse_->data(),
                                          .stream = stream.stream};

        GPU_NTT(input1.data(), output.data(), intt_table_->data(),
                modulus_->data(), cfg_intt, 2 * Q_size_, Q_size_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

} // namespace heongpu
