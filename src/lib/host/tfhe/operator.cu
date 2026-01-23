// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/tfhe/operator.cuh>

namespace heongpu
{
    HELogicOperator<Scheme::TFHE>::HELogicOperator(
        HEContext<Scheme::TFHE>& context)
    {
        prime_ = context.prime_;
        ntt_table_ = context.ntt_table_;
        intt_table_ = context.intt_table_;
        n_inverse_ = context.n_inverse_;

        ks_base_bit_ = context.ks_base_bit_;
        ks_length_ = context.ks_length_;

        ks_stdev_ = context.ks_stdev_;
        bk_stdev_ = context.bk_stdev_;
        max_stdev_ = context.max_stdev_;

        // LWE Context
        n_ = context.n_;

        // TLWE Context
        N_ = context.N_;
        Npower_ = int(log2l(N_));
        k_ = context.k_;

        // TGSW Context
        bk_l_ = context.bk_l_;
        bk_bg_bit_ = context.bk_bg_bit_;

        bk_half_ = context.half_bg_;
        bk_mask_ = context.mask_mod_;
        bk_offset_ = context.offset_;

        encode_mu = encode_to_torus32(1, 8);
    }

    __host__ void HELogicOperator<Scheme::TFHE>::NAND_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_nand = encode_to_torus32(1, 8);

        int shape_ = input1.shape_;

        tfhe_nand_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_nand, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::AND_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_and = encode_to_torus32(1, 8);
        encode_and = -encode_and;

        int shape_ = input1.shape_;

        tfhe_and_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_and, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::AND_N_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_and = encode_to_torus32(1, 8);
        encode_and = -encode_and;

        int shape_ = input1.shape_;

        tfhe_and_first_not_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_and, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::NOR_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_nor = encode_to_torus32(1, 8);
        encode_nor = -encode_nor;

        int shape_ = input1.shape_;

        tfhe_nor_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_nor, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::OR_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_or = encode_to_torus32(1, 8);

        int shape_ = input1.shape_;

        tfhe_or_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_or, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::XNOR_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_xnor = encode_to_torus32(1, 4);
        encode_xnor = -encode_xnor;

        int shape_ = input1.shape_;

        tfhe_xnor_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_xnor, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::XOR_pre_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& input2,
        Ciphertext<Scheme::TFHE>& output, cudaStream_t stream)
    {
        int32_t encode_xor = encode_to_torus32(1, 4);

        int shape_ = input1.shape_;

        tfhe_xor_pre_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input2.a_device_location_.data(), input2.b_device_location_.data(),
            encode_xor, input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = output.variances_[i] + input1.variances_[i] +
                                   input2.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::NOT_computation(
        Ciphertext<Scheme::TFHE>& input1, Ciphertext<Scheme::TFHE>& output,
        cudaStream_t stream)
    {
        int shape_ = input1.shape_;

        tfhe_not_comp_kernel<<<shape_, 512, 0, stream>>>(
            output.a_device_location_.data(), output.b_device_location_.data(),
            input1.a_device_location_.data(), input1.b_device_location_.data(),
            input1.n_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] = input1.variances_[i];
        }
    }

    __host__ void HELogicOperator<Scheme::TFHE>::bootstrapping(
        Ciphertext<Scheme::TFHE>& input, Ciphertext<Scheme::TFHE>& output,
        Bootstrappingkey<Scheme::TFHE>& boot_key, cudaStream_t stream)
    {
        int shape_ = input.shape_;

        Data64 total_temp_boot_size = (Data64) shape_ * (Data64) (k_ + 1) *
                                      (Data64) (bk_l_ + 1) * (Data64) (k_ + 1) *
                                      (Data64) N_;
        DeviceVector<Data64> temp_boot(total_temp_boot_size, stream);

        Data64 total_temp_boot_size2 =
            (Data64) shape_ * (Data64) (k_ + 1) * (Data64) N_;
        DeviceVector<int32_t> temp_boot2(total_temp_boot_size2, stream);

        tfhe_bootstrapping_kernel_unique_step1<<<dim3(shape_, (k_ + 1), bk_l_),
                                                 512, 0, stream>>>(
            input.a_device_location_.data(), input.b_device_location_.data(),
            temp_boot.data(), boot_key.boot_key_device_location_.data(),
            ntt_table_->data(), prime_, encode_mu, bk_offset_, bk_mask_,
            bk_half_, n_, N_, Npower_, k_, bk_bg_bit_, bk_l_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        tfhe_bootstrapping_kernel_unique_step2<<<dim3(shape_, (k_ + 1)), 512, 0,
                                                 stream>>>(
            temp_boot.data(), input.b_device_location_.data(),
            temp_boot2.data(), intt_table_->data(), n_inverse_, prime_,
            encode_mu, n_, N_, Npower_, k_, bk_l_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (int i = 1; i < n_; i++)
        {
            tfhe_bootstrapping_kernel_regular_step1<<<
                dim3(shape_, (k_ + 1), bk_l_), 512, 0, stream>>>(
                input.a_device_location_.data(),
                input.b_device_location_.data(), temp_boot2.data(),
                temp_boot.data(), boot_key.boot_key_device_location_.data(), i,
                ntt_table_->data(), prime_, bk_offset_, bk_mask_, bk_half_, n_,
                N_, Npower_, k_, bk_bg_bit_, bk_l_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            tfhe_bootstrapping_kernel_regular_step2<<<dim3(shape_, (k_ + 1)),
                                                      512, 0, stream>>>(
                temp_boot.data(), temp_boot2.data(), intt_table_->data(),
                n_inverse_, prime_, n_, N_, k_, bk_l_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] =
                output.variances_[i] + (n_ * input.variances_[i]);
        }

        tfhe_sample_extraction_kernel<<<dim3(shape_, k_), 512, 0, stream>>>(
            temp_boot2.data(), output.a_device_location_.data(),
            output.b_device_location_.data(), N_, k_, 0);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ void HELogicOperator<Scheme::TFHE>::key_switching(
        Ciphertext<Scheme::TFHE>& input, Ciphertext<Scheme::TFHE>& output,
        Bootstrappingkey<Scheme::TFHE>& boot_key, cudaStream_t stream)
    {
        int shape_ = input.shape_;

        tfhe_key_switching_kernel<<<shape_, 512, 0, stream>>>(
            input.a_device_location_.data(), input.b_device_location_.data(),
            output.a_device_location_.data(), output.b_device_location_.data(),
            boot_key.switch_key_device_location_a_.data(),
            boot_key.switch_key_device_location_b_.data(),
            boot_key.ks_base_bit_, boot_key.ks_length_, n_, N_, k_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        for (size_t i = 0; i < shape_; i++)
        {
            output.variances_[i] =
                output.variances_[i] +
                ((N_ * ks_length_ * ((1 << ks_base_bit_) - 1)) *
                 boot_key.switch_key_variances_[0]);
        }
    }

    __host__ Ciphertext<Scheme::TFHE>
    HELogicOperator<Scheme::TFHE>::generate_empty_ciphertext(
        int n, int shape, cudaStream_t stream)
    {
        Ciphertext<Scheme::TFHE> cipher;

        cipher.n_ = n;
        cipher.alpha_min_ = ks_stdev_;
        cipher.alpha_max_ = max_stdev_;

        cipher.shape_ = shape;

        cipher.a_device_location_.resize(cipher.n_ * cipher.shape_, stream);
        cipher.b_device_location_.resize(cipher.shape_, stream);

        cipher.variances_.resize(cipher.shape_);
        std::fill(cipher.variances_.begin(), cipher.variances_.end(), 0.0);

        return cipher;
    }

    __host__ int32_t HELogicOperator<Scheme::TFHE>::encode_to_torus32(
        uint32_t mu, uint32_t m_size)
    {
        uint64_t interval = ((1ULL << 63) / m_size) * 2;
        uint64_t phase64 = mu * interval;
        return static_cast<int32_t>(phase64 >> 32);
    }

} // namespace heongpu
