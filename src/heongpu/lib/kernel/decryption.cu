// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "decryption.cuh"

namespace heongpu
{
    __global__ void sk_multiplication(Data64* ct1, Data64* sk, Data64* output,
                                      Modulus64* modulus, int n_power,
                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_1 = ct1[index];
        Data64 sk_ = sk[index];

        output[index] = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);
    }

    __global__ void sk_multiplicationx3(Data64* ct1, Data64* sk,
                                        Modulus64* modulus, int n_power,
                                        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_1 = ct1[index];
        Data64 sk_ = sk[index];
        ct1[index] = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);

        Data64 ct_2 = ct1[index + (decomp_mod_count << n_power)];
        Data64 sk2_ = OPERATOR_GPU_64::mult(sk_, sk_, modulus[block_y]);
        ct1[index + (decomp_mod_count << n_power)] =
            OPERATOR_GPU_64::mult(ct_2, sk2_, modulus[block_y]);
    }

    __global__ void decryption_kernel(Data64* ct0, Data64* ct1, Data64* plain,
                                      Modulus64* modulus, Modulus64 plain_mod,
                                      Modulus64 gamma, Data64* Qi_t,
                                      Data64* Qi_gamma, Data64* Qi_inverse,
                                      Data64 mulq_inv_t, Data64 mulq_inv_gamma,
                                      Data64 inv_gamma, int n_power,
                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt =
                OPERATOR_GPU_64::add(ct0[location], ct1[location], modulus[i]);

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void decryption_kernelx3(Data64* ct0, Data64* ct1, Data64* ct2,
                                        Data64* plain, Modulus64* modulus,
                                        Modulus64 plain_mod, Modulus64 gamma,
                                        Data64* Qi_t, Data64* Qi_gamma,
                                        Data64* Qi_inverse, Data64 mulq_inv_t,
                                        Data64 mulq_inv_gamma, Data64 inv_gamma,
                                        int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt =
                OPERATOR_GPU_64::add(ct0[location], ct1[location], modulus[i]);

            mt = OPERATOR_GPU_64::add(mt, ct2[location], modulus[i]);

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void coeff_multadd(Data64* input1, Data64* input2,
                                  Data64* output, Modulus64 plain_mod,
                                  Modulus64* modulus, int n_power,
                                  int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_0 = input1[index];
        Data64 ct_1 = input2[index];

        ct_0 = OPERATOR_GPU_64::add(ct_1, ct_0, modulus[block_y]);
        ct_0 = OPERATOR_GPU_64::mult(ct_0, plain_mod.value, modulus[block_y]);

        output[index] = ct_0;
    }

    __global__ void compose_kernel(Data64* input, Data64* output,
                                   Modulus64* modulus, Data64* Mi_inv,
                                   Data64* Mi, Data64* decryption_modulus,
                                   int coeff_modulus_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 compose_result[50]; // TODO: Define size as global variable
        Data64 big_integer_result[50]; // TODO: Define size as global variable

        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data64 base = input[idx + (i << n_power)];
            Data64 temp = OPERATOR_GPU_64::mult(base, Mi_inv[i], modulus[i]);

            biginteger::multiply(Mi + (i * coeff_modulus_count),
                                 coeff_modulus_count, temp, big_integer_result,
                                 coeff_modulus_count);

            int carry = biginteger::add_inplace(
                compose_result, big_integer_result, coeff_modulus_count);

            bool check = biginteger::is_greater_or_equal(
                compose_result, decryption_modulus, coeff_modulus_count);

            if (check)
            {
                biginteger::sub2(compose_result, decryption_modulus,
                                 coeff_modulus_count, compose_result);
            }
        }

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            output[idx + (i << n_power)] = compose_result[i];
        }
    }

    __global__ void find_max_norm_kernel(Data64* input, Data64* output,
                                         Data64* upper_half_threshold,
                                         Data64* decryption_modulus,
                                         int coeff_modulus_count, int n_power)
    {
        int idx = threadIdx.x;
        int offset = blockDim.x;
        int iteration_count = (1 << n_power) / offset;

        Data64 big_integer_input[50]; // TODO: Define size as global variable
        Data64 big_integer_temp[50]; // TODO: Define size as global variable

        // Data64 *big_integer_input = (Data64 *)alloca(20 * sizeof(Data64));
        // Data64 *big_integer_temp = (Data64 *)alloca(20 * sizeof(Data64));

        biginteger::set_zero(big_integer_temp, coeff_modulus_count);

        for (int i = 0; i < iteration_count; i++)
        {
            for (int j = 0; j < coeff_modulus_count; j++)
            {
                big_integer_input[j] =
                    input[idx + (i * offset) + (j << n_power)];
            }

            bool check = biginteger::is_greater_or_equal(
                big_integer_input, upper_half_threshold, coeff_modulus_count);

            if (check)
            {
                biginteger::sub2(decryption_modulus, big_integer_input,
                                 coeff_modulus_count, big_integer_input);
            }

            check = biginteger::is_greater(big_integer_input, big_integer_temp,
                                           coeff_modulus_count);

            if (check)
            {
                biginteger::set(big_integer_input, coeff_modulus_count,
                                big_integer_temp);
            }

            __syncthreads();
        }

        extern __shared__ Data64 shared_memory[]; // 1024 64-bit

        __syncthreads();

        int offset_in = blockDim.x;
        for (int outer = 0; outer < 5; outer++)
        { // since blocksize is 512?

            if (idx < offset_in)
            {
                offset_in = offset_in >> 1;
                for (int i = 0; i < coeff_modulus_count; i++)
                {
                    if (idx >= offset_in)
                    {
                        shared_memory[idx % offset_in] = big_integer_temp[i];
                    }

                    __syncthreads();

                    if (idx < offset_in)
                    {
                        big_integer_input[i] = shared_memory[idx];
                    }

                    __syncthreads();
                }

                bool check = biginteger::is_greater(
                    big_integer_input, big_integer_temp, coeff_modulus_count);

                if (check)
                {
                    biginteger::set(big_integer_input, coeff_modulus_count,
                                    big_integer_temp);
                }

                __syncthreads();
            }
        }

        __syncthreads();

        if (idx == 0)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                output[i] = big_integer_temp[i];
            }
        }
    }

    __global__ void sk_multiplication_ckks(Data64* ciphertext,
                                           Data64* plaintext, Data64* sk,
                                           Modulus64* modulus, int n_power,
                                           int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data64 ct_0 = ciphertext[index];
        Data64 ct_1 = ciphertext[index + (decomp_mod_count << n_power)];
        Data64 sk_ = sk[index];

        ct_1 = OPERATOR_GPU_64::mult(ct_1, sk_, modulus[block_y]);
        ct_0 = OPERATOR_GPU_64::add(ct_1, ct_0, modulus[block_y]);

        plaintext[index] = ct_0;
    }

    //////////////////
    //////////////////

    __global__ void decryption_fusion_bfv_kernel(
        Data64* ct, Data64* plain, Modulus64* modulus, Modulus64 plain_mod,
        Modulus64 gamma, Data64* Qi_t, Data64* Qi_gamma, Data64* Qi_inverse,
        Data64 mulq_inv_t, Data64 mulq_inv_gamma, Data64 inv_gamma, int n_power,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data64 sum_t = 0;
        Data64 sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data64 mt = ct[location];

            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, plain_mod.value, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, gamma_, modulus[i]);

            mt = OPERATOR_GPU_64::mult(mt, Qi_inverse[i], modulus[i]);

            Data64 mt_in_t = OPERATOR_GPU_64::reduce_forced(mt, plain_mod);
            Data64 mt_in_gamma = OPERATOR_GPU_64::reduce_forced(mt, gamma);

            mt_in_t = OPERATOR_GPU_64::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma =
                OPERATOR_GPU_64::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = OPERATOR_GPU_64::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = OPERATOR_GPU_64::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = OPERATOR_GPU_64::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = OPERATOR_GPU_64::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data64 gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data64 gamma_ =
                OPERATOR_GPU_64::reduce_forced(gamma.value, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(gamma_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::add(sum_t, result, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data64 sum_t_ = OPERATOR_GPU_64::reduce_forced(sum_t, plain_mod);
            Data64 sum_gamma_ =
                OPERATOR_GPU_64::reduce_forced(sum_gamma, plain_mod);

            Data64 result = OPERATOR_GPU_64::sub(sum_t_, sum_gamma_, plain_mod);
            result = OPERATOR_GPU_64::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    //////////////////
    //////////////////

} // namespace heongpu