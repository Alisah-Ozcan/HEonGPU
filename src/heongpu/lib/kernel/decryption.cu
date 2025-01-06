// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "decryption.cuh"

namespace heongpu
{
    __global__ void sk_multiplication(Data* ct1, Data* sk, Data* output,
                                      Modulus* modulus, int n_power,
                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data ct_1 = ct1[index];
        Data sk_ = sk[index];

        output[index] = VALUE_GPU::mult(ct_1, sk_, modulus[block_y]);
    }

    __global__ void sk_multiplicationx3(Data* ct1, Data* sk, Modulus* modulus,
                                        int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data ct_1 = ct1[index];
        Data sk_ = sk[index];
        ct1[index] = VALUE_GPU::mult(ct_1, sk_, modulus[block_y]);

        Data ct_2 = ct1[index + (decomp_mod_count << n_power)];
        Data sk2_ = VALUE_GPU::mult(sk_, sk_, modulus[block_y]);
        ct1[index + (decomp_mod_count << n_power)] =
            VALUE_GPU::mult(ct_2, sk2_, modulus[block_y]);
    }

    __global__ void decryption_kernel(Data* ct0, Data* ct1, Data* plain,
                                      Modulus* modulus, Modulus plain_mod,
                                      Modulus gamma, Data* Qi_t, Data* Qi_gamma,
                                      Data* Qi_inverse, Data mulq_inv_t,
                                      Data mulq_inv_gamma, Data inv_gamma,
                                      int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data sum_t = 0;
        Data sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data mt = VALUE_GPU::add(ct0[location], ct1[location], modulus[i]);

            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, plain_mod.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, gamma_, modulus[i]);

            mt = VALUE_GPU::mult(mt, Qi_inverse[i], modulus[i]);

            Data mt_in_t = VALUE_GPU::reduce_forced(mt, plain_mod);
            Data mt_in_gamma = VALUE_GPU::reduce_forced(mt, gamma);

            mt_in_t = VALUE_GPU::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma = VALUE_GPU::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = VALUE_GPU::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = VALUE_GPU::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = VALUE_GPU::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = VALUE_GPU::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(gamma_, sum_gamma_, plain_mod);
            result = VALUE_GPU::add(sum_t, result, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data sum_t_ = VALUE_GPU::reduce_forced(sum_t, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(sum_t_, sum_gamma_, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void decryption_kernelx3(Data* ct0, Data* ct1, Data* ct2,
                                        Data* plain, Modulus* modulus,
                                        Modulus plain_mod, Modulus gamma,
                                        Data* Qi_t, Data* Qi_gamma,
                                        Data* Qi_inverse, Data mulq_inv_t,
                                        Data mulq_inv_gamma, Data inv_gamma,
                                        int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data sum_t = 0;
        Data sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data mt = VALUE_GPU::add(ct0[location], ct1[location], modulus[i]);

            mt = VALUE_GPU::add(mt, ct2[location], modulus[i]);

            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, plain_mod.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, gamma_, modulus[i]);

            mt = VALUE_GPU::mult(mt, Qi_inverse[i], modulus[i]);

            Data mt_in_t = VALUE_GPU::reduce_forced(mt, plain_mod);
            Data mt_in_gamma = VALUE_GPU::reduce_forced(mt, gamma);

            mt_in_t = VALUE_GPU::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma = VALUE_GPU::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = VALUE_GPU::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = VALUE_GPU::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = VALUE_GPU::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = VALUE_GPU::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(gamma_, sum_gamma_, plain_mod);
            result = VALUE_GPU::add(sum_t, result, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data sum_t_ = VALUE_GPU::reduce_forced(sum_t, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(sum_t_, sum_gamma_, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    __global__ void coeff_multadd(Data* input1, Data* input2, Data* output,
                                  Modulus plain_mod, Modulus* modulus,
                                  int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data ct_0 = input1[index];
        Data ct_1 = input2[index];

        ct_0 = VALUE_GPU::add(ct_1, ct_0, modulus[block_y]);
        ct_0 = VALUE_GPU::mult(ct_0, plain_mod.value, modulus[block_y]);

        output[index] = ct_0;
    }

    __global__ void compose_kernel(Data* input, Data* output, Modulus* modulus,
                                   Data* Mi_inv, Data* Mi,
                                   Data* decryption_modulus,
                                   int coeff_modulus_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data compose_result[50]; // TODO: Define size as global variable
        Data big_integer_result[50]; // TODO: Define size as global variable

        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data base = input[idx + (i << n_power)];
            Data temp = VALUE_GPU::mult(base, Mi_inv[i], modulus[i]);

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

    __global__ void find_max_norm_kernel(Data* input, Data* output,
                                         Data* upper_half_threshold,
                                         Data* decryption_modulus,
                                         int coeff_modulus_count, int n_power)
    {
        int idx = threadIdx.x;
        int offset = blockDim.x;
        int iteration_count = (1 << n_power) / offset;

        Data big_integer_input[50]; // TODO: Define size as global variable
        Data big_integer_temp[50]; // TODO: Define size as global variable

        // Data *big_integer_input = (Data *)alloca(20 * sizeof(Data));
        // Data *big_integer_temp = (Data *)alloca(20 * sizeof(Data));

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

        extern __shared__ Data shared_memory[]; // 1024 64-bit

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

    __global__ void sk_multiplication_ckks(Data* ciphertext, Data* plaintext,
                                           Data* sk, Modulus* modulus,
                                           int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y; // decomp_mod_count

        int index = idx + (block_y << n_power);

        Data ct_0 = ciphertext[index];
        Data ct_1 = ciphertext[index + (decomp_mod_count << n_power)];
        Data sk_ = sk[index];

        ct_1 = VALUE_GPU::mult(ct_1, sk_, modulus[block_y]);
        ct_0 = VALUE_GPU::add(ct_1, ct_0, modulus[block_y]);

        plaintext[index] = ct_0;
    }

    //////////////////
    //////////////////

    __global__ void decryption_fusion_bfv_kernel(
        Data* ct, Data* plain, Modulus* modulus, Modulus plain_mod,
        Modulus gamma, Data* Qi_t, Data* Qi_gamma, Data* Qi_inverse,
        Data mulq_inv_t, Data mulq_inv_gamma, Data inv_gamma, int n_power,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data sum_t = 0;
        Data sum_gamma = 0;

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int location = idx + (i << n_power);

            Data mt = ct[location];

            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, plain_mod.value, modulus[i]);

            mt = VALUE_GPU::mult(mt, gamma_, modulus[i]);

            mt = VALUE_GPU::mult(mt, Qi_inverse[i], modulus[i]);

            Data mt_in_t = VALUE_GPU::reduce_forced(mt, plain_mod);
            Data mt_in_gamma = VALUE_GPU::reduce_forced(mt, gamma);

            mt_in_t = VALUE_GPU::mult(mt_in_t, Qi_t[i], plain_mod);
            mt_in_gamma = VALUE_GPU::mult(mt_in_gamma, Qi_gamma[i], gamma);

            sum_t = VALUE_GPU::add(sum_t, mt_in_t, plain_mod);
            sum_gamma = VALUE_GPU::add(sum_gamma, mt_in_gamma, gamma);
        }

        sum_t = VALUE_GPU::mult(sum_t, mulq_inv_t, plain_mod);
        sum_gamma = VALUE_GPU::mult(sum_gamma, mulq_inv_gamma, gamma);

        Data gamma_2 = gamma.value >> 1;

        if (sum_gamma > gamma_2)
        {
            Data gamma_ = VALUE_GPU::reduce_forced(gamma.value, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(gamma_, sum_gamma_, plain_mod);
            result = VALUE_GPU::add(sum_t, result, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
        else
        {
            Data sum_t_ = VALUE_GPU::reduce_forced(sum_t, plain_mod);
            Data sum_gamma_ = VALUE_GPU::reduce_forced(sum_gamma, plain_mod);

            Data result = VALUE_GPU::sub(sum_t_, sum_gamma_, plain_mod);
            result = VALUE_GPU::mult(result, inv_gamma, plain_mod);

            plain[idx] = result;
        }
    }

    //////////////////
    //////////////////

} // namespace heongpu