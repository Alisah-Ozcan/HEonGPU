// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "switchkey.cuh"

namespace heongpu
{

    __global__ void cipher_broadcast_kernel(Data64* input, Data64* output,
                                            Modulus64* modulus, int n_power,
                                            int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count

        int location = (rns_mod_count * block_y) << n_power;
        Data64 input_ = input[idx + (block_y << n_power)];
        Data64 one_ = 1;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            output[idx + (i << n_power) + location] =
                OPERATOR_GPU_64::mult(one_, input_, modulus[i]);
        }
    }

    __global__ void
    cipher_broadcast_leveled_kernel(Data64* input, Data64* output,
                                    Modulus64* modulus, int first_rns_mod_count,
                                    int current_rns_mod_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Current Decomposition Modulus Count

        int location = (current_rns_mod_count * block_y) << n_power;

        Data64 input_ = input[idx + (block_y << n_power)];
        int level = first_rns_mod_count - current_rns_mod_count;
#pragma unroll
        for (int i = 0; i < current_rns_mod_count; i++)
        {
            int mod_index;
            if (i < gridDim.y)
            {
                mod_index = i;
            }
            else
            {
                mod_index = i + level;
            }

            Data64 result =
                OPERATOR_GPU_64::reduce_forced(input_, modulus[mod_index]);

            output[idx + (i << n_power) + location] = result;
        }
    }

    __global__ void multiply_accumulate_kernel(Data64* input, Data64* relinkey,
                                               Data64* output,
                                               Modulus64* modulus, int n_power,
                                               int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_offset1 = (decomp_mod_count + 1) << n_power;
        int key_offset2 = (decomp_mod_count + 1) << (n_power + 1);

        Data64 ct_0_sum = 0;
        Data64 ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            Data64 in_piece = input[idx + (block_y << n_power) +
                                    ((i * (decomp_mod_count + 1)) << n_power)];

            Data64 rk0 =
                relinkey[idx + (block_y << n_power) + (key_offset2 * i)];
            Data64 rk1 = relinkey[idx + (block_y << n_power) +
                                  (key_offset2 * i) + key_offset1];

            Data64 mult0 =
                OPERATOR_GPU_64::mult(in_piece, rk0, modulus[block_y]);
            Data64 mult1 =
                OPERATOR_GPU_64::mult(in_piece, rk1, modulus[block_y]);

            ct_0_sum = OPERATOR_GPU_64::add(ct_0_sum, mult0, modulus[block_y]);
            ct_1_sum = OPERATOR_GPU_64::add(ct_1_sum, mult1, modulus[block_y]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) + key_offset1] = ct_1_sum;
    }

    __global__ void
    multiply_accumulate_method_II_kernel(Data64* input, Data64* relinkey,
                                         Data64* output, Modulus64* modulus,
                                         int n_power, int Q_tilda_size, int d)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_offset1 = (Q_tilda_size) << n_power;
        int key_offset2 = (Q_tilda_size) << (n_power + 1);

        Data64 ct_0_sum = 0;
        Data64 ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 in_piece = input[idx + (block_y << n_power) +
                                    ((i * (Q_tilda_size)) << n_power)];

            Data64 rk0 =
                relinkey[idx + (block_y << n_power) + (key_offset2 * i)];
            Data64 rk1 = relinkey[idx + (block_y << n_power) +
                                  (key_offset2 * i) + key_offset1];

            Data64 mult0 =
                OPERATOR_GPU_64::mult(in_piece, rk0, modulus[block_y]);
            Data64 mult1 =
                OPERATOR_GPU_64::mult(in_piece, rk1, modulus[block_y]);

            ct_0_sum = OPERATOR_GPU_64::add(ct_0_sum, mult0, modulus[block_y]);
            ct_1_sum = OPERATOR_GPU_64::add(ct_1_sum, mult1, modulus[block_y]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) + key_offset1] = ct_1_sum;
    }

    __global__ void multiply_accumulate_leveled_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* modulus,
        int first_rns_mod_count, int current_decomp_mod_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_index = (block_y == current_decomp_mod_count)
                            ? (first_rns_mod_count - 1)
                            : block_y;

        int key_offset1 = first_rns_mod_count << n_power;
        int key_offset2 = first_rns_mod_count << (n_power + 1);

        Data64 ct_0_sum = 0;
        Data64 ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < current_decomp_mod_count; i++)
        {
            Data64 in_piece =
                input[idx + (block_y << n_power) +
                      ((i * (current_decomp_mod_count + 1)) << n_power)];

            Data64 rk0 =
                relinkey[idx + (key_index << n_power) + (key_offset2 * i)];
            Data64 rk1 = relinkey[idx + (key_index << n_power) +
                                  (key_offset2 * i) + key_offset1];

            Data64 mult0 =
                OPERATOR_GPU_64::mult(in_piece, rk0, modulus[key_index]);
            Data64 mult1 =
                OPERATOR_GPU_64::mult(in_piece, rk1, modulus[key_index]);

            ct_0_sum =
                OPERATOR_GPU_64::add(ct_0_sum, mult0, modulus[key_index]);
            ct_1_sum =
                OPERATOR_GPU_64::add(ct_1_sum, mult1, modulus[key_index]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) +
               ((current_decomp_mod_count + 1) << n_power)] = ct_1_sum;
    }

    __global__ void multiply_accumulate_leveled_method_II_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* modulus,
        int first_rns_mod_count, int current_decomp_mod_count,
        int current_rns_mod_count, int d, int level, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_index =
            (block_y < current_decomp_mod_count) ? block_y : (block_y + level);

        int key_offset1 = first_rns_mod_count << n_power;
        int key_offset2 = first_rns_mod_count << (n_power + 1);

        Data64 ct_0_sum = 0;
        Data64 ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 in_piece = input[idx + (block_y << n_power) +
                                    ((i * (current_rns_mod_count)) << n_power)];

            Data64 rk0 =
                relinkey[idx + (key_index << n_power) + (key_offset2 * i)];
            Data64 rk1 = relinkey[idx + (key_index << n_power) +
                                  (key_offset2 * i) + key_offset1];

            Data64 mult0 =
                OPERATOR_GPU_64::mult(in_piece, rk0, modulus[key_index]);
            Data64 mult1 =
                OPERATOR_GPU_64::mult(in_piece, rk1, modulus[key_index]);

            ct_0_sum =
                OPERATOR_GPU_64::add(ct_0_sum, mult0, modulus[key_index]);
            ct_1_sum =
                OPERATOR_GPU_64::add(ct_1_sum, mult1, modulus[key_index]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) +
               (current_rns_mod_count << n_power)] = ct_1_sum;
    }

    __global__ void divide_round_lastq_kernel(Data64* input, Data64* ct,
                                              Data64* output,
                                              Modulus64* modulus, Data64* half,
                                              Data64* half_mod,
                                              Data64* last_q_modinv,
                                              int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_ct = input[idx + (decomp_mod_count << n_power) +
                               (((decomp_mod_count + 1) << n_power) * block_z)];

        last_ct =
            OPERATOR_GPU_64::add(last_ct, half[0], modulus[decomp_mod_count]);

        Data64 zero_ = 0;
        last_ct = OPERATOR_GPU_64::add(last_ct, zero_, modulus[block_y]);

        last_ct =
            OPERATOR_GPU_64::sub(last_ct, half_mod[block_y], modulus[block_y]);

        Data64 input_ = input[idx + (block_y << n_power) +
                              (((decomp_mod_count + 1) << n_power) * block_z)];

        input_ = OPERATOR_GPU_64::sub(input_, last_ct, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        Data64 ct_in = ct[idx + (block_y << n_power) +
                          (((decomp_mod_count) << n_power) * block_z)];

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_ct = input[idx + (decomp_mod_count << n_power) +
                               (((decomp_mod_count + 1) << n_power) * block_z)];

        last_ct =
            OPERATOR_GPU_64::add(last_ct, half[0], modulus[decomp_mod_count]);

        Data64 zero_ = 0;
        last_ct = OPERATOR_GPU_64::add(last_ct, zero_, modulus[block_y]);

        last_ct =
            OPERATOR_GPU_64::sub(last_ct, half_mod[block_y], modulus[block_y]);

        Data64 input_ = input[idx + (block_y << n_power) +
                              (((decomp_mod_count + 1) << n_power) * block_z)];

        input_ = OPERATOR_GPU_64::sub(input_, last_ct, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        Data64 ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((decomp_mod_count) << n_power) * block_z)];
        }

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_extended_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        Data64 zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = OPERATOR_GPU_64::add(
                last_ct_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                    modulus[Q_size + j]);
                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                modulus[block_y]);
            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data64 ct_in =
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)];

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    __global__ void divide_round_lastq_extended_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        Data64 zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = OPERATOR_GPU_64::add(
                last_ct_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                    modulus[Q_size + j]);
                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                modulus[block_y]);
            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data64 ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((Q_size) << n_power) * block_z)];
        }

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    __global__ void DivideRoundLastqNewP_leveled(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        Data64 zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = OPERATOR_GPU_64::add(
                last_ct_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                    modulus[Q_size + j]);
                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                modulus[block_y]);
            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data64 ct_in =
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)];

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    ////////////
    __global__ void divide_round_lastq_leveled_stage_one_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, int n_power, int first_decomp_count,
        int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Cipher Size (2)

        Data64 last_ct =
            input[idx + (current_decomp_count << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_y)];

        last_ct =
            OPERATOR_GPU_64::add(last_ct, half[0], modulus[first_decomp_count]);

#pragma unroll
        for (int i = 0; i < current_decomp_count; i++)
        {
            Data64 last_ct_i =
                OPERATOR_GPU_64::reduce_forced(last_ct, modulus[i]);

            last_ct_i =
                OPERATOR_GPU_64::sub(last_ct_i, half_mod[i], modulus[i]);

            output[idx + (i << n_power) +
                   (((current_decomp_count) << n_power) * block_y)] = last_ct_i;
        }
    }

    __global__ void divide_round_lastq_leveled_stage_two_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data64 input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = OPERATOR_GPU_64::sub(input_, last_ct, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        Data64 ct_in = ct[idx + (block_y << n_power) +
                          (((current_decomp_count) << n_power) * block_z)];

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_leveled_stage_two_switchkey_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data64 input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = OPERATOR_GPU_64::sub(input_, last_ct, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        Data64 ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];
        }

        ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = ct_in;
    }

    ///////////////////////////////////////////
    // FOR RESCALE

    __global__ void move_cipher_leveled_kernel(Data64* input, Data64* output,
                                               int n_power,
                                               int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // current_decomp_count - 1
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 r_input =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        output[idx + (block_y << n_power) +
               (((current_decomp_count + 1) << n_power) * block_z)] = r_input;
    }

    __global__ void divide_round_lastq_rescale_kernel(
        Data64* input_last, Data64* input, Data64* output, Modulus64* modulus,
        Data64* last_q_modinv, int n_power, int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data64 input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = OPERATOR_GPU_64::sub(input_, last_ct, modulus[block_y]);

        input_ = OPERATOR_GPU_64::mult(input_, last_q_modinv[block_y],
                                       modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = input_;
    }

    ////////////////////////////////////////////////////////////////////////////

    __global__ void base_conversion_DtoB_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int l, int d_tilda, int d, int r_prime)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * r_prime) << n_power);
        int matrix_index = I_location * r_prime;

        Data64 partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = ciphertext[location + (i << n_power)];
            partial[i] = OPERATOR_GPU_64::mult(
                temp, Mi_inv_D_to_B[I_location + i], modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = round(r);
        r = roundf(r);
        Data64 r_ = static_cast<Data64>(r);

        for (int i = 0; i < r_prime; i++)
        {
            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    partial[j],
                    base_change_matrix_D_to_B[j + (i * I_j) + matrix_index],
                    B_base[i]);
                temp = OPERATOR_GPU_64::add(temp, mult, B_base[i]);
            }

            Data64 r_mul = OPERATOR_GPU_64::mult(
                r_, prod_D_to_B[i + (block_y * r_prime)], B_base[i]);
            r_mul = OPERATOR_GPU_64::sub(temp, r_mul, B_base[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoQtilde_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int l, int Q_tilda, int d)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * Q_tilda) << n_power);
        int matrix_index = I_location * Q_tilda;

        Data64 partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = ciphertext[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_D_to_Qtilda[I_location + i],
                                      modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = round(r);
        r = roundf(r);
        Data64 r_ = static_cast<Data64>(r);

        for (int i = 0; i < Q_tilda; i++)
        {
            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    partial[j],
                    base_change_matrix_D_to_Qtilda[j + (i * I_j) +
                                                   matrix_index],
                    modulus[i]);
                temp = OPERATOR_GPU_64::add(temp, mult, modulus[i]);
            }

            Data64 r_mul = OPERATOR_GPU_64::mult(
                r_, prod_D_to_Qtilda[i + (block_y * Q_tilda)], modulus[i]);
            r_mul = OPERATOR_GPU_64::sub(temp, r_mul, modulus[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoB_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int d_tilda, int d, int r_prime, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * r_prime) << n_power);
        int matrix_index = I_location * r_prime;

        Data64 partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = ciphertext[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_D_to_B[I_location + i],
                                      modulus[mod_index[I_location + i]]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[mod_index[I_location + i]].value);
            r += (div / mod);
        }

        // r = roundf(r);
        r = round(r);
        Data64 r_ = static_cast<Data64>(r);

        for (int i = 0; i < r_prime; i++)
        {
            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data64 mult = OPERATOR_GPU_64::mult(
                    partial[j],
                    base_change_matrix_D_to_B[j + (i * I_j) + matrix_index],
                    B_base[i]);
                temp = OPERATOR_GPU_64::add(temp, mult, B_base[i]);
            }

            Data64 r_mul = OPERATOR_GPU_64::mult(
                r_, prod_D_to_B[i + (block_y * r_prime)], B_base[i]);
            r_mul = OPERATOR_GPU_64::sub(temp, r_mul, B_base[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoQtilde_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int d, int current_Qtilda_size, int current_Q_size, int level,
        int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * current_Qtilda_size) << n_power);
        int matrix_index = I_location * current_Qtilda_size;

        Data64 partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = ciphertext[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_D_to_Qtilda[I_location + i],
                                      modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = roundf(r);
        r = round(r);
        Data64 r_ = static_cast<Data64>(r);

        for (int i = 0; i < current_Qtilda_size; i++)
        {
            int mod_location = (i < current_Q_size) ? i : (i + level);

            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data64 mult = OPERATOR_GPU_64::reduce_forced(
                    partial[j], modulus[mod_location]);
                mult = OPERATOR_GPU_64::mult(
                    mult,
                    base_change_matrix_D_to_Qtilda[j + (i * I_j) +
                                                   matrix_index],
                    modulus[mod_location]);
                temp = OPERATOR_GPU_64::add(temp, mult, modulus[mod_location]);
            }

            Data64 r_mul = OPERATOR_GPU_64::mult(
                r_, prod_D_to_Qtilda[i + (block_y * current_Qtilda_size)],
                modulus[mod_location]);
            r_mul = OPERATOR_GPU_64::sub(temp, r_mul, modulus[mod_location]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void multiply_accumulate_extended_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* B_prime,
        int n_power, int d_tilda, int d, int r_prime)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // r_prime
        int block_z = blockIdx.z; // d_tilda

        int key_offset1 = (r_prime * d_tilda) << n_power;
        int key_offset2 = (r_prime * d_tilda) << (n_power + 1);

        int offset1 = idx + (block_y << n_power);
        int offset2 = offset1 + ((r_prime << n_power) * block_z);

        Modulus modulus = B_prime[block_y];

        Data64* relinkey_ = relinkey + offset2;

        Data64 ct_0_sum = 0;
        Data64 ct_1_sum = 0;

#pragma unroll 2
        for (int i = 0; i < d; i++)
        {
            Data64 in_piece =
                __ldg(input + offset1 + ((i * r_prime) << n_power));

            Data64 rk0 = __ldg(relinkey_ + (key_offset2 * i));
            Data64 mult0 = OPERATOR_GPU_64::mult(in_piece, rk0, modulus);
            ct_0_sum = OPERATOR_GPU_64::add(ct_0_sum, mult0, modulus);

            Data64 rk1 = __ldg(relinkey_ + (key_offset2 * i) + key_offset1);
            Data64 mult1 = OPERATOR_GPU_64::mult(in_piece, rk1, modulus);
            ct_1_sum = OPERATOR_GPU_64::add(ct_1_sum, mult1, modulus);
        }

        output[offset2] = ct_0_sum;
        output[offset2 + key_offset1] = ct_1_sum;
    }

    __global__ void base_conversion_BtoD_relin_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d_tilda
        int block_z = blockIdx.z; // 2

        int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];
        int location_out =
            idx + (I_location << n_power) + ((l_tilda << n_power) * block_z);

        int location = idx + ((r_prime << n_power) * block_y) +
                       (((d_tilda * r_prime) << n_power) * block_z);
        int matrix_index = I_location * r_prime;

        Data64 partial[20];
        float r = 0;
#pragma unroll
        for (int i = 0; i < r_prime; i++)
        {
            Data64 temp = input[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_B_to_D[i], B_base[i]);
            float div = static_cast<float>(partial[i]);
            float mod = static_cast<float>(B_base[i].value);
            r += (div / mod);
        }
        // r = roundf(r);
        r = round(r);
        Data64 r_ = static_cast<Data64>(r);

#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < r_prime; j++)
            {
                Data64 partial_ = OPERATOR_GPU_64::reduce(
                    partial[j], modulus[i + I_location]); // new

                Data64 mult = OPERATOR_GPU_64::mult(
                    partial_,
                    base_change_matrix_B_to_D[j + (i * r_prime) + matrix_index],
                    modulus[i + I_location]);
                temp =
                    OPERATOR_GPU_64::add(temp, mult, modulus[i + I_location]);
            }

            Data64 r_mul = OPERATOR_GPU_64::mult(
                r_, prod_B_to_D[i + I_location], modulus[i + I_location]);
            temp = OPERATOR_GPU_64::sub(temp, r_mul, modulus[i + I_location]);

            temp =
                OPERATOR_GPU_64::reduce(temp, modulus[i + I_location]); // new

            output[location_out + (i << n_power)] = temp;
        }
    }

    __global__ void base_conversion_BtoD_relin_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d_tilda
        int block_z = blockIdx.z; // 2

        int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];
        int location_out =
            idx + (I_location << n_power) + ((l_tilda << n_power) * block_z);

        int location = idx + ((r_prime << n_power) * block_y) +
                       (((d_tilda * r_prime) << n_power) * block_z);
        int matrix_index = I_location * r_prime;

        Data64 partial[20];
        float r = 0;
#pragma unroll
        for (int i = 0; i < r_prime; i++)
        {
            Data64 temp = input[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_B_to_D[i], B_base[i]);
            float div = static_cast<float>(partial[i]);
            float mod = static_cast<float>(B_base[i].value);
            r += (div / mod);
        }
        // r = roundf(r);
        r = round(r);
        Data64 r_ = static_cast<Data64>(r);

#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = 0;
#pragma unroll
            for (int j = 0; j < r_prime; j++)
            {
                // Data64 partial_ = OPERATOR_GPU_64::reduce(partial[j],
                // modulus[mod_index[I_location + i]]); // new
                Data64 partial_ = OPERATOR_GPU_64::reduce_forced(
                    partial[j], modulus[mod_index[I_location + i]]); // new

                Data64 mult = OPERATOR_GPU_64::mult(
                    partial_,
                    base_change_matrix_B_to_D[j + (i * r_prime) + matrix_index],
                    modulus[mod_index[I_location + i]]);
                temp = OPERATOR_GPU_64::add(temp, mult,
                                            modulus[mod_index[I_location + i]]);
            }

            Data64 r_mul =
                OPERATOR_GPU_64::mult(r_, prod_B_to_D[i + I_location],
                                      modulus[mod_index[I_location + i]]);
            temp = OPERATOR_GPU_64::sub(temp, r_mul,
                                        modulus[mod_index[I_location + i]]);

            // temp = OPERATOR_GPU_64::reduce(temp, modulus[mod_index[I_location
            // + i]]);// new
            temp = OPERATOR_GPU_64::reduce_forced(
                temp, modulus[mod_index[I_location + i]]); // new

            output[location_out + (i << n_power)] = temp;
        }
    }

    __global__ void divide_round_lastq_extended_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, Data64* last_q_modinv, int n_power, int Q_prime_size,
        int Q_size, int first_Q_prime_size, int first_Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ =
                OPERATOR_GPU_64::add(last_ct_add_half_, half[i],
                                     modulus[(first_Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::reduce_forced(
                    last_ct_add_half_, modulus[first_Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(
                    temp1, half_mod[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[first_Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::reduce_forced(last_ct_add_half_,
                                                          modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (first_Q_prime_size - 1 - i);
        }

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            input_;
    }

    __global__ void global_memory_replace_kernel(Data64* input, Data64* output,
                                                 int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        int location =
            idx + (block_y << n_power) + ((gridDim.y << n_power) * block_z);

        Data64 in_reg = input[location];

        output[location] = in_reg;
    }

    __global__ void
    global_memory_replace_offset_kernel(Data64* input, Data64* output,
                                        int current_decomposition_count,
                                        int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        int location_in = idx + (block_y << n_power) +
                          ((current_decomposition_count << n_power) * block_z);
        int location_out =
            idx + (block_y << n_power) +
            (((current_decomposition_count - 1) << n_power) * block_z);

        Data64 in_reg = input[location_in];

        output[location_out] = in_reg;
    }

    __global__ void
    cipher_broadcast_switchkey_kernel(Data64* cipher, Data64* out0,
                                      Data64* out1, Modulus64* modulus,
                                      int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        int rns_mod_count = (decomp_mod_count + 1);

        Data64 result_value = cipher[idx + (block_y << n_power) +
                                     ((decomp_mod_count << n_power) * block_z)];

        if (block_z == 0)
        {
            out0[idx + (block_y << n_power) +
                 ((decomp_mod_count << n_power) * block_z)] = result_value;
        }
        else
        {
            int location = (rns_mod_count * block_y) << n_power;

            for (int i = 0; i < rns_mod_count; i++)
            {
                out1[idx + (i << n_power) + location] = result_value;
            }
        }
    }

    __global__ void cipher_broadcast_switchkey_method_II_kernel(
        Data64* cipher, Data64* out0, Data64* out1, Modulus64* modulus,
        int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 result_value = cipher[idx + (block_y << n_power) +
                                     ((decomp_mod_count << n_power) * block_z)];

        if (block_z == 0)
        {
            out0[idx + (block_y << n_power)] = result_value;
        }
        else
        {
            out1[idx + (block_y << n_power)] = result_value;
        }
    }

    __global__ void cipher_broadcast_switchkey_leveled_kernel(
        Data64* cipher, Data64* out0, Data64* out1, Modulus64* modulus,
        int n_power, int first_rns_mod_count, int current_rns_mod_count,
        int current_decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data64 result_value =
            cipher[idx + (block_y << n_power) +
                   ((current_decomp_mod_count << n_power) * block_z)];

        if (block_z == 0)
        {
            out0[idx + (block_y << n_power) +
                 ((current_decomp_mod_count << n_power) * block_z)] =
                result_value;
        }
        else
        {
            int location = (current_rns_mod_count * block_y) << n_power;
            int level = first_rns_mod_count - current_rns_mod_count;
            for (int i = 0; i < current_rns_mod_count; i++)
            {
                int mod_index;
                if (i < current_decomp_mod_count)
                {
                    mod_index = i;
                }
                else
                {
                    mod_index = i + level;
                }

                Data64 reduced_result = OPERATOR_GPU_64::reduce_forced(
                    result_value, modulus[mod_index]);

                out1[idx + (i << n_power) + location] = reduced_result;
            }
        }
    }

    __global__ void addition_switchkey(Data64* in1, Data64* in2, Data64* out,
                                       Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            out[location] = OPERATOR_GPU_64::add(in1[location], in2[location],
                                                 modulus[idy]);
        }
        else
        {
            out[location] = in1[location];
        }
    }

    __global__ void negacyclic_shift_poly_coeffmod_kernel(Data64* cipher_in,
                                                          Data64* cipher_out,
                                                          Modulus64* modulus,
                                                          int shift,
                                                          int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        int coeff_count_minus_one = (1 << n_power) - 1;

        int index_raw = idx + shift;
        int index = index_raw & coeff_count_minus_one;
        Data64 result_value = cipher_in[idx + (block_y << n_power) +
                                        ((gridDim.y << n_power) * block_z)];

        if ((index_raw >> n_power) & 1)
        {
            result_value = (modulus[block_y].value - result_value);
        }

        cipher_out[index + (block_y << n_power) +
                   ((gridDim.y << n_power) * block_z)] = result_value;
    }

    /////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////
    // Optimized Hoisting-Rotations

    __global__ void ckks_duplicate_kernel(Data64* cipher, Data64* output,
                                          Modulus64* modulus, int n_power,
                                          int first_rns_mod_count,
                                          int current_rns_mod_count,
                                          int current_decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count

        Data64 result_value = cipher[idx + (block_y << n_power) +
                                     ((current_decomp_mod_count << n_power))];

        int location = (current_rns_mod_count * block_y) << n_power;
        int level = first_rns_mod_count - current_rns_mod_count;

        for (int i = 0; i < current_rns_mod_count; i++)
        {
            int mod_index;
            if (i < current_decomp_mod_count)
            {
                mod_index = i;
            }
            else
            {
                mod_index = i + level;
            }

            Data64 reduced_result = OPERATOR_GPU_64::reduce_forced(
                result_value, modulus[mod_index]);

            output[idx + (i << n_power) + location] = reduced_result;
        }
    }

    __global__ void bfv_duplicate_kernel(Data64* cipher, Data64* output1,
                                         Data64* output2, Modulus64* modulus,
                                         int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Decomposition Modulus Count

        Data64 result_value = cipher[idx + (block_y << n_power) +
                                     ((gridDim.y << n_power) * block_z)];

        if (block_z == 0)
        {
            output1[idx + (block_y << n_power)] = result_value;
        }
        else
        {
            int location = (rns_mod_count * block_y) << n_power;

            for (int i = 0; i < rns_mod_count; i++)
            {
                Data64 reduced_result =
                    OPERATOR_GPU_64::reduce_forced(result_value, modulus[i]);

                output2[idx + (i << n_power) + location] = reduced_result;
            }
        }
    }

    __global__ void divide_round_lastq_permute_ckks_kernel(
        Data64* input, Data64* input2, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int first_Q_prime_size,
        int first_Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ =
                OPERATOR_GPU_64::add(last_ct_add_half_, half[i],
                                     modulus[(first_Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::reduce_forced(
                    last_ct_add_half_, modulus[first_Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(
                    temp1, half_mod[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[first_Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::reduce_forced(last_ct_add_half_,
                                                          modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (first_Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data64 ct_in = input2[idx + (block_y << n_power)];

            ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

            //
            int coeff_count_minus_one = (1 << n_power) - 1;

            int index_raw = idx * galois_elt;
            int index = index_raw & coeff_count_minus_one;

            if ((index_raw >> n_power) & 1)
            {
                ct_in = (modulus[block_y].value - ct_in);
            }
            //

            output[index + (block_y << n_power) +
                   (((Q_size) << n_power) * block_z)] = ct_in;
        }
        else
        {
            //
            int coeff_count_minus_one = (1 << n_power) - 1;

            int index_raw = idx * galois_elt;
            int index = index_raw & coeff_count_minus_one;

            if ((index_raw >> n_power) & 1)
            {
                input_ = (modulus[block_y].value - input_);
            }
            //

            output[index + (block_y << n_power) +
                   (((Q_size) << n_power) * block_z)] = input_;
        }
    }

    __global__ void divide_round_lastq_permute_bfv_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data64 last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data64 input_ = input[idx + (block_y << n_power) +
                              ((Q_prime_size << n_power) * block_z)];

        Data64 zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data64 last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = OPERATOR_GPU_64::add(
                last_ct_add_half_, half[i], modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                    modulus[Q_size + j]);
                temp1 = OPERATOR_GPU_64::sub(temp1,
                                             half_mod[location_ + Q_size + j],
                                             modulus[Q_size + j]);

                temp1 = OPERATOR_GPU_64::sub(last_ct[j], temp1,
                                             modulus[Q_size + j]);

                last_ct[j] = OPERATOR_GPU_64::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data64 temp1 = OPERATOR_GPU_64::add(last_ct_add_half_, zero_,
                                                modulus[block_y]);
            temp1 = OPERATOR_GPU_64::sub(temp1, half_mod[location_ + block_y],
                                         modulus[block_y]);

            temp1 = OPERATOR_GPU_64::sub(input_, temp1, modulus[block_y]);

            input_ = OPERATOR_GPU_64::mult(
                temp1, last_q_modinv[location_ + block_y], modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data64 ct_in = ct[idx + (block_y << n_power)];

            ct_in = OPERATOR_GPU_64::add(ct_in, input_, modulus[block_y]);

            //
            int coeff_count_minus_one = (1 << n_power) - 1;

            int index_raw = idx * galois_elt;
            int index = index_raw & coeff_count_minus_one;

            if ((index_raw >> n_power) & 1)
            {
                ct_in = (modulus[block_y].value - ct_in);
            }
            //

            output[index + (block_y << n_power) +
                   (((Q_size) << n_power) * block_z)] = ct_in;
        }
        else
        {
            //
            int coeff_count_minus_one = (1 << n_power) - 1;

            int index_raw = idx * galois_elt;
            int index = index_raw & coeff_count_minus_one;

            if ((index_raw >> n_power) & 1)
            {
                input_ = (modulus[block_y].value - input_);
            }
            //

            output[index + (block_y << n_power) +
                   (((Q_size) << n_power) * block_z)] = input_;
        }
    }

} // namespace heongpu