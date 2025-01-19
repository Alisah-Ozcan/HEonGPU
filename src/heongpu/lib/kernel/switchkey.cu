// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "switchkey.cuh"

namespace heongpu
{

    __global__ void cipher_broadcast_kernel(Data* input, Data* output,
                                            Modulus* modulus, int n_power,
                                            int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count

        int location = (rns_mod_count * block_y) << n_power;
        Data input_ = input[idx + (block_y << n_power)];
        Data one_ = 1;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            output[idx + (i << n_power) + location] =
                VALUE_GPU::mult(one_, input_, modulus[i]);
        }
    }

    __global__ void cipher_broadcast_leveled_kernel(Data* input, Data* output,
                                                    Modulus* modulus,
                                                    int first_rns_mod_count,
                                                    int current_rns_mod_count,
                                                    int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Current Decomposition Modulus Count

        int location = (current_rns_mod_count * block_y) << n_power;

        Data input_ = input[idx + (block_y << n_power)];
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

            Data result = VALUE_GPU::reduce_forced(input_, modulus[mod_index]);

            output[idx + (i << n_power) + location] = result;
        }
    }

    __global__ void multiply_accumulate_kernel(Data* input, Data* relinkey,
                                               Data* output, Modulus* modulus,
                                               int n_power,
                                               int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_offset1 = (decomp_mod_count + 1) << n_power;
        int key_offset2 = (decomp_mod_count + 1) << (n_power + 1);

        Data ct_0_sum = 0;
        Data ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            Data in_piece = input[idx + (block_y << n_power) +
                                  ((i * (decomp_mod_count + 1)) << n_power)];

            Data rk0 = relinkey[idx + (block_y << n_power) + (key_offset2 * i)];
            Data rk1 = relinkey[idx + (block_y << n_power) + (key_offset2 * i) +
                                key_offset1];

            Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus[block_y]);
            Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus[block_y]);

            ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus[block_y]);
            ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus[block_y]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) + key_offset1] = ct_1_sum;
    }

    __global__ void
    multiply_accumulate_method_II_kernel(Data* input, Data* relinkey,
                                         Data* output, Modulus* modulus,
                                         int n_power, int Q_tilda_size, int d)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_offset1 = (Q_tilda_size) << n_power;
        int key_offset2 = (Q_tilda_size) << (n_power + 1);

        Data ct_0_sum = 0;
        Data ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data in_piece = input[idx + (block_y << n_power) +
                                  ((i * (Q_tilda_size)) << n_power)];

            Data rk0 = relinkey[idx + (block_y << n_power) + (key_offset2 * i)];
            Data rk1 = relinkey[idx + (block_y << n_power) + (key_offset2 * i) +
                                key_offset1];

            Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus[block_y]);
            Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus[block_y]);

            ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus[block_y]);
            ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus[block_y]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) + key_offset1] = ct_1_sum;
    }

    __global__ void multiply_accumulate_leveled_kernel(
        Data* input, Data* relinkey, Data* output, Modulus* modulus,
        int first_rns_mod_count, int current_decomp_mod_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_index = (block_y == current_decomp_mod_count)
                            ? (first_rns_mod_count - 1)
                            : block_y;

        int key_offset1 = first_rns_mod_count << n_power;
        int key_offset2 = first_rns_mod_count << (n_power + 1);

        Data ct_0_sum = 0;
        Data ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < current_decomp_mod_count; i++)
        {
            Data in_piece =
                input[idx + (block_y << n_power) +
                      ((i * (current_decomp_mod_count + 1)) << n_power)];

            Data rk0 =
                relinkey[idx + (key_index << n_power) + (key_offset2 * i)];
            Data rk1 = relinkey[idx + (key_index << n_power) +
                                (key_offset2 * i) + key_offset1];

            Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus[key_index]);
            Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus[key_index]);

            ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus[key_index]);
            ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus[key_index]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) +
               ((current_decomp_mod_count + 1) << n_power)] = ct_1_sum;
    }

    __global__ void multiply_accumulate_leveled_method_II_kernel(
        Data* input, Data* relinkey, Data* output, Modulus* modulus,
        int first_rns_mod_count, int current_decomp_mod_count,
        int current_rns_mod_count, int d, int level, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // RNS Modulus Count

        int key_index =
            (block_y < current_decomp_mod_count) ? block_y : (block_y + level);

        int key_offset1 = first_rns_mod_count << n_power;
        int key_offset2 = first_rns_mod_count << (n_power + 1);

        Data ct_0_sum = 0;
        Data ct_1_sum = 0;
#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data in_piece = input[idx + (block_y << n_power) +
                                  ((i * (current_rns_mod_count)) << n_power)];

            Data rk0 =
                relinkey[idx + (key_index << n_power) + (key_offset2 * i)];
            Data rk1 = relinkey[idx + (key_index << n_power) +
                                (key_offset2 * i) + key_offset1];

            Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus[key_index]);
            Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus[key_index]);

            ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus[key_index]);
            ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus[key_index]);
        }

        output[idx + (block_y << n_power)] = ct_0_sum;
        output[idx + (block_y << n_power) +
               (current_rns_mod_count << n_power)] = ct_1_sum;
    }

    __global__ void divide_round_lastq_kernel(Data* input, Data* ct,
                                              Data* output, Modulus* modulus,
                                              Data* half, Data* half_mod,
                                              Data* last_q_modinv, int n_power,
                                              int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_ct = input[idx + (decomp_mod_count << n_power) +
                             (((decomp_mod_count + 1) << n_power) * block_z)];

        last_ct = VALUE_GPU::add(last_ct, half[0], modulus[decomp_mod_count]);

        Data zero_ = 0;
        last_ct = VALUE_GPU::add(last_ct, zero_, modulus[block_y]);

        last_ct = VALUE_GPU::sub(last_ct, half_mod[block_y], modulus[block_y]);

        Data input_ = input[idx + (block_y << n_power) +
                            (((decomp_mod_count + 1) << n_power) * block_z)];

        input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        Data ct_in = ct[idx + (block_y << n_power) +
                        (((decomp_mod_count) << n_power) * block_z)];

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_switchkey_kernel(
        Data* input, Data* ct, Data* output, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, int n_power, int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_ct = input[idx + (decomp_mod_count << n_power) +
                             (((decomp_mod_count + 1) << n_power) * block_z)];

        last_ct = VALUE_GPU::add(last_ct, half[0], modulus[decomp_mod_count]);

        Data zero_ = 0;
        last_ct = VALUE_GPU::add(last_ct, zero_, modulus[block_y]);

        last_ct = VALUE_GPU::sub(last_ct, half_mod[block_y], modulus[block_y]);

        Data input_ = input[idx + (block_y << n_power) +
                            (((decomp_mod_count + 1) << n_power) * block_z)];

        input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        Data ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((decomp_mod_count) << n_power) * block_z)];
        }

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_extended_kernel(
        Data* input, Data* ct, Data* output, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, int n_power, int Q_prime_size,
        int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        Data zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = VALUE_GPU::add(last_ct_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::add(last_ct_add_half_, zero_,
                                            modulus[Q_size + j]);
                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1, modulus[Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::add(last_ct_add_half_, zero_, modulus[block_y]);
            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data ct_in =
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)];

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    __global__ void divide_round_lastq_extended_switchkey_kernel(
        Data* input, Data* ct, Data* output, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, int n_power, int Q_prime_size,
        int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        Data zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = VALUE_GPU::add(last_ct_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::add(last_ct_add_half_, zero_,
                                            modulus[Q_size + j]);
                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1, modulus[Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::add(last_ct_add_half_, zero_, modulus[block_y]);
            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((Q_size) << n_power) * block_z)];
        }

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    __global__ void DivideRoundLastqNewP_leveled(Data* input, Data* ct,
                                                 Data* output, Modulus* modulus,
                                                 Data* half, Data* half_mod,
                                                 Data* last_q_modinv,
                                                 int n_power, int Q_prime_size,
                                                 int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        Data zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = VALUE_GPU::add(last_ct_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::add(last_ct_add_half_, zero_,
                                            modulus[Q_size + j]);
                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1, modulus[Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::add(last_ct_add_half_, zero_, modulus[block_y]);
            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        Data ct_in =
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)];

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            ct_in;
    }

    ////////////
    __global__ void divide_round_lastq_leveled_stage_one_kernel(
        Data* input, Data* output, Modulus* modulus, Data* half, Data* half_mod,
        int n_power, int first_decomp_count, int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Cipher Size (2)

        Data last_ct =
            input[idx + (current_decomp_count << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_y)];

        last_ct = VALUE_GPU::add(last_ct, half[0], modulus[first_decomp_count]);

#pragma unroll
        for (int i = 0; i < current_decomp_count; i++)
        {
            Data last_ct_i = VALUE_GPU::reduce_forced(last_ct, modulus[i]);

            last_ct_i = VALUE_GPU::sub(last_ct_i, half_mod[i], modulus[i]);

            output[idx + (i << n_power) +
                   (((current_decomp_count) << n_power) * block_y)] = last_ct_i;
        }
    }

    __global__ void divide_round_lastq_leveled_stage_two_kernel(
        Data* input_last, Data* input, Data* ct, Data* output, Modulus* modulus,
        Data* last_q_modinv, int n_power, int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        Data ct_in = ct[idx + (block_y << n_power) +
                        (((current_decomp_count) << n_power) * block_z)];

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = ct_in;
    }

    __global__ void divide_round_lastq_leveled_stage_two_switchkey_kernel(
        Data* input_last, Data* input, Data* ct, Data* output, Modulus* modulus,
        Data* last_q_modinv, int n_power, int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        Data ct_in = 0ULL;
        if (block_z == 0)
        {
            ct_in = ct[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];
        }

        ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = ct_in;
    }

    ///////////////////////////////////////////
    // FOR RESCALE

    __global__ void move_cipher_leveled_kernel(Data* input, Data* output,
                                               int n_power,
                                               int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // current_decomp_count - 1
        int block_z = blockIdx.z; // Cipher Size (2)

        Data r_input =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        output[idx + (block_y << n_power) +
               (((current_decomp_count + 1) << n_power) * block_z)] = r_input;
    }

    __global__ void divide_round_lastq_rescale_kernel(
        Data* input_last, Data* input, Data* output, Modulus* modulus,
        Data* last_q_modinv, int n_power, int current_decomp_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_ct =
            input_last[idx + (block_y << n_power) +
                       (((current_decomp_count) << n_power) * block_z)];

        Data input_ =
            input[idx + (block_y << n_power) +
                  (((current_decomp_count + 1) << n_power) * block_z)];

        input_ = VALUE_GPU::sub(input_, last_ct, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        output[idx + (block_y << n_power) +
               (((current_decomp_count) << n_power) * block_z)] = input_;
    }

    ////////////////////////////////////////////////////////////////////////////

    __global__ void base_conversion_DtoB_relin_kernel(
        Data* ciphertext, Data* output, Modulus* modulus, Modulus* B_base,
        Data* base_change_matrix_D_to_B, Data* Mi_inv_D_to_B, Data* prod_D_to_B,
        int* I_j_, int* I_location_, int n_power, int l, int d_tilda, int d,
        int r_prime)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * r_prime) << n_power);
        int matrix_index = I_location * r_prime;

        Data partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = ciphertext[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_D_to_B[I_location + i],
                                         modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = round(r);
        r = roundf(r);
        Data r_ = static_cast<Data>(r);

        for (int i = 0; i < r_prime; i++)
        {
            Data temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data mult = VALUE_GPU::mult(
                    partial[j],
                    base_change_matrix_D_to_B[j + (i * I_j) + matrix_index],
                    B_base[i]);
                temp = VALUE_GPU::add(temp, mult, B_base[i]);
            }

            Data r_mul = VALUE_GPU::mult(
                r_, prod_D_to_B[i + (block_y * r_prime)], B_base[i]);
            r_mul = VALUE_GPU::sub(temp, r_mul, B_base[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoQtilde_relin_kernel(
        Data* ciphertext, Data* output, Modulus* modulus,
        Data* base_change_matrix_D_to_Qtilda, Data* Mi_inv_D_to_Qtilda,
        Data* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power, int l,
        int Q_tilda, int d)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * Q_tilda) << n_power);
        int matrix_index = I_location * Q_tilda;

        Data partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = ciphertext[location + (i << n_power)];
            partial[i] =
                VALUE_GPU::mult(temp, Mi_inv_D_to_Qtilda[I_location + i],
                                modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = round(r);
        r = roundf(r);
        Data r_ = static_cast<Data>(r);

        for (int i = 0; i < Q_tilda; i++)
        {
            Data temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data mult = VALUE_GPU::mult(
                    partial[j],
                    base_change_matrix_D_to_Qtilda[j + (i * I_j) +
                                                   matrix_index],
                    modulus[i]);
                temp = VALUE_GPU::add(temp, mult, modulus[i]);
            }

            Data r_mul = VALUE_GPU::mult(
                r_, prod_D_to_Qtilda[i + (block_y * Q_tilda)], modulus[i]);
            r_mul = VALUE_GPU::sub(temp, r_mul, modulus[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoB_relin_leveled_kernel(
        Data* ciphertext, Data* output, Modulus* modulus, Modulus* B_base,
        Data* base_change_matrix_D_to_B, Data* Mi_inv_D_to_B, Data* prod_D_to_B,
        int* I_j_, int* I_location_, int n_power, int d_tilda, int d,
        int r_prime, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * r_prime) << n_power);
        int matrix_index = I_location * r_prime;

        Data partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = ciphertext[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_D_to_B[I_location + i],
                                         modulus[mod_index[I_location + i]]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[mod_index[I_location + i]].value);
            r += (div / mod);
        }

        // r = roundf(r);
        r = round(r);
        Data r_ = static_cast<Data>(r);

        for (int i = 0; i < r_prime; i++)
        {
            Data temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data mult = VALUE_GPU::mult(
                    partial[j],
                    base_change_matrix_D_to_B[j + (i * I_j) + matrix_index],
                    B_base[i]);
                temp = VALUE_GPU::add(temp, mult, B_base[i]);
            }

            Data r_mul = VALUE_GPU::mult(
                r_, prod_D_to_B[i + (block_y * r_prime)], B_base[i]);
            r_mul = VALUE_GPU::sub(temp, r_mul, B_base[i]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void base_conversion_DtoQtilde_relin_leveled_kernel(
        Data* ciphertext, Data* output, Modulus* modulus,
        Data* base_change_matrix_D_to_Qtilda, Data* Mi_inv_D_to_Qtilda,
        Data* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power, int d,
        int current_Qtilda_size, int current_Q_size, int level, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d

        const int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];

        int location = idx + (I_location << n_power);
        int location_out = idx + ((block_y * current_Qtilda_size) << n_power);
        int matrix_index = I_location * current_Qtilda_size;

        Data partial[20];
        float r = 0;
        float div;
        float mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = ciphertext[location + (i << n_power)];
            partial[i] =
                VALUE_GPU::mult(temp, Mi_inv_D_to_Qtilda[I_location + i],
                                modulus[I_location + i]);
            div = static_cast<float>(partial[i]);
            mod = static_cast<float>(modulus[I_location + i].value);
            r += (div / mod);
        }

        // r = roundf(r);
        r = round(r);
        Data r_ = static_cast<Data>(r);

        for (int i = 0; i < current_Qtilda_size; i++)
        {
            int mod_location = (i < current_Q_size) ? i : (i + level);

            Data temp = 0;
#pragma unroll
            for (int j = 0; j < I_j; j++)
            {
                Data mult =
                    VALUE_GPU::reduce_forced(partial[j], modulus[mod_location]);
                mult = VALUE_GPU::mult(
                    mult,
                    base_change_matrix_D_to_Qtilda[j + (i * I_j) +
                                                   matrix_index],
                    modulus[mod_location]);
                temp = VALUE_GPU::add(temp, mult, modulus[mod_location]);
            }

            Data r_mul = VALUE_GPU::mult(
                r_, prod_D_to_Qtilda[i + (block_y * current_Qtilda_size)],
                modulus[mod_location]);
            r_mul = VALUE_GPU::sub(temp, r_mul, modulus[mod_location]);
            output[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void multiply_accumulate_extended_kernel(
        Data* input, Data* relinkey, Data* output, Modulus* B_prime,
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

        Data* relinkey_ = relinkey + offset2;

        Data ct_0_sum = 0;
        Data ct_1_sum = 0;

#pragma unroll 2
        for (int i = 0; i < d; i++)
        {
            Data in_piece = __ldg(input + offset1 + ((i * r_prime) << n_power));

            Data rk0 = __ldg(relinkey_ + (key_offset2 * i));
            Data mult0 = VALUE_GPU::mult(in_piece, rk0, modulus);
            ct_0_sum = VALUE_GPU::add(ct_0_sum, mult0, modulus);

            Data rk1 = __ldg(relinkey_ + (key_offset2 * i) + key_offset1);
            Data mult1 = VALUE_GPU::mult(in_piece, rk1, modulus);
            ct_1_sum = VALUE_GPU::add(ct_1_sum, mult1, modulus);
        }

        output[offset2] = ct_0_sum;
        output[offset2 + key_offset1] = ct_1_sum;
    }

    __global__ void base_conversion_BtoD_relin_kernel(
        Data* input, Data* output, Modulus* modulus, Modulus* B_base,
        Data* base_change_matrix_B_to_D, Data* Mi_inv_B_to_D, Data* prod_B_to_D,
        int* I_j_, int* I_location_, int n_power, int l_tilda, int d_tilda,
        int d, int r_prime)
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

        Data partial[20];
        float r = 0;
#pragma unroll
        for (int i = 0; i < r_prime; i++)
        {
            Data temp = input[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_B_to_D[i], B_base[i]);
            float div = static_cast<float>(partial[i]);
            float mod = static_cast<float>(B_base[i].value);
            r += (div / mod);
        }
        // r = roundf(r);
        r = round(r);
        Data r_ = static_cast<Data>(r);

#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = 0;
#pragma unroll
            for (int j = 0; j < r_prime; j++)
            {
                Data partial_ = VALUE_GPU::reduce(
                    partial[j], modulus[i + I_location]); // new

                Data mult = VALUE_GPU::mult(
                    partial_,
                    base_change_matrix_B_to_D[j + (i * r_prime) + matrix_index],
                    modulus[i + I_location]);
                temp = VALUE_GPU::add(temp, mult, modulus[i + I_location]);
            }

            Data r_mul = VALUE_GPU::mult(r_, prod_B_to_D[i + I_location],
                                         modulus[i + I_location]);
            temp = VALUE_GPU::sub(temp, r_mul, modulus[i + I_location]);

            temp = VALUE_GPU::reduce(temp, modulus[i + I_location]); // new

            output[location_out + (i << n_power)] = temp;
        }
    }

    __global__ void base_conversion_BtoD_relin_leveled_kernel(
        Data* input, Data* output, Modulus* modulus, Modulus* B_base,
        Data* base_change_matrix_B_to_D, Data* Mi_inv_B_to_D, Data* prod_B_to_D,
        int* I_j_, int* I_location_, int n_power, int l_tilda, int d_tilda,
        int d, int r_prime, int* mod_index)
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

        Data partial[20];
        float r = 0;
#pragma unroll
        for (int i = 0; i < r_prime; i++)
        {
            Data temp = input[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_B_to_D[i], B_base[i]);
            float div = static_cast<float>(partial[i]);
            float mod = static_cast<float>(B_base[i].value);
            r += (div / mod);
        }
        // r = roundf(r);
        r = round(r);
        Data r_ = static_cast<Data>(r);

#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = 0;
#pragma unroll
            for (int j = 0; j < r_prime; j++)
            {
                // Data partial_ = VALUE_GPU::reduce(partial[j],
                // modulus[mod_index[I_location + i]]); // new
                Data partial_ = VALUE_GPU::reduce_forced(
                    partial[j], modulus[mod_index[I_location + i]]); // new

                Data mult = VALUE_GPU::mult(
                    partial_,
                    base_change_matrix_B_to_D[j + (i * r_prime) + matrix_index],
                    modulus[mod_index[I_location + i]]);
                temp = VALUE_GPU::add(temp, mult,
                                      modulus[mod_index[I_location + i]]);
            }

            Data r_mul = VALUE_GPU::mult(r_, prod_B_to_D[i + I_location],
                                         modulus[mod_index[I_location + i]]);
            temp =
                VALUE_GPU::sub(temp, r_mul, modulus[mod_index[I_location + i]]);

            // temp = VALUE_GPU::reduce(temp, modulus[mod_index[I_location +
            // i]]);// new
            temp = VALUE_GPU::reduce_forced(
                temp, modulus[mod_index[I_location + i]]); // new

            output[location_out + (i << n_power)] = temp;
        }
    }

    __global__ void divide_round_lastq_extended_leveled_kernel(
        Data* input, Data* output, Modulus* modulus, Data* half, Data* half_mod,
        Data* last_q_modinv, int n_power, int Q_prime_size, int Q_size,
        int first_Q_prime_size, int first_Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ =
                VALUE_GPU::add(last_ct_add_half_, half[i],
                               modulus[(first_Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::reduce_forced(
                    last_ct_add_half_, modulus[first_Q_size + j]);

                temp1 = VALUE_GPU::sub(temp1,
                                       half_mod[location_ + first_Q_size + j],
                                       modulus[first_Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1,
                                       modulus[first_Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::reduce_forced(last_ct_add_half_, modulus[block_y]);

            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (first_Q_prime_size - 1 - i);
        }

        output[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            input_;
    }

    __global__ void global_memory_replace_kernel(Data* input, Data* output,
                                                 int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        int location =
            idx + (block_y << n_power) + ((gridDim.y << n_power) * block_z);

        Data in_reg = input[location];

        output[location] = in_reg;
    }

    __global__ void global_memory_replace_offset_kernel(
        Data* input, Data* output, int current_decomposition_count, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        int location_in = idx + (block_y << n_power) +
                          ((current_decomposition_count << n_power) * block_z);
        int location_out =
            idx + (block_y << n_power) +
            (((current_decomposition_count - 1) << n_power) * block_z);

        Data in_reg = input[location_in];

        output[location_out] = in_reg;
    }

    __global__ void cipher_broadcast_switchkey_kernel(Data* cipher, Data* out0,
                                                      Data* out1,
                                                      Modulus* modulus,
                                                      int n_power,
                                                      int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        int rns_mod_count = (decomp_mod_count + 1);

        Data result_value = cipher[idx + (block_y << n_power) +
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
        Data* cipher, Data* out0, Data* out1, Modulus* modulus, int n_power,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data result_value = cipher[idx + (block_y << n_power) +
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
        Data* cipher, Data* out0, Data* out1, Modulus* modulus, int n_power,
        int first_rns_mod_count, int current_rns_mod_count,
        int current_decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data result_value =
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

                Data reduced_result =
                    VALUE_GPU::reduce_forced(result_value, modulus[mod_index]);

                out1[idx + (i << n_power) + location] = reduced_result;
            }
        }
    }

    __global__ void addition_switchkey(Data* in1, Data* in2, Data* out,
                                       Modulus* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            out[location] =
                VALUE_GPU::add(in1[location], in2[location], modulus[idy]);
        }
        else
        {
            out[location] = in1[location];
        }
    }

    __global__ void negacyclic_shift_poly_coeffmod_kernel(Data* cipher_in,
                                                          Data* cipher_out,
                                                          Modulus* modulus,
                                                          int shift,
                                                          int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        int coeff_count_minus_one = (1 << n_power) - 1;

        int index_raw = idx + shift;
        int index = index_raw & coeff_count_minus_one;
        Data result_value = cipher_in[idx + (block_y << n_power) +
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

    __global__ void ckks_duplicate_kernel(Data* cipher, Data* output,
                                          Modulus* modulus, int n_power,
                                          int first_rns_mod_count,
                                          int current_rns_mod_count,
                                          int current_decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count

        Data result_value = cipher[idx + (block_y << n_power) +
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

            Data reduced_result =
                VALUE_GPU::reduce_forced(result_value, modulus[mod_index]);

            output[idx + (i << n_power) + location] = reduced_result;
        }
    }

    __global__ void bfv_duplicate_kernel(Data* cipher, Data* output1,
                                         Data* output2, Modulus* modulus,
                                         int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Decomposition Modulus Count

        Data result_value = cipher[idx + (block_y << n_power) +
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
                Data reduced_result =
                    VALUE_GPU::reduce_forced(result_value, modulus[i]);

                output2[idx + (i << n_power) + location] = reduced_result;
            }
        }
    }

    __global__ void divide_round_lastq_permute_ckks_kernel(
        Data* input, Data* input2, Data* output, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, int galois_elt, int n_power,
        int Q_prime_size, int Q_size, int first_Q_prime_size, int first_Q_size,
        int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ =
                VALUE_GPU::add(last_ct_add_half_, half[i],
                               modulus[(first_Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::reduce_forced(
                    last_ct_add_half_, modulus[first_Q_size + j]);

                temp1 = VALUE_GPU::sub(temp1,
                                       half_mod[location_ + first_Q_size + j],
                                       modulus[first_Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1,
                                       modulus[first_Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + first_Q_size + j],
                    modulus[first_Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::reduce_forced(last_ct_add_half_, modulus[block_y]);

            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (first_Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data ct_in = input2[idx + (block_y << n_power)];

            ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

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
        Data* input, Data* ct, Data* output, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, int galois_elt, int n_power,
        int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_ct[15];
        for (int i = 0; i < P_size; i++)
        {
            last_ct[i] = input[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
        }

        Data input_ = input[idx + (block_y << n_power) +
                            ((Q_prime_size << n_power) * block_z)];

        Data zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_ct_add_half_ = last_ct[(P_size - 1 - i)];
            last_ct_add_half_ = VALUE_GPU::add(last_ct_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::add(last_ct_add_half_, zero_,
                                            modulus[Q_size + j]);
                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_ct[j], temp1, modulus[Q_size + j]);

                last_ct[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::add(last_ct_add_half_, zero_, modulus[block_y]);
            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data ct_in = ct[idx + (block_y << n_power)];

            ct_in = VALUE_GPU::add(ct_in, input_, modulus[block_y]);

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