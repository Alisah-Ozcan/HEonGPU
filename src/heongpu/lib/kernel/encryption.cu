// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encryption.cuh"

namespace heongpu
{
    // Not cryptographically secure, will be fixed later.
    __global__ void enc_error_kernel(Data* u_e, Modulus* modulus, int n_power,
                                     int rns_mod_count, int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // u, e1, e2

        curandStatePhilox4_32_10_t state; // not secure
        curand_init(seed, idx + (block_y << n_power), idx + (block_y << n_power), &state);

        if (block_y == 0)
        { // u
            // TODO: make it efficient
            Data random_number = curand(&state) & 3; // 0,1,2,3
            if (random_number == 3)
            {
                random_number -= 3; // 0,1,2
            }

            uint64_t flag = static_cast<uint64_t>(
                -static_cast<int64_t>(random_number == 0));

#pragma unroll
            for (int i = 0; i < rns_mod_count; i++)
            {
                int location = i << n_power;
                Data result = random_number;
                result = result + (flag & modulus[i].value) - 1;
                u_e[idx + location] = result;
            }
        }
        else if (block_y == 1)
        { // e1
            float noise = curand_normal(&state);
            noise = noise * error_std_dev; // SIGMA

            uint64_t flag =
                static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

#pragma unroll
            for (int i = 0; i < rns_mod_count; i++)
            {
                Data rn_ULL =
                    static_cast<Data>(noise) + (flag & modulus[i].value);
                int location = i << n_power;
                u_e[idx + location + ((rns_mod_count) << n_power)] = rn_ULL;
            }
        }
        else
        { // e2
            float noise = curand_normal(&state);
            noise = noise * error_std_dev; // SIGMA

            uint64_t flag =
                static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

#pragma unroll
            for (int i = 0; i < rns_mod_count; i++)
            {
                Data rn_ULL =
                    static_cast<Data>(noise) + (flag & modulus[i].value);
                int location = i << n_power;
                u_e[idx + location + ((rns_mod_count) << (n_power + 1))] =
                    rn_ULL;
            }
        }
    }

    __global__ void pk_u_kernel(Data* pk, Data* u, Data* pk_u, Modulus* modulus,
                                int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count
        int block_z = blockIdx.z; // 2

        Data pk_ = pk[idx + (block_y << n_power) +
                      ((rns_mod_count << n_power) * block_z)];
        Data u_ = u[idx + (block_y << n_power)];

        Data pk_u_ = VALUE_GPU::mult(pk_, u_, modulus[block_y]);

        pk_u[idx + (block_y << n_power) +
             ((rns_mod_count << n_power) * block_z)] = pk_u_;
    }

    __global__ void EncDivideRoundLastq(Data* pk, Data* e, Data* plain,
                                        Data* ct, Modulus* modulus, Data half,
                                        Data* half_mod, Data* last_q_modinv,
                                        Modulus plain_mod, Data Q_mod_t,
                                        Data upper_threshold,
                                        Data* coeffdiv_plain, int n_power,
                                        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count
        int block_z = blockIdx.z; // Cipher Size (2)

        Data last_pk = pk[idx + (decomp_mod_count << n_power) +
                          (((decomp_mod_count + 1) << n_power) * block_z)];
        Data last_e = e[idx + (decomp_mod_count << n_power) +
                        (((decomp_mod_count + 1) << n_power) * block_z)];

        last_pk = VALUE_GPU::add(last_pk, last_e, modulus[decomp_mod_count]);

        last_pk = VALUE_GPU::add(last_pk, half, modulus[decomp_mod_count]);

        Data zero_ = 0;
        last_pk = VALUE_GPU::add(last_pk, zero_, modulus[block_y]);

        last_pk = VALUE_GPU::sub(last_pk, half_mod[block_y], modulus[block_y]);

        Data input_ = pk[idx + (block_y << n_power) +
                         (((decomp_mod_count + 1) << n_power) * block_z)];

        //
        Data e_ = e[idx + (block_y << n_power) +
                    (((decomp_mod_count + 1) << n_power) * block_z)];
        input_ = VALUE_GPU::add(input_, e_, modulus[block_y]);
        //

        input_ = VALUE_GPU::sub(input_, last_pk, modulus[block_y]);

        input_ =
            VALUE_GPU::mult(input_, last_q_modinv[block_y], modulus[block_y]);

        if (block_z == 0)
        {
            Data message = plain[idx];
            Data fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data ct_0 = VALUE_GPU::mult(message, coeffdiv_plain[block_y],
                                        modulus[block_y]);
            ct_0 = VALUE_GPU::add(ct_0, fix, modulus[block_y]);

            input_ = VALUE_GPU::add(input_, ct_0, modulus[block_y]);

            ct[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = input_;
        }
        else
        {
            ct[idx + (block_y << n_power) +
               (((decomp_mod_count) << n_power) * block_z)] = input_;
        }
    }

    __global__ void EncDivideRoundLastqNewP(
        Data* pk, Data* e, Data* plain, Data* ct, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, Modulus plain_mod, Data Q_mod_t,
        Data upper_threshold, Data* coeffdiv_plain, int n_power,
        int Q_prime_size, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_pk[15];
        for (int i = 0; i < P_size; i++)
        {
            Data last_pk_ = pk[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
            Data last_e_ = e[idx + ((Q_size + i) << n_power) +
                             ((Q_prime_size << n_power) * block_z)];
            last_pk[i] = VALUE_GPU::add(last_pk_, last_e_, modulus[Q_size + i]);
        }

        Data input_ = pk[idx + (block_y << n_power) +
                         ((Q_prime_size << n_power) * block_z)];
        Data e_ = e[idx + (block_y << n_power) +
                    ((Q_prime_size << n_power) * block_z)];
        input_ = VALUE_GPU::add(input_, e_, modulus[block_y]);

        Data zero_ = 0;
        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_pk_add_half_ = last_pk[(P_size - 1 - i)];
            last_pk_add_half_ = VALUE_GPU::add(last_pk_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);
            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 = VALUE_GPU::add(last_pk_add_half_, zero_,
                                            modulus[Q_size + j]);
                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_pk[j], temp1, modulus[Q_size + j]);

                last_pk[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            Data temp1 =
                VALUE_GPU::add(last_pk_add_half_, zero_, modulus[block_y]);
            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        if (block_z == 0)
        {
            Data message = plain[idx];
            Data fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data ct_0 = VALUE_GPU::mult(message, coeffdiv_plain[block_y],
                                        modulus[block_y]);
            ct_0 = VALUE_GPU::add(ct_0, fix, modulus[block_y]);

            input_ = VALUE_GPU::add(input_, ct_0, modulus[block_y]);

            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
                input_;
        }
        else
        {
            ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
                input_;
        }
    }

    __global__ void EncDivideRoundLastqNewP_ckks(Data* pk, Data* e, Data* ct,
                                                 Modulus* modulus, Data* half,
                                                 Data* half_mod,
                                                 Data* last_q_modinv,
                                                 int n_power, int Q_prime_size,
                                                 int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)
        int block_z = blockIdx.z; // Cipher Size (2)

        // Max P size is 15.
        Data last_pk[15];
        for (int i = 0; i < P_size; i++)
        {
            Data last_pk_ = pk[idx + ((Q_size + i) << n_power) +
                               ((Q_prime_size << n_power) * block_z)];
            Data last_e_ = e[idx + ((Q_size + i) << n_power) +
                             ((Q_prime_size << n_power) * block_z)];
            last_pk[i] = VALUE_GPU::add(last_pk_, last_e_, modulus[Q_size + i]);
        }

        Data input_ = pk[idx + (block_y << n_power) +
                         ((Q_prime_size << n_power) * block_z)];
        Data e_ = e[idx + (block_y << n_power) +
                    ((Q_prime_size << n_power) * block_z)];
        input_ = VALUE_GPU::add(input_, e_, modulus[block_y]);

        int location_ = 0;
        for (int i = 0; i < P_size; i++)
        {
            Data last_pk_add_half_ = last_pk[(P_size - 1 - i)];
            last_pk_add_half_ = VALUE_GPU::add(last_pk_add_half_, half[i],
                                               modulus[(Q_prime_size - 1 - i)]);

            for (int j = 0; j < (P_size - 1 - i); j++)
            {
                Data temp1 =
                    VALUE_GPU::reduce(last_pk_add_half_, modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + Q_size + j],
                                       modulus[Q_size + j]);

                temp1 = VALUE_GPU::sub(last_pk[j], temp1, modulus[Q_size + j]);

                last_pk[j] = VALUE_GPU::mult(
                    temp1, last_q_modinv[location_ + Q_size + j],
                    modulus[Q_size + j]);
            }

            // Data temp1 = VALUE_GPU::reduce(last_pk_add_half_,
            // modulus[block_y]);
            Data temp1 =
                VALUE_GPU::reduce_forced(last_pk_add_half_, modulus[block_y]);

            temp1 = VALUE_GPU::sub(temp1, half_mod[location_ + block_y],
                                   modulus[block_y]);

            temp1 = VALUE_GPU::sub(input_, temp1, modulus[block_y]);

            input_ = VALUE_GPU::mult(temp1, last_q_modinv[location_ + block_y],
                                     modulus[block_y]);

            location_ = location_ + (Q_prime_size - 1 - i);
        }

        ct[idx + (block_y << n_power) + (((Q_size) << n_power) * block_z)] =
            input_;
    }

    __global__ void cipher_message_add(Data* ciphertext, Data* plaintext,
                                       Modulus* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // Decomposition Modulus Count (Q_size)

        Data ct_0 = ciphertext[idx + (block_y << n_power)];
        Data plaintext_ = plaintext[idx + (block_y << n_power)];

        ct_0 = VALUE_GPU::add(ct_0, plaintext_, modulus[block_y]);

        ciphertext[idx + (block_y << n_power)] = ct_0;
    }

} // namespace heongpu
