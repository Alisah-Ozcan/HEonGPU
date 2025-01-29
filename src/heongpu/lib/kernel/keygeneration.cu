// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "keygeneration.cuh"

namespace heongpu
{

    /////////////////////////////////////////////////////////////////////////////////////
    // Secret Key Generation

    // Not cryptographically secure, will be fixed later.
    __global__ void secretkey_gen_kernel(int* secret_key, int hamming_weight,
                                         int n_power, int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        secret_key[idx] = 0;

        curandState_t state;
        curand_init(seed, idx, 0, &state);

        if (idx < hamming_weight)
        {
            int mask = (1 << n_power) - 1;
            int location = curand(&state) & mask;
            int value = (curand(&state) & 2) * 2 - 1; // -1 or 1
            atomicExch(&secret_key[location], value);
        }
    }

    __global__ void secretkey_rns_kernel(int* input, Data64* output,
                                         Modulus64* modulus, int n_power,
                                         int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        int sk_ = input[idx];

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int location = i << n_power;

            Data64 result;
            if (sk_ < 0)
            {
                result = modulus[i].value - 1;
            }
            else
            {
                result = static_cast<Data64>(sk_);
            }

            output[idx + location] = result;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Public Key Generation

    __global__ void publickey_gen_kernel(Data64* public_key, Data64* secret_key,
                                         Data64* error_poly, Data64* a_poly,
                                         Modulus64* modulus, int n_power,
                                         int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);

        Data64 sk = secret_key[location];
        Data64 e = error_poly[location];
        Data64 a = a_poly[location];

        Data64 temp = OPERATOR_GPU_64::mult(sk, a, modulus[block_y]);
        temp = OPERATOR_GPU_64::add(temp, e, modulus[block_y]);

        Data64 zero = 0ULL;
        temp = OPERATOR_GPU_64::sub(zero, temp, modulus[block_y]);

        public_key[idx + (block_y << n_power)] = temp;

        public_key[idx + (block_y << n_power) + (rns_mod_count << n_power)] = a;
    }

    __global__ void threshold_pk_addition(Data64* pk1, Data64* pk2,
                                          Data64* pkout, Modulus64* modulus,
                                          int n_power, bool first)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);
        int offset = (gridDim.y << n_power);

        Data64 pk1_0_ = pk1[location];
        Data64 pk2_0_ = pk2[location];

        Data64 sum_ = OPERATOR_GPU_64::add(pk1_0_, pk2_0_, modulus[block_y]);

        pkout[location] = sum_;

        if (first)
        {
            Data64 A = pk1[location + offset];
            pkout[location + offset] = A;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Relinearization Key Generation

    __global__ void relinkey_gen_kernel(Data64* relin_key, Data64* secret_key,
                                        Data64* error_poly, Data64* a_poly,
                                        Modulus64* modulus, Data64* factor,
                                        int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;

        Data64 sk = secret_key[idx + (block_y << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((rns_mod_count * i) << n_power)];
            Data64 a = a_poly[idx + (block_y << n_power) +
                              ((rns_mod_count * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(sk, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data64 temp = OPERATOR_GPU_64::mult(sk, sk, modulus[block_y]);
                temp = OPERATOR_GPU_64::mult(temp, factor[block_y],
                                             modulus[block_y]);

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            relin_key[idx + location1 +
                      ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            relin_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1)) +
                      (rns_mod_count << n_power)] = a;
        }
    }

    ////////////////////////
    //////////////////////// -

    __global__ void multi_party_relinkey_piece_method_I_stage_I_kernel(
        Data64* rk, Data64* sk, Data64* a, Data64* u, Data64* e0, Data64* e1,
        Modulus64* modulus, Data64* factor, int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = block_y << n_power;

        Data64 sk_ = sk[idx + (block_y << n_power)];
        Data64 u_ = u[idx + (block_y << n_power)];

        Data64 zero = 0ULL;
#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data64 e_0 = e0[idx + location + ((rns_mod_count * i) << n_power)];
            Data64 e_1 = e1[idx + location + ((rns_mod_count * i) << n_power)];
            Data64 a_ = a[idx + location + ((rns_mod_count * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(u_, a_, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e_0, modulus[block_y]);

            if (i == block_y)
            {
                Data64 temp = OPERATOR_GPU_64::mult(sk_, factor[block_y],
                                                    modulus[block_y]);

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            Data64 rk_1 = OPERATOR_GPU_64::mult(a_, sk_, modulus[block_y]);
            rk_1 = OPERATOR_GPU_64::add(rk_1, e_1, modulus[block_y]);

            rk[idx + location + ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            rk[idx + location + ((rns_mod_count * i) << (n_power + 1)) +
               (rns_mod_count << n_power)] = rk_1;
        }
    }

    __global__ void multi_party_relinkey_piece_method_II_stage_I_kernel(
        Data64* rk, Data64* sk, Data64* a, Data64* u, Data64* e0, Data64* e1,
        Modulus64* modulus, Data64* factor, int* Sk_pair, int n_power,
        int l_tilda, int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data64 sk_ = sk[idx + (block_y << n_power)];
        Data64 u_ = u[idx + (block_y << n_power)];

        Data64 zero = 0ULL;
#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 e_0 = e0[idx + location + ((l_tilda * i) << n_power)];
            Data64 e_1 = e1[idx + location + ((l_tilda * i) << n_power)];
            Data64 a_ = a[idx + location + ((l_tilda * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(u_, a_, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data64 temp = OPERATOR_GPU_64::mult(
                    sk_, factor[(0 * Q_size) + block_y], modulus[block_y]);

                for (int j = 1; j < P_size; j++)
                {
                    temp = OPERATOR_GPU_64::mult(
                        temp, factor[(j * Q_size) + block_y], modulus[block_y]);
                }

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            Data64 rk_1 = OPERATOR_GPU_64::mult(a_, sk_, modulus[block_y]);
            rk_1 = OPERATOR_GPU_64::add(rk_1, e_1, modulus[block_y]);

            rk[idx + location + ((l_tilda * i) << (n_power + 1))] = rk_0;
            rk[idx + location + ((l_tilda * i) << (n_power + 1)) +
               (l_tilda << n_power)] = rk_1;
        }
    }

    __global__ void multi_party_relinkey_piece_method_I_II_stage_II_kernel(
        Data64* rk_1, Data64* rk_2, Data64* sk, Data64* u, Data64* e0,
        Data64* e1, Modulus64* modulus, int n_power, int rns_mod_count,
        int decomp_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = block_y << n_power;

        Data64 sk_ = sk[idx + (block_y << n_power)];
        Data64 u_ = u[idx + (block_y << n_power)];

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            Data64 e_0 = e0[idx + location + ((rns_mod_count * i) << n_power)];
            Data64 e_1 = e1[idx + location + ((rns_mod_count * i) << n_power)];

            Data64 rk_0_ =
                rk_1[idx + location + ((rns_mod_count * i) << (n_power + 1))];

            Data64 rk_1_ =
                rk_1[idx + location + ((rns_mod_count * i) << (n_power + 1)) +
                     (rns_mod_count << n_power)];

            rk_0_ = OPERATOR_GPU_64::mult(rk_0_, sk_, modulus[block_y]);
            rk_0_ = OPERATOR_GPU_64::add(rk_0_, e_0, modulus[block_y]);

            // u_ = OPERATOR_GPU_64::sub(u_, sk_, modulus[block_y]);
            Data64 u_s = OPERATOR_GPU_64::sub(u_, sk_, modulus[block_y]);
            rk_1_ = OPERATOR_GPU_64::mult(rk_1_, u_s, modulus[block_y]);
            rk_1_ = OPERATOR_GPU_64::add(rk_1_, e_1, modulus[block_y]);

            rk_2[idx + location + ((rns_mod_count * i) << (n_power + 1))] =
                rk_0_;
            rk_2[idx + location + ((rns_mod_count * i) << (n_power + 1)) +
                 (rns_mod_count << n_power)] = rk_1_;
        }
    }

    __global__ void multi_party_relinkey_method_I_stage_I_kernel(
        Data64* rk1, Data64* rk2, Data64* rk_out, Modulus64* modulus,
        int n_power, int Q_prime_size, int l)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);

#pragma unroll
        for (int i = 0; i < l; i++)
        {
            Data64 rk1_0 =
                rk1[location + ((Q_prime_size * i) << (n_power + 1))];
            Data64 rk1_1 =
                rk1[location + ((Q_prime_size * i) << (n_power + 1)) +
                    (Q_prime_size << n_power)];

            Data64 rk2_0 =
                rk2[location + ((Q_prime_size * i) << (n_power + 1))];
            Data64 rk2_1 =
                rk2[location + ((Q_prime_size * i) << (n_power + 1)) +
                    (Q_prime_size << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::add(rk1_0, rk2_0, modulus[block_y]);
            Data64 rk_1 = OPERATOR_GPU_64::add(rk1_1, rk2_1, modulus[block_y]);

            rk_out[location + ((Q_prime_size * i) << (n_power + 1))] = rk_0;
            rk_out[location + ((Q_prime_size * i) << (n_power + 1)) +
                   (Q_prime_size << n_power)] = rk_1;
        }
    }

    __global__ void multi_party_relinkey_method_I_stage_I_kernel(
        Data64* rk_in, Data64* rk_out, Modulus64* modulus, int n_power,
        int Q_prime_size, int l, bool first)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);

        if (first)
        {
#pragma unroll
            for (int i = 0; i < l; i++)
            {
                rk_out[location + ((Q_prime_size * i) << (n_power + 1))] =
                    rk_in[location + ((Q_prime_size * i) << (n_power + 1))];
                rk_out[location + ((Q_prime_size * i) << (n_power + 1)) +
                       (Q_prime_size << n_power)] =
                    rk_in[location + ((Q_prime_size * i) << (n_power + 1)) +
                          (Q_prime_size << n_power)];
            }
        }
        else
        {
#pragma unroll
            for (int i = 0; i < l; i++)
            {
                Data64 rk1_0 =
                    rk_in[location + ((Q_prime_size * i) << (n_power + 1))];
                Data64 rk1_1 =
                    rk_in[location + ((Q_prime_size * i) << (n_power + 1)) +
                          (Q_prime_size << n_power)];

                Data64 rk2_0 =
                    rk_out[location + ((Q_prime_size * i) << (n_power + 1))];
                Data64 rk2_1 =
                    rk_out[location + ((Q_prime_size * i) << (n_power + 1)) +
                           (Q_prime_size << n_power)];

                Data64 rk_0 =
                    OPERATOR_GPU_64::add(rk1_0, rk2_0, modulus[block_y]);
                Data64 rk_1 =
                    OPERATOR_GPU_64::add(rk1_1, rk2_1, modulus[block_y]);

                rk_out[location + ((Q_prime_size * i) << (n_power + 1))] = rk_0;
                rk_out[location + ((Q_prime_size * i) << (n_power + 1)) +
                       (Q_prime_size << n_power)] = rk_1;
            }
        }
    }

    __global__ void multi_party_relinkey_method_I_stage_II_kernel(
        Data64* rk1, Data64* rk2, Data64* rk_out, Modulus64* modulus,
        int n_power, int Q_prime_size, int l)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);

#pragma unroll
        for (int i = 0; i < l; i++)
        {
            Data64 rk1_0 =
                rk1[location + ((Q_prime_size * i) << (n_power + 1))];
            Data64 rk1_1 =
                rk1[location + ((Q_prime_size * i) << (n_power + 1)) +
                    (Q_prime_size << n_power)];

            Data64 rk_1 = rk2[location + ((Q_prime_size * i) << (n_power + 1)) +
                              (Q_prime_size << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::add(rk1_0, rk1_1, modulus[block_y]);

            rk_out[location + ((Q_prime_size * i) << (n_power + 1))] = rk_0;
            rk_out[location + ((Q_prime_size * i) << (n_power + 1)) +
                   (Q_prime_size << n_power)] = rk_1;
        }
    }

    __global__ void multi_party_relinkey_method_I_stage_II_kernel(
        Data64* rk_in, Data64* rk_out, Modulus64* modulus, int n_power,
        int Q_prime_size, int l)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = idx + (block_y << n_power);

#pragma unroll
        for (int i = 0; i < l; i++)
        {
            Data64 rk1_0 =
                rk_in[location + ((Q_prime_size * i) << (n_power + 1))];
            Data64 rk1_1 =
                rk_in[location + ((Q_prime_size * i) << (n_power + 1)) +
                      (Q_prime_size << n_power)];

            // Data64 rk_0 = OPERATOR_GPU_64::add(rk1_0, rk1_1,
            // modulus[block_y]);

            Data64 rk_0 =
                rk_out[location + ((Q_prime_size * i) << (n_power + 1))];
            rk_0 = OPERATOR_GPU_64::add(rk_0, rk1_0, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, rk1_1, modulus[block_y]);

            rk_out[location + ((Q_prime_size * i) << (n_power + 1))] = rk_0;
        }
    }

    ////////////////////////
    ////////////////////////

    __global__ void relinkey_DtoB_kernel(
        Data64* relin_key_temp, Data64* relin_key, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int l_tilda, int d_tilda, int d, int r_prime)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d_tilda
        int block_z = blockIdx.z; // d x 2

        int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];
        int matrix_index = I_location * r_prime;
        int location_out = idx + ((r_prime << n_power) * block_y) +
                           (((r_prime * d_tilda) << n_power) * block_z);

        int location =
            idx + (I_location << n_power) + ((block_z * l_tilda) << n_power);

        Data64 partial[20];
        double r = 0;
        double div;
        double mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = relin_key_temp[location + (i << n_power)];
            partial[i] = OPERATOR_GPU_64::mult(
                temp, Mi_inv_D_to_B[I_location + i], modulus[I_location + i]);
            div = static_cast<double>(partial[i]);
            mod = static_cast<double>(modulus[I_location + i].value);
            r += (div / mod);
        }

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
            relin_key[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void relinkey_DtoB_leveled_kernel(
        Data64* relin_key_temp, Data64* relin_key, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int l_tilda, int d_tilda, int d, int r_prime,
        int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // d_tilda
        int block_z = blockIdx.z; // d x 2

        int I_j = I_j_[block_y];
        int I_location = I_location_[block_y];
        int matrix_index = I_location * r_prime;
        int location_out = idx + ((r_prime << n_power) * block_y) +
                           (((r_prime * d_tilda) << n_power) * block_z);

        int location =
            idx + (I_location << n_power) + ((block_z * l_tilda) << n_power);

        Data64 partial[20];
        double r = 0;
        double div;
        double mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data64 temp = relin_key_temp[location + (i << n_power)];
            partial[i] =
                OPERATOR_GPU_64::mult(temp, Mi_inv_D_to_B[I_location + i],
                                      modulus[mod_index[I_location + i]]);
            div = static_cast<double>(partial[i]);
            mod = static_cast<double>(modulus[mod_index[I_location + i]].value);
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
            relin_key[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void relinkey_gen_II_kernel(
        Data64* relin_key_temp, Data64* secret_key, Data64* error_poly,
        Data64* a_poly, Modulus64* modulus, Data64* factor, int* Sk_pair,
        int n_power, int l_tilda, int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data64 sk = secret_key[idx + (block_y << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((l_tilda * i) << n_power)];
            Data64 a =
                a_poly[idx + (block_y << n_power) + ((l_tilda * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(sk, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data64 temp = OPERATOR_GPU_64::mult(sk, sk, modulus[block_y]);

                for (int j = 0; j < P_size; j++)
                {
                    temp = OPERATOR_GPU_64::mult(
                        temp, factor[(j * Q_size) + block_y], modulus[block_y]);
                }

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1))] =
                rk_0;
            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                           (l_tilda << n_power)] = a;
        }
    }

    __global__ void relinkey_gen_II_leveled_kernel(
        Data64* relin_key_temp, Data64* secret_key, Data64* error_poly,
        Data64* a_poly, Modulus64* modulus, Data64* factor, int* Sk_pair,
        int n_power, int l_tilda, int d, int Q_size, int P_size, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int index_mod = mod_index[block_y];

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data64 sk = secret_key[idx + (index_mod << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((l_tilda * i) << n_power)];
            Data64 a =
                a_poly[idx + (block_y << n_power) + ((l_tilda * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(sk, a, modulus[index_mod]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[index_mod]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[index_mod]);

            if (i == Sk_index)
            {
                Data64 temp = OPERATOR_GPU_64::mult(sk, sk, modulus[index_mod]);

                for (int j = 0; j < P_size; j++)
                {
                    temp = OPERATOR_GPU_64::mult(
                        temp, factor[(j * Q_size) + index_mod],
                        modulus[index_mod]);
                }

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[index_mod]);
            }

            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1))] =
                rk_0;
            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                           (l_tilda << n_power)] = a;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Galois Key Generation

    int steps_to_galois_elt(int steps, int coeff_count, int group_order)
    {
        int n = coeff_count;
        int m32 = n * 2;
        int m = m32;

        if (steps == 0)
        {
            return m - 1;
        }
        else
        {
            int sign = steps < 0;
            int pos_steps = abs(steps);

            if (pos_steps >= (n >> 1))
            {
                std::cerr
                    << "Galois Key can not be generated, Step count too large "
                    << std::endl;
                return 0;
            }

            if (sign)
            {
                steps = (n >> 1) - pos_steps;
            }
            else
            {
                steps = pos_steps;
            }

            int gen = group_order; // 5; //3;
            int galois_elt = 1;
            while (steps > 0)
            {
                galois_elt = galois_elt * gen;
                galois_elt = galois_elt & (m - 1);

                steps = steps - 1;
            }

            return galois_elt;
        }
    }

    __device__ int bitreverse_gpu(int index, int n_power)
    {
        int res_1 = 0;
        for (int i = 0; i < n_power; i++)
        {
            res_1 <<= 1;
            res_1 = (index & 1) | res_1;
            index >>= 1;
        }
        return res_1;
    }

    __device__ int permutation(int index, int galois_elt, int coeff_count,
                               int n_power)
    {
        int coeff_count_minus_one = coeff_count - 1;
        int i = index + coeff_count;

        int reversed = bitreverse_gpu(i, n_power + 1);

        int index_raw = (galois_elt * reversed) >> 1;

        index_raw = index_raw & coeff_count_minus_one;

        return bitreverse_gpu(index_raw, n_power);
    }

    __global__ void galoiskey_gen_kernel(Data64* galois_key, Data64* secret_key,
                                         Data64* error_poly, Data64* a_poly,
                                         Modulus64* modulus, Data64* factor,
                                         int galois_elt, int n_power,
                                         int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;
        int coeff_count = 1 << n_power;

        int permutation_location =
            permutation(idx, galois_elt, coeff_count, n_power);
        Data64 sk_permutation =
            secret_key[(block_y << n_power) + permutation_location];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((rns_mod_count * i) << n_power)];
            Data64 a = a_poly[idx + (block_y << n_power) +
                              ((rns_mod_count * i) << n_power)];

            Data64 gk_0 =
                OPERATOR_GPU_64::mult(sk_permutation, a, modulus[block_y]);
            gk_0 = OPERATOR_GPU_64::add(gk_0, e, modulus[block_y]);
            Data64 zero = 0;

            gk_0 = OPERATOR_GPU_64::sub(zero, gk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data64 sk = secret_key[idx + (block_y << n_power)];

                sk = OPERATOR_GPU_64::mult(sk, factor[block_y],
                                           modulus[block_y]);

                gk_0 = OPERATOR_GPU_64::add(gk_0, sk, modulus[block_y]);
            }

            galois_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1))] = gk_0;
            galois_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1)) +
                       (rns_mod_count << n_power)] = a;
        }
    }

    __global__ void galoiskey_gen_II_kernel(
        Data64* galois_key_temp, Data64* secret_key, Data64* error_poly,
        Data64* a_poly, Modulus64* modulus, Data64* factor, int galois_elt,
        int* Sk_pair, int n_power, int l_tilda, int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];
        int coeff_count = 1 << n_power;

        int permutation_location =
            permutation(idx, galois_elt, coeff_count, n_power);
        Data64 sk_permutation =
            secret_key[(block_y << n_power) + permutation_location];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((l_tilda * i) << n_power)];
            Data64 a =
                a_poly[idx + (block_y << n_power) + ((l_tilda * i) << n_power)];

            Data64 rk_0 =
                OPERATOR_GPU_64::mult(sk_permutation, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data64 sk = secret_key[idx + (block_y << n_power)];

                for (int j = 0; j < P_size; j++)
                {
                    sk = OPERATOR_GPU_64::mult(
                        sk, factor[(j * Q_size) + block_y], modulus[block_y]);
                }

                rk_0 = OPERATOR_GPU_64::add(rk_0, sk, modulus[block_y]);
            }

            galois_key_temp[idx + location1 +
                            ((l_tilda * i) << (n_power + 1))] = rk_0;
            galois_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                            (l_tilda << n_power)] = a;
        }
    }

    /////////////////////

    __global__ void multi_party_galoiskey_gen_method_I_II_kernel(
        Data64* gk_1, Data64* gk_2, Modulus64* modulus, int n_power,
        int rns_mod_count, int decomp_mod_count, bool first)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location = block_y << n_power;
        int offset = (rns_mod_count << n_power);

#pragma unroll
        for (int i = 0; i < decomp_mod_count; i++)
        {
            int inner_location =
                idx + location + ((rns_mod_count * i) << (n_power + 1));

            if (first)
            {
                gk_1[inner_location] = gk_2[inner_location];
                gk_1[inner_location + offset] = gk_2[inner_location + offset];
            }
            else
            {
                Data64 gk_1_r = gk_1[inner_location];
                Data64 gk_2_r = gk_2[inner_location];

                Data64 sum_r =
                    OPERATOR_GPU_64::add(gk_1_r, gk_2_r, modulus[block_y]);
                gk_1[inner_location] = sum_r;
            }
        }
    }

    /////////////////////

    __global__ void switchkey_gen_kernel(Data64* switch_key,
                                         Data64* new_secret_key,
                                         Data64* old_secret_key,
                                         Data64* error_poly, Data64* a_poly,
                                         Modulus64* modulus, Data64* factor,
                                         int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;

        Data64 new_sk = new_secret_key[idx + (block_y << n_power)];
        Data64 old_sk = old_secret_key[idx + (block_y << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((rns_mod_count * i) << n_power)];
            Data64 a = a_poly[idx + (block_y << n_power) +
                              ((rns_mod_count * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(new_sk, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data64 temp = OPERATOR_GPU_64::mult(old_sk, factor[block_y],
                                                    modulus[block_y]);

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1)) +
                       (rns_mod_count << n_power)] = a;
        }
    }

    __global__ void switchkey_gen_II_kernel(
        Data64* switch_key, Data64* new_secret_key, Data64* old_secret_key,
        Data64* error_poly, Data64* a_poly, Modulus64* modulus, Data64* factor,
        int* Sk_pair, int n_power, int l_tilda, int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data64 new_sk = new_secret_key[idx + (block_y << n_power)];
        Data64 old_sk = old_secret_key[idx + (block_y << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data64 e = error_poly[idx + (block_y << n_power) +
                                  ((l_tilda * i) << n_power)];
            Data64 a =
                a_poly[idx + (block_y << n_power) + ((l_tilda * i) << n_power)];

            Data64 rk_0 = OPERATOR_GPU_64::mult(new_sk, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data64 temp = old_sk;

                for (int j = 0; j < P_size; j++)
                {
                    temp = OPERATOR_GPU_64::mult(
                        temp, factor[(j * Q_size) + block_y], modulus[block_y]);
                }

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            switch_key[idx + location1 + ((l_tilda * i) << (n_power + 1))] =
                rk_0;
            switch_key[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                       (l_tilda << n_power)] = a;
        }
    }

    ////////////////

    __global__ void switchkey_kernel(Data64* switch_key, Data64* new_secret_key,
                                     Data64* old_secret_key, Data64* e_a,
                                     Modulus64* modulus, Data64* factor,
                                     int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;

        Data64 new_sk = new_secret_key[idx + (block_y << n_power)];
        Data64 old_sk = old_secret_key[idx + (block_y << n_power)];
        Data64 e = e_a[idx + (block_y << n_power)];
        Data64 a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data64 rk_0 = OPERATOR_GPU_64::mult(new_sk, a, modulus[block_y]);
            rk_0 = OPERATOR_GPU_64::add(rk_0, e, modulus[block_y]);
            Data64 zero = 0;

            rk_0 = OPERATOR_GPU_64::sub(zero, rk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data64 temp = OPERATOR_GPU_64::mult(old_sk, factor[block_y],
                                                    modulus[block_y]);

                rk_0 = OPERATOR_GPU_64::add(rk_0, temp, modulus[block_y]);
            }

            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1)) +
                       (rns_mod_count << n_power)] = a;
        }
    }

} // namespace heongpu