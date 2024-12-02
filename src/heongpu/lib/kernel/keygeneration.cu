// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "keygeneration.cuh"

namespace heongpu
{

    /////////////////////////////////////////////////////////////////////////////////////
    // Secret Key Generation

    __device__ int conjugate(int* data, int& idx, int& n_power)
    {
        int n = 1 << n_power;

        if (idx == 0)
        {
            return data[0];
        }

        int mask = n - 1;
        int new_location = (n - idx) & mask;
        return (-data[new_location]);
    }

    __global__ void conjugate_kernel(int* conj_secret_key,
                                     int* orginal_secret_key, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        conj_secret_key[idx] = conjugate(orginal_secret_key, idx, n_power);
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void sk_gen_kernel(int* secret_key, int hamming_weight,
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

    __global__ void sk_rns_kernel(int* input, Data* output, Modulus* modulus,
                                  int n_power, int rns_mod_count, int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        int sk_ = input[idx];

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int location = i << n_power;

            Data result;
            if (sk_ < 0)
            {
                result = modulus[i].value - 1;
            }
            else
            {
                result = static_cast<Data>(sk_);
            }

            output[idx + location] = result;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Public Key Generation

    // Not cryptographically secure, will be fixed later.
    __global__ void error_kernel(Data* a_e, Modulus* modulus, int n_power,
                                 int rns_mod_count, int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // a or e

        curandState_t state;
        curand_init(seed, idx, idx, &state);

        if (block_y == 0)
        { // e

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
                a_e[idx + location] = rn_ULL;
            }
        }
        else
        { // a

#pragma unroll
            for (int i = 0; i < rns_mod_count; i++)
            {
                int location = i << n_power;

                uint32_t rn_lo = curand(&state);
                uint32_t rn_hi = curand(&state);

                uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
                                    static_cast<uint64_t>(rn_lo);
                Data rn_ULL = static_cast<Data>(combined);
                rn_ULL = VALUE_GPU::reduce_forced(rn_ULL, modulus[i]);

                a_e[idx + location + ((rns_mod_count) << n_power)] = rn_ULL;
            }
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void error_kernel_leveled(Data* a_e, Modulus* modulus,
                                         int n_power, int mod_count,
                                         int* mod_index, int seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // a or e

        curandState_t state;
        curand_init(seed, idx, idx, &state);

        if (block_y == 0)
        { // e

            float noise = curand_normal(&state);
            noise = noise * error_std_dev; // SIGMA

            uint64_t flag =
                static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

#pragma unroll
            for (int i = 0; i < mod_count; i++)
            {
                int index_mod = mod_index[i];
                Data rn_ULL = static_cast<Data>(noise) +
                              (flag & modulus[index_mod].value);
                int location = i << n_power;
                a_e[idx + location] = rn_ULL;
            }
        }
        else
        { // a

#pragma unroll
            for (int i = 0; i < mod_count; i++)
            {
                int location = i << n_power;
                int index_mod = mod_index[i];

                uint32_t rn_lo = curand(&state);
                uint32_t rn_hi = curand(&state);

                uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
                                    static_cast<uint64_t>(rn_lo);
                Data rn_ULL = static_cast<Data>(combined);
                rn_ULL = VALUE_GPU::reduce_forced(rn_ULL, modulus[index_mod]);

                a_e[idx + location + ((mod_count) << n_power)] = rn_ULL;
            }
        }
    }

    __global__ void pk_kernel(Data* public_key, Data* secret_key, Data* e_a,
                              Modulus* modulus, int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count
        int block_z = blockIdx.z; // 2

        if (block_z == 0)
        {
            Data sk = secret_key[idx + (block_y << n_power)];
            Data e = e_a[idx + (block_y << n_power)];
            Data a =
                e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];

            Data temp = VALUE_GPU::mult(sk, a, modulus[block_y]);
            temp = VALUE_GPU::add(temp, e, modulus[block_y]);
            Data zero = 0;

            public_key[idx + (block_y << n_power)] =
                VALUE_GPU::sub(zero, temp, modulus[block_y]);
        }
        else
        { // block_z == 1
            Data a =
                e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];
            public_key[idx + (block_y << n_power) +
                       (rns_mod_count << n_power)] = a;
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////
    // Relinearization Key Generation

    __global__ void relinkey_kernel(Data* relin_key, Data* secret_key,
                                    Data* e_a, Modulus* modulus, Data* factor,
                                    int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;

        Data sk = secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data rk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data temp = VALUE_GPU::mult(sk, sk, modulus[block_y]);
                temp = VALUE_GPU::mult(temp, factor[block_y], modulus[block_y]);

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
            }

            relin_key[idx + location1 +
                      ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            relin_key[idx + location1 + ((rns_mod_count * i) << (n_power + 1)) +
                      (rns_mod_count << n_power)] = a;
        }
    }

    __global__ void relinkey_DtoB_kernel(Data* relin_key_temp, Data* relin_key,
                                         Modulus* modulus, Modulus* B_base,
                                         Data* base_change_matrix_D_to_B,
                                         Data* Mi_inv_D_to_B, Data* prod_D_to_B,
                                         int* I_j_, int* I_location_,
                                         int n_power, int l_tilda, int d_tilda,
                                         int d, int r_prime)
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

        Data partial[20];
        double r = 0;
        double div;
        double mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = relin_key_temp[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_D_to_B[I_location + i],
                                         modulus[I_location + i]);
            div = static_cast<double>(partial[i]);
            mod = static_cast<double>(modulus[I_location + i].value);
            r += (div / mod);
        }

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
            relin_key[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void relinkey_DtoB_kernel_leveled2(
        Data* relin_key_temp, Data* relin_key, Modulus* modulus,
        Modulus* B_base, Data* base_change_matrix_D_to_B, Data* Mi_inv_D_to_B,
        Data* prod_D_to_B, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index)
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

        Data partial[20];
        double r = 0;
        double div;
        double mod;
#pragma unroll
        for (int i = 0; i < I_j; i++)
        {
            Data temp = relin_key_temp[location + (i << n_power)];
            partial[i] = VALUE_GPU::mult(temp, Mi_inv_D_to_B[I_location + i],
                                         modulus[mod_index[I_location + i]]);
            div = static_cast<double>(partial[i]);
            mod = static_cast<double>(modulus[mod_index[I_location + i]].value);
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
            relin_key[location_out + (i << n_power)] = r_mul;
        }
    }

    __global__ void
    relinkey_kernel_externel_product(Data* relin_key_temp, Data* secret_key,
                                     Data* e_a, Modulus* modulus, Data* factor,
                                     int* Sk_pair, int n_power, int l_tilda,
                                     int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data sk = secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (l_tilda << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data rk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data temp = VALUE_GPU::mult(sk, sk, modulus[block_y]);

                for (int j = 0; j < P_size; j++)
                {
                    temp = VALUE_GPU::mult(temp, factor[(j * Q_size) + block_y],
                                           modulus[block_y]);
                }

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
            }

            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1))] =
                rk_0;
            relin_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                           (l_tilda << n_power)] = a;
        }
    }

    __global__ void relinkey_kernel_externel_product_leveled(
        Data* relin_key_temp, Data* secret_key, Data* e_a, Modulus* modulus,
        Data* factor, int* Sk_pair, int n_power, int l_tilda, int d, int Q_size,
        int P_size, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int index_mod = mod_index[block_y];

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        // Data sk = secret_key[idx + (block_y << n_power)];
        Data sk = secret_key[idx + (index_mod << n_power)];

        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (l_tilda << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data rk_0 = VALUE_GPU::mult(sk, a, modulus[index_mod]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[index_mod]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[index_mod]);

            if (i == Sk_index)
            {
                Data temp = VALUE_GPU::mult(sk, sk, modulus[index_mod]);

                for (int j = 0; j < P_size; j++)
                {
                    temp =
                        VALUE_GPU::mult(temp, factor[(j * Q_size) + index_mod],
                                        modulus[index_mod]);
                }

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[index_mod]);
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

            int gen = group_order; //5; //3;
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

    __global__ void galoiskey_method_I_kernel(Data* galois_key,
                                              Data* secret_key, Data* e_a,
                                              Modulus* modulus, Data* factor,
                                              int galois_elt, int n_power,
                                              int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;
        int coeff_count = 1 << n_power;

        Data sk = secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data gk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
            gk_0 = VALUE_GPU::add(gk_0, e, modulus[block_y]);
            Data zero = 0;

            gk_0 = VALUE_GPU::sub(zero, gk_0, modulus[block_y]);

            if (i == block_y)
            {
                int permutation_location =
                    permutation(idx, galois_elt, coeff_count, n_power);
                Data sk_permutation =
                    secret_key[(block_y << n_power) + permutation_location];

                sk_permutation = VALUE_GPU::mult(
                    sk_permutation, factor[block_y], modulus[block_y]);

                gk_0 = VALUE_GPU::add(gk_0, sk_permutation, modulus[block_y]);
            }

            galois_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1))] = gk_0;
            galois_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1)) +
                       (rns_mod_count << n_power)] = a;
        }
    }

    __global__ void galoiskey_method_II_kernel(Data* galois_key_temp,
                                               Data* secret_key, Data* e_a,
                                               Modulus* modulus, Data* factor,
                                               int galois_elt, int* Sk_pair,
                                               int n_power, int l_tilda, int d,
                                               int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];
        int coeff_count = 1 << n_power;

        Data sk = secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (l_tilda << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data rk_0 = VALUE_GPU::mult(sk, a, modulus[block_y]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                // Data temp = VALUE_GPU::mult(sk, sk, modulus[block_y]);

                int permutation_location =
                    permutation(idx, galois_elt, coeff_count, n_power);
                Data temp =
                    secret_key[(block_y << n_power) + permutation_location];

                for (int j = 0; j < P_size; j++)
                {
                    temp = VALUE_GPU::mult(temp, factor[(j * Q_size) + block_y],
                                           modulus[block_y]);
                }

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
            }

            galois_key_temp[idx + location1 +
                            ((l_tilda * i) << (n_power + 1))] = rk_0;
            galois_key_temp[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                            (l_tilda << n_power)] = a;
        }
    }

    __global__ void switchkey_kernel(Data* switch_key, Data* new_secret_key,
                                     Data* old_secret_key, Data* e_a,
                                     Modulus* modulus, Data* factor,
                                     int n_power, int rns_mod_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // rns_mod_count

        int location1 = block_y << n_power;

        Data new_sk = new_secret_key[idx + (block_y << n_power)];
        Data old_sk = old_secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (rns_mod_count << n_power)];

#pragma unroll
        for (int i = 0; i < rns_mod_count - 1; i++)
        {
            Data rk_0 = VALUE_GPU::mult(new_sk, a, modulus[block_y]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

            if (i == block_y)
            {
                Data temp =
                    VALUE_GPU::mult(old_sk, factor[block_y], modulus[block_y]);

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
            }

            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1))] = rk_0;
            switch_key[idx + location1 +
                       ((rns_mod_count * i) << (n_power + 1)) +
                       (rns_mod_count << n_power)] = a;
        }
    }

    __global__ void switchkey_kernel_method_II(
        Data* switch_key, Data* new_secret_key, Data* old_secret_key, Data* e_a,
        Modulus* modulus, Data* factor, int* Sk_pair, int n_power, int l_tilda,
        int d, int Q_size, int P_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y; // l_tilda

        int location1 = block_y << n_power;
        int Sk_index = Sk_pair[block_y];

        Data new_sk = new_secret_key[idx + (block_y << n_power)];
        Data old_sk = old_secret_key[idx + (block_y << n_power)];
        Data e = e_a[idx + (block_y << n_power)];
        Data a = e_a[idx + (block_y << n_power) + (l_tilda << n_power)];

#pragma unroll
        for (int i = 0; i < d; i++)
        {
            Data rk_0 = VALUE_GPU::mult(new_sk, a, modulus[block_y]);
            rk_0 = VALUE_GPU::add(rk_0, e, modulus[block_y]);
            Data zero = 0;

            rk_0 = VALUE_GPU::sub(zero, rk_0, modulus[block_y]);

            if (i == Sk_index)
            {
                Data temp = old_sk;

                for (int j = 0; j < P_size; j++)
                {
                    temp = VALUE_GPU::mult(temp, factor[(j * Q_size) + block_y],
                                           modulus[block_y]);
                }

                rk_0 = VALUE_GPU::add(rk_0, temp, modulus[block_y]);
            }

            switch_key[idx + location1 + ((l_tilda * i) << (n_power + 1))] =
                rk_0;
            switch_key[idx + location1 + ((l_tilda * i) << (n_power + 1)) +
                       (l_tilda << n_power)] = a;
        }
    }

} // namespace heongpu