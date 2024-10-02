// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "addition.cuh"

namespace heongpu
{
    __global__ void addition(Data* in1, Data* in2, Data* out, Modulus* modulus,
                             int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        out[location] =
            VALUE_GPU::add(in1[location], in2[location], modulus[idy]);
    }

    __global__ void substraction(Data* in1, Data* in2, Data* out,
                                 Modulus* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        out[location] =
            VALUE_GPU::sub(in1[location], in2[location], modulus[idy]);
    }

    __global__ void negation(Data* in1, Data* out, Modulus* modulus,
                             int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        Data zero = 0;

        out[location] = VALUE_GPU::sub(zero, in1[location], modulus[idy]);
    }

    __global__ void addition_plain_bfv_poly(Data* cipher, Data* plain,
                                            Data* output, Modulus* modulus,
                                            Modulus plain_mod, Data Q_mod_t,
                                            Data upper_threshold,
                                            Data* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher size

        int location =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);

        if (block_z == 0)
        {
            Data message = plain[idx];
            Data ciphertext = cipher[location];

            Data fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data result = VALUE_GPU::mult(message, coeffdiv_plain[block_y],
                                          modulus[block_y]);
            result = VALUE_GPU::add(result, fix, modulus[block_y]);

            result = VALUE_GPU::add(result, ciphertext, modulus[block_y]);

            output[location] = result;
        }
        else
        {
            Data ciphertext = cipher[location];
            output[location] = ciphertext;
        }
    }

    __global__ void
    addition_plain_bfv_poly_inplace(Data* cipher, Data* plain, Data* output,
                                    Modulus* modulus, Modulus plain_mod,
                                    Data Q_mod_t, Data upper_threshold,
                                    Data* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count

        int location = idx + (block_y << n_power);

        Data message = plain[idx];
        Data ciphertext = cipher[location];

        Data fix = message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);

        Data result =
            VALUE_GPU::mult(message, coeffdiv_plain[block_y], modulus[block_y]);
        result = VALUE_GPU::add(result, fix, modulus[block_y]);

        result = VALUE_GPU::add(result, ciphertext, modulus[block_y]);

        output[location] = result;
    }

    __global__ void substraction_plain_bfv_poly(Data* cipher, Data* plain,
                                                Data* output, Modulus* modulus,
                                                Modulus plain_mod, Data Q_mod_t,
                                                Data upper_threshold,
                                                Data* coeffdiv_plain,
                                                int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher size

        int location =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        if (block_z == 0)
        {
            Data message = plain[idx];
            Data ciphertext = cipher[location];

            Data fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data result = VALUE_GPU::mult(message, coeffdiv_plain[block_y],
                                          modulus[block_y]);
            result = VALUE_GPU::add(result, fix, modulus[block_y]);

            result = VALUE_GPU::sub(ciphertext, result, modulus[block_y]);

            output[location] = result;
        }
        else
        {
            Data ciphertext = cipher[location];
            output[location] = ciphertext;
        }
    }

    __global__ void
    substraction_plain_bfv_poly_inplace(Data* cipher, Data* plain, Data* output,
                                        Modulus* modulus, Modulus plain_mod,
                                        Data Q_mod_t, Data upper_threshold,
                                        Data* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count

        int location = idx + (block_y << n_power);

        Data message = plain[idx];
        Data ciphertext = cipher[location];

        Data fix = message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);

        Data result =
            VALUE_GPU::mult(message, coeffdiv_plain[block_y], modulus[block_y]);
        result = VALUE_GPU::add(result, fix, modulus[block_y]);

        result = VALUE_GPU::sub(ciphertext, result, modulus[block_y]);

        output[location] = result;
    }

    __global__ void addition_plain_ckks_poly(Data* in1, Data* in2, Data* out,
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
            // out[location] = in1[location];
            Data ciphertext = in1[location];
            out[location] = ciphertext;
        }
    }

    __global__ void substraction_plain_ckks_poly(Data* in1, Data* in2,
                                                 Data* out, Modulus* modulus,
                                                 int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            out[location] =
                VALUE_GPU::sub(in1[location], in2[location], modulus[idy]);
        }
        else
        {
            out[location] = in1[location];
        }
    }

} // namespace heongpu