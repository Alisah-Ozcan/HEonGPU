// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "addition.cuh"

namespace heongpu
{
    __global__ void addition(Data64* in1, Data64* in2, Data64* out,
                             Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        out[location] =
            OPERATOR_GPU_64::add(in1[location], in2[location], modulus[idy]);
    }

    __global__ void substraction(Data64* in1, Data64* in2, Data64* out,
                                 Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        out[location] =
            OPERATOR_GPU_64::sub(in1[location], in2[location], modulus[idy]);
    }

    __global__ void negation(Data64* in1, Data64* out, Modulus64* modulus,
                             int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        Data64 zero = 0;

        out[location] = OPERATOR_GPU_64::sub(zero, in1[location], modulus[idy]);
    }

    __global__ void addition_plain_bfv_poly(Data64* cipher, Data64* plain,
                                            Data64* output, Modulus64* modulus,
                                            Modulus64 plain_mod, Data64 Q_mod_t,
                                            Data64 upper_threshold,
                                            Data64* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher size

        int location =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);

        if (block_z == 0)
        {
            Data64 message = plain[idx];
            Data64 ciphertext = cipher[location];

            Data64 fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data64 result = OPERATOR_GPU_64::mult(
                message, coeffdiv_plain[block_y], modulus[block_y]);
            result = OPERATOR_GPU_64::add(result, fix, modulus[block_y]);

            result = OPERATOR_GPU_64::add(result, ciphertext, modulus[block_y]);

            output[location] = result;
        }
        else
        {
            Data64 ciphertext = cipher[location];
            output[location] = ciphertext;
        }
    }

    __global__ void addition_plain_bfv_poly_inplace(
        Data64* cipher, Data64* plain, Data64* output, Modulus64* modulus,
        Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
        Data64* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count

        int location = idx + (block_y << n_power);

        Data64 message = plain[idx];
        Data64 ciphertext = cipher[location];

        Data64 fix = message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);

        Data64 result = OPERATOR_GPU_64::mult(message, coeffdiv_plain[block_y],
                                              modulus[block_y]);
        result = OPERATOR_GPU_64::add(result, fix, modulus[block_y]);

        result = OPERATOR_GPU_64::add(result, ciphertext, modulus[block_y]);

        output[location] = result;
    }

    __global__ void
    substraction_plain_bfv_poly(Data64* cipher, Data64* plain, Data64* output,
                                Modulus64* modulus, Modulus64 plain_mod,
                                Data64 Q_mod_t, Data64 upper_threshold,
                                Data64* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher size

        int location =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        if (block_z == 0)
        {
            Data64 message = plain[idx];
            Data64 ciphertext = cipher[location];

            Data64 fix = message * Q_mod_t;
            fix = fix + upper_threshold;
            fix = int(fix / plain_mod.value);

            Data64 result = OPERATOR_GPU_64::mult(
                message, coeffdiv_plain[block_y], modulus[block_y]);
            result = OPERATOR_GPU_64::add(result, fix, modulus[block_y]);

            result = OPERATOR_GPU_64::sub(ciphertext, result, modulus[block_y]);

            output[location] = result;
        }
        else
        {
            Data64 ciphertext = cipher[location];
            output[location] = ciphertext;
        }
    }

    __global__ void substraction_plain_bfv_poly_inplace(
        Data64* cipher, Data64* plain, Data64* output, Modulus64* modulus,
        Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
        Data64* coeffdiv_plain, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count

        int location = idx + (block_y << n_power);

        Data64 message = plain[idx];
        Data64 ciphertext = cipher[location];

        Data64 fix = message * Q_mod_t;
        fix = fix + upper_threshold;
        fix = int(fix / plain_mod.value);

        Data64 result = OPERATOR_GPU_64::mult(message, coeffdiv_plain[block_y],
                                              modulus[block_y]);
        result = OPERATOR_GPU_64::add(result, fix, modulus[block_y]);

        result = OPERATOR_GPU_64::sub(ciphertext, result, modulus[block_y]);

        output[location] = result;
    }

    __global__ void addition_plain_ckks_poly(Data64* in1, Data64* in2,
                                             Data64* out, Modulus64* modulus,
                                             int n_power)
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
            Data64 ciphertext = in1[location];
            out[location] = ciphertext;
        }
    }

    __global__ void substraction_plain_ckks_poly(Data64* in1, Data64* in2,
                                                 Data64* out,
                                                 Modulus64* modulus,
                                                 int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            out[location] = OPERATOR_GPU_64::sub(in1[location], in2[location],
                                                 modulus[idy]);
        }
        else
        {
            out[location] = in1[location];
        }
    }

    __global__ void addition_constant_plain_ckks_poly(Data64* in1, double in2,
                                                      Data64* out,
                                                      Modulus64* modulus,
                                                      double two_pow_64,
                                                      int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            double message_r = in2;

            double coeff_double = round(message_r);
            bool is_negative = signbit(coeff_double);
            coeff_double = fabs(coeff_double);

            // Change Type
            Data64 coeff[2] = {
                static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
                static_cast<std::uint64_t>(coeff_double / two_pow_64)};

            Data64 pt;
            if (is_negative)
            {
                pt = OPERATOR_GPU_64::reduce(coeff, modulus[idy]);
                pt = OPERATOR_GPU_64::sub(modulus[idy].value, pt, modulus[idy]);
            }
            else
            {
                pt = OPERATOR_GPU_64::reduce(coeff, modulus[idy]);
            }

            out[location] =
                OPERATOR_GPU_64::add(in1[location], pt, modulus[idy]);
        }
        else
        {
            Data64 ciphertext = in1[location];
            out[location] = ciphertext;
        }
    }

    __global__ void
    substraction_constant_plain_ckks_poly(Data64* in1, double in2, Data64* out,
                                          Modulus64* modulus, double two_pow_64,
                                          int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        if (idz == 0)
        {
            double message_r = in2;

            double coeff_double = round(message_r);
            bool is_negative = signbit(coeff_double);
            coeff_double = fabs(coeff_double);

            // Change Type
            Data64 coeff[2] = {
                static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
                static_cast<std::uint64_t>(coeff_double / two_pow_64)};

            Data64 pt;
            if (is_negative)
            {
                pt = OPERATOR_GPU_64::reduce(coeff, modulus[idy]);
                pt = OPERATOR_GPU_64::sub(modulus[idy].value, pt, modulus[idy]);
            }
            else
            {
                pt = OPERATOR_GPU_64::reduce(coeff, modulus[idy]);
            }

            out[location] =
                OPERATOR_GPU_64::sub(in1[location], pt, modulus[idy]);
        }
        else
        {
            out[location] = in1[location];
        }
    }

    __global__ void set_zero_cipher_ckks_poly(Data64* in1, Modulus64* modulus,
                                              int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location = idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        in1[location] = 0;
    }

} // namespace heongpu