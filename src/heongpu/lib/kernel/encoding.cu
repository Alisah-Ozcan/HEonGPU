// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encoding.cuh"

namespace heongpu
{

    __global__ void encode_kernel_bfv(Data64* message_encoded, Data64* message,
                                      Data64* location_info,
                                      Modulus64* plain_mod, int message_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];

        if (idx < message_size)
        {
            int64_t message_in = static_cast<int64_t>(message[idx]);
            message_in =
                (message_in < 0) ? message_in + plain_mod[0].value : message_in;

            message_encoded[location] = static_cast<Data64>(message_in);
        }
        else
        {
            Data64 zero = 0;
            message_encoded[location] = zero;
        }
    }

    __global__ void decode_kernel_bfv(Data64* message, Data64* message_encoded,
                                      Data64* location_info)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];
        message[idx] = message_encoded[location];
    }

    __global__ void encode_kernel_double_ckks_conversion(
        Data64* plaintext, double message, Modulus64* modulus,
        int coeff_modulus_count, double two_pow_64, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        double message_r = message;

        double coeff_double = round(message_r);
        bool is_negative = signbit(coeff_double);
        coeff_double = fabs(coeff_double);

        // Change Type
        Data64 coeff[2] = {
            static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
            static_cast<std::uint64_t>(coeff_double / two_pow_64)};

        if (is_negative)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                Data64 temp = OPERATOR_GPU_64::reduce(coeff, modulus[i]);
                plaintext[idx + (i << n_power)] =
                    OPERATOR_GPU_64::sub(modulus[i].value, temp, modulus[i]);
            }
        }
        else
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                plaintext[idx + (i << n_power)] =
                    OPERATOR_GPU_64::reduce(coeff, modulus[i]);
            }
        }
    }

    __global__ void encode_kernel_int_ckks_conversion(Data64* plaintext,
                                                      std::int64_t message,
                                                      Modulus64* modulus,
                                                      int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int block_y = blockIdx.y;
        int location = idx + (block_y << n_power);

        Modulus64 mod = modulus[block_y];
        std::int64_t message_r = message;

        if (message < 0)
        {
            message_r = message_r + mod.value;
            Data64 message_d = static_cast<Data64>(message_r);
            message_d = OPERATOR_GPU_64::reduce_forced(message_d, mod);
            plaintext[location] = message_d;
        }
        else
        {
            Data64 message_d = static_cast<Data64>(message_r);
            message_d = OPERATOR_GPU_64::reduce_forced(message_d, mod);
            plaintext[location] = message_d;
        }
    }

    //////////////////////////////
    //////////////////////////////
    //////////////////////////////

    __global__ void double_to_complex_kernel(double* input, Complex64* output)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        double in = input[idx];

        Complex64 c_in(in, 0.0);
        output[idx] = c_in;
    }

    __global__ void complex_to_double_kernel(Complex64* input, double* output)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        Complex64 in = input[idx];

        double d_in = in.real();
        output[idx] = d_in;
    }

    //

    __global__ void
    encode_kernel_ckks_conversion(Data64* plaintext, Complex64* complex_message,
                                  Modulus64* modulus, int coeff_modulus_count,
                                  double two_pow_64, int* reverse_order,
                                  int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // slot_count

        int order = reverse_order[idx];
        Complex64 partial_message = complex_message[order];

        double coeff_double = round(partial_message.real());
        bool is_negative = signbit(coeff_double);
        coeff_double = fabs(coeff_double);

        // Change Type
        Data64 coeff[2] = {
            static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
            static_cast<std::uint64_t>(coeff_double / two_pow_64)};

        if (is_negative)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                Data64 temp = OPERATOR_GPU_64::reduce(coeff, modulus[i]);
                plaintext[idx + (i << n_power)] =
                    OPERATOR_GPU_64::sub(modulus[i].value, temp, modulus[i]);
            }
        }
        else
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                plaintext[idx + (i << n_power)] =
                    OPERATOR_GPU_64::reduce(coeff, modulus[i]);
            }
        }

        // TODO: make it efficient
        int offset = 1 << (n_power - 1);

        double coeff_double2 = round(partial_message.imag());
        bool is_negative2 = signbit(coeff_double2);
        coeff_double2 = fabs(coeff_double2);

        // Change Type
        Data64 coeff2[2] = {
            static_cast<std::uint64_t>(fmod(coeff_double2, two_pow_64)),
            static_cast<std::uint64_t>(coeff_double2 / two_pow_64)};

        if (is_negative2)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                Data64 temp = OPERATOR_GPU_64::reduce(coeff2, modulus[i]);
                plaintext[idx + offset + (i << n_power)] =
                    OPERATOR_GPU_64::sub(modulus[i].value, temp, modulus[i]);
            }
        }
        else
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                plaintext[idx + offset + (i << n_power)] =
                    OPERATOR_GPU_64::reduce(coeff2, modulus[i]);
            }
        }
    }

    __global__ void encode_kernel_compose(
        Complex64* complex_message, Data64* plaintext, Modulus64* modulus,
        Data64* Mi_inv, Data64* Mi, Data64* upper_half_threshold,
        Data64* decryption_modulus, int coeff_modulus_count, double scale,
        double two_pow_64, int* reverse_order, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // slot_count
        double inv_scale = double(1.0) / scale;
        double two_pow_64_reg = two_pow_64;
        int offset = 1 << (n_power - 1);

        Data64 compose_result[50]; // TODO: Define size as global variable
        Data64 big_integer_result[50]; // TODO: Define size as global variable

        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data64 base = plaintext[idx + (i << n_power)];
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

        double result_real = double(0.0);

        bool check1 = biginteger::is_greater_or_equal(
            compose_result, upper_half_threshold, coeff_modulus_count);

        if (check1)
        {
            double scaled_two_pow_64 = inv_scale;
            for (std::size_t j = 0; j < coeff_modulus_count;
                 j++, scaled_two_pow_64 *= two_pow_64_reg)
            {
                if (compose_result[j] > decryption_modulus[j])
                {
                    auto diff = compose_result[j] - decryption_modulus[j];
                    result_real +=
                        diff ? static_cast<double>(diff) * scaled_two_pow_64
                             : 0.0;
                }
                else
                {
                    auto diff = decryption_modulus[j] - compose_result[j];
                    result_real -=
                        diff ? static_cast<double>(diff) * scaled_two_pow_64
                             : 0.0;
                }
            }
        }
        else
        {
            double scaled_two_pow_64 = inv_scale;
            for (std::size_t j = 0; j < coeff_modulus_count;
                 j++, scaled_two_pow_64 *= two_pow_64_reg)
            {
                auto curr_coeff = compose_result[j];
                result_real += curr_coeff ? static_cast<double>(curr_coeff) *
                                                scaled_two_pow_64
                                          : 0.0;
            }
        }

        //////////////////////////
        //////////////////////////
        // TODO: make it efficient
        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data64 base = plaintext[idx + offset + (i << n_power)];
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

        double result_imag = double(0.0);

        bool check2 = biginteger::is_greater_or_equal(
            compose_result, upper_half_threshold, coeff_modulus_count);

        if (check2)
        {
            double scaled_two_pow_64 = inv_scale;
            for (std::size_t j = 0; j < coeff_modulus_count;
                 j++, scaled_two_pow_64 *= two_pow_64_reg)
            {
                if (compose_result[j] > decryption_modulus[j])
                {
                    auto diff = compose_result[j] - decryption_modulus[j];
                    result_imag +=
                        diff ? static_cast<double>(diff) * scaled_two_pow_64
                             : 0.0;
                }
                else
                {
                    auto diff = decryption_modulus[j] - compose_result[j];
                    result_imag -=
                        diff ? static_cast<double>(diff) * scaled_two_pow_64
                             : 0.0;
                }
            }
        }
        else
        {
            double scaled_two_pow_64 = inv_scale;
            for (std::size_t j = 0; j < coeff_modulus_count;
                 j++, scaled_two_pow_64 *= two_pow_64_reg)
            {
                auto curr_coeff = compose_result[j];
                result_imag += curr_coeff ? static_cast<double>(curr_coeff) *
                                                scaled_two_pow_64
                                          : 0.0;
            }
        }

        Complex64 result_c(result_real, result_imag);

        int order = reverse_order[idx];
        complex_message[order] = result_c;
    }

} // namespace heongpu