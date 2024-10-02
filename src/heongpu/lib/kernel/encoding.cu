// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "encoding.cuh"

namespace heongpu
{

    __global__ void encode_kernel_bfv(Data* message_encoded, Data* message,
                                      Data* location_info, Modulus* plain_mod,
                                      int message_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];

        if (idx < message_size)
        {
            int64_t message_in = static_cast<int64_t>(message[idx]);
            message_in =
                (message_in < 0) ? message_in + plain_mod[0].value : message_in;

            message_encoded[location] = static_cast<Data>(message_in);
        }
        else
        {
            Data zero = 0;
            message_encoded[location] = zero;
        }
    }

    __global__ void decode_kernel_bfv(Data* message, Data* message_encoded,
                                      Data* location_info)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];
        message[idx] = message_encoded[location];
    }

    __global__ void encode_kernel_ckks(COMPLEX* message_encoded,
                                       double* message, Data* location_info,
                                       int slot_count)
    {
        int idx = blockIdx.x * blockDim.x +
                  threadIdx.x; // slot_count = (ringsize / 2)

        int location1 = location_info[idx];
        int location2 =
            location_info[slot_count + idx]; // TODO: find efficient way.

        double message_reg = message[idx];
        COMPLEX c_message(message_reg, 0.0);

        message_encoded[location1] = c_message;
        message_encoded[location2] = c_message; // conj
    }

    __global__ void encode_kernel_ckks(COMPLEX* message_encoded,
                                       COMPLEX* message, Data* location_info,
                                       int slot_count)
    {
        int idx = blockIdx.x * blockDim.x +
                  threadIdx.x; // slot_count = (ringsize / 2)

        int location1 = location_info[idx];
        int location2 =
            location_info[slot_count + idx]; // TODO: find efficient way.

        COMPLEX c_message = message[idx];

        message_encoded[location1] = c_message;
        message_encoded[location2] = c_message.conjugate(); // conj
    }

    __global__ void
    encode_kernel_ckks_conversion(Data* plaintext, COMPLEX* complex_message,
                                  Modulus* modulus, int coeff_modulus_count,
                                  double two_pow_64, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        COMPLEX partial_message = complex_message[idx];

        double coeff_double = round(partial_message.real());
        bool is_negative = signbit(coeff_double);
        coeff_double = fabs(coeff_double);

        // Change Type
        Data coeff[2] = {
            static_cast<std::uint64_t>(fmod(coeff_double, two_pow_64)),
            static_cast<std::uint64_t>(coeff_double / two_pow_64)};

        if (is_negative)
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                Data temp = VALUE_GPU::reduce(coeff, modulus[i]);
                plaintext[idx + (i << n_power)] =
                    VALUE_GPU::sub(modulus[i].value, temp, modulus[i]);
            }
        }
        else
        {
            for (int i = 0; i < coeff_modulus_count; i++)
            {
                plaintext[idx + (i << n_power)] =
                    VALUE_GPU::reduce(coeff, modulus[i]);
            }
        }
    }

    __global__ void encode_kernel_compose(COMPLEX* complex_message,
                                          Data* plaintext, Modulus* modulus,
                                          Data* Mi_inv, Data* Mi,
                                          Data* upper_half_threshold,
                                          Data* decryption_modulus,
                                          int coeff_modulus_count, double scale,
                                          double two_pow_64, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size

        Data compose_result[50]; // TODO: Define size as global variable
        Data big_integer_result[50]; // TODO: Define size as global variable

        biginteger::set_zero(compose_result, coeff_modulus_count);

#pragma unroll
        for (int i = 0; i < coeff_modulus_count; i++)
        {
            Data base = plaintext[idx + (i << n_power)];
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

        double inv_scale = double(1.0) / scale;

        double result_d = double(0.0);

        double two_pow_64_reg = two_pow_64;

        bool check = biginteger::is_greater_or_equal(
            compose_result, upper_half_threshold, coeff_modulus_count);

        if (check)
        {
            double scaled_two_pow_64 = inv_scale;
            for (std::size_t j = 0; j < coeff_modulus_count;
                 j++, scaled_two_pow_64 *= two_pow_64_reg)
            {
                if (compose_result[j] > decryption_modulus[j])
                {
                    auto diff = compose_result[j] - decryption_modulus[j];
                    result_d +=
                        diff ? static_cast<double>(diff) * scaled_two_pow_64
                             : 0.0;
                }
                else
                {
                    auto diff = decryption_modulus[j] - compose_result[j];
                    result_d -=
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
                result_d += curr_coeff ? static_cast<double>(curr_coeff) *
                                             scaled_two_pow_64
                                       : 0.0;
            }
        }

        COMPLEX result_c(result_d, 0.0);
        complex_message[idx] = result_c;
    }

    __global__ void decode_kernel_ckks(double* message,
                                       COMPLEX* message_encoded,
                                       Data* location_info, int slot_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];

        COMPLEX c_message = message_encoded[location];
        double message_real_part = c_message.real();
        message[idx] = message_real_part;
    }

    __global__ void decode_kernel_ckks(COMPLEX* message,
                                       COMPLEX* message_encoded,
                                       Data* location_info, int slot_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int location = location_info[idx];

        COMPLEX c_message = message_encoded[location];
        message[idx] = c_message;
    }

} // namespace heongpu