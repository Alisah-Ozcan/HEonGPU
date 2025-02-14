// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef ENCODING_H
#define ENCODING_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "context.cuh"
#include "fft.cuh"
#include "bigintegerarith.cuh"

namespace heongpu
{

    __global__ void encode_kernel_bfv(Data64* message_encoded, Data64* message,
                                      Data64* location_info,
                                      Modulus64* plain_mod, int message_size);

    __global__ void decode_kernel_bfv(Data64* message, Data64* message_encoded,
                                      Data64* location_info);

    __global__ void encode_kernel_double_ckks_conversion(
        Data64* plaintext, double message, Modulus64* modulus,
        int coeff_modulus_count, double two_pow_64, int n_power);

    __global__ void encode_kernel_int_ckks_conversion(Data64* plaintext,
                                                      std::int64_t message,
                                                      Modulus64* modulus,
                                                      int n_power);

    __global__ void double_to_complex_kernel(double* input, Complex64* output);

    __global__ void complex_to_double_kernel(Complex64* input, double* output);

    __global__ void
    encode_kernel_ckks_conversion(Data64* plaintext, Complex64* complex_message,
                                  Modulus64* modulus, int coeff_modulus_count,
                                  double two_pow_64, int* reverse_order,
                                  int n_power);

    __global__ void encode_kernel_compose(
        Complex64* complex_message, Data64* plaintext, Modulus64* modulus,
        Data64* Mi_inv, Data64* Mi, Data64* upper_half_threshold,
        Data64* decryption_modulus, int coeff_modulus_count, double scale,
        double two_pow_64, int* reverse_order, int n_power);

} // namespace heongpu
#endif // ENCODING_H