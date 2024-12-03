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

    __global__ void encode_kernel_bfv(Data* message_encoded, Data* message,
                                      Data* location_info, Modulus* plain_mod,
                                      int message_size);

    __global__ void decode_kernel_bfv(Data* message, Data* message_encoded,
                                      Data* location_info);

    __global__ void encode_kernel_double_ckks_conversion(
        Data* plaintext, double message, Modulus* modulus,
        int coeff_modulus_count, double two_pow_64, int n_power);

    __global__ void encode_kernel_int_ckks_conversion(Data* plaintext,
                                                      std::int64_t message,
                                                      Modulus* modulus,
                                                      int n_power);

    __global__ void double_to_complex_kernel(double* input, COMPLEX* output);

    __global__ void complex_to_double_kernel(COMPLEX* input, double* output);

    __global__ void
    encode_kernel_ckks_conversion(Data* plaintext, COMPLEX* complex_message,
                                  Modulus* modulus, int coeff_modulus_count,
                                  double two_pow_64, int* reverse_order,
                                  int n_power);

    __global__ void
    encode_kernel_compose(COMPLEX* complex_message, Data* plaintext,
                          Modulus* modulus, Data* Mi_inv, Data* Mi,
                          Data* upper_half_threshold, Data* decryption_modulus,
                          int coeff_modulus_count, double scale,
                          double two_pow_64, int* reverse_order, int n_power);

} // namespace heongpu
#endif // ENCODING_H