// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef DECRYPTION_H
#define DECRYPTION_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "context.cuh"
#include "bigintegerarith.cuh"

namespace heongpu
{

    __global__ void sk_multiplication(Data64* ct1, Data64* sk, Data64* output,
                                      Modulus64* modulus, int n_power,
                                      int decomp_mod_count);

    __global__ void sk_multiplicationx3(Data64* ct1, Data64* sk,
                                        Modulus64* modulus, int n_power,
                                        int decomp_mod_count);

    __global__ void decryption_kernel(Data64* ct0, Data64* ct1, Data64* plain,
                                      Modulus64* modulus, Modulus64 plain_mod,
                                      Modulus64 gamma, Data64* Qi_t,
                                      Data64* Qi_gamma, Data64* Qi_inverse,
                                      Data64 mulq_inv_t, Data64 mulq_inv_gamma,
                                      Data64 inv_gamma, int n_power,
                                      int decomp_mod_count);

    __global__ void decryption_kernelx3(Data64* ct0, Data64* ct1, Data64* ct2,
                                        Data64* plain, Modulus64* modulus,
                                        Modulus64 plain_mod, Modulus64 gamma,
                                        Data64* Qi_t, Data64* Qi_gamma,
                                        Data64* Qi_inverse, Data64 mulq_inv_t,
                                        Data64 mulq_inv_gamma, Data64 inv_gamma,
                                        int n_power, int decomp_mod_count);

    __global__ void coeff_multadd(Data64* input1, Data64* input2,
                                  Data64* output, Modulus64 plain_mod,
                                  Modulus64* modulus, int n_power,
                                  int decomp_mod_count);

    __global__ void compose_kernel(Data64* input, Data64* output,
                                   Modulus64* modulus, Data64* Mi_inv,
                                   Data64* Mi, Data64* decryption_modulus,
                                   int coeff_modulus_count, int n_power);

    // TODO: make it efficient with cooperative group
    __global__ void find_max_norm_kernel(Data64* input, Data64* output,
                                         Data64* upper_half_threshold,
                                         Data64* decryption_modulus,
                                         int coeff_modulus_count, int n_power);

    __global__ void sk_multiplication_ckks(Data64* ciphertext,
                                           Data64* plaintext, Data64* sk,
                                           Modulus64* modulus, int n_power,
                                           int decomp_mod_count);

    __global__ void decryption_fusion_bfv_kernel(
        Data64* ct, Data64* plain, Modulus64* modulus, Modulus64 plain_mod,
        Modulus64 gamma, Data64* Qi_t, Data64* Qi_gamma, Data64* Qi_inverse,
        Data64 mulq_inv_t, Data64 mulq_inv_gamma, Data64 inv_gamma, int n_power,
        int decomp_mod_count);

} // namespace heongpu
#endif // DECRYPTION_H