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

    __global__ void sk_multiplication(Data* ct1, Data* sk, Data* output,
                                      Modulus* modulus, int n_power,
                                      int decomp_mod_count);

    __global__ void sk_multiplicationx3(Data* ct1, Data* sk, Modulus* modulus,
                                        int n_power, int decomp_mod_count);

    __global__ void decryption_kernel(Data* ct0, Data* ct1, Data* plain,
                                      Modulus* modulus, Modulus plain_mod,
                                      Modulus gamma, Data* Qi_t, Data* Qi_gamma,
                                      Data* Qi_inverse, Data mulq_inv_t,
                                      Data mulq_inv_gamma, Data inv_gamma,
                                      int n_power, int decomp_mod_count);

    __global__ void decryption_kernelx3(Data* ct0, Data* ct1, Data* ct2,
                                        Data* plain, Modulus* modulus,
                                        Modulus plain_mod, Modulus gamma,
                                        Data* Qi_t, Data* Qi_gamma,
                                        Data* Qi_inverse, Data mulq_inv_t,
                                        Data mulq_inv_gamma, Data inv_gamma,
                                        int n_power, int decomp_mod_count);

    __global__ void coeff_multadd(Data* input1, Data* input2, Data* output,
                                  Modulus plain_mod, Modulus* modulus,
                                  int n_power, int decomp_mod_count);

    __global__ void compose_kernel(Data* input, Data* output, Modulus* modulus,
                                   Data* Mi_inv, Data* Mi,
                                   Data* decryption_modulus,
                                   int coeff_modulus_count, int n_power);

    // TODO: make it efficient with cooperative group
    __global__ void find_max_norm_kernel(Data* input, Data* output,
                                         Data* upper_half_threshold,
                                         Data* decryption_modulus,
                                         int coeff_modulus_count, int n_power);

    __global__ void sk_multiplication_ckks(Data* ciphertext, Data* plaintext,
                                           Data* sk, Modulus* modulus,
                                           int n_power, int decomp_mod_count);

} // namespace heongpu
#endif // DECRYPTION_H