// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_DECRYPTION_H
#define HEONGPU_DECRYPTION_H

#include <curand_kernel.h>
#include "modular_arith.cuh"
#include "bigintegerarith.cuh"
#include "util.cuh"

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

    //////////////////////////////////////////////////////////////////////////////////

    __global__ void decrypt_lwe_kernel(int32_t* sk, int32_t* input_a,
                                       int32_t* input_b, int32_t* output, int n,
                                       int k);

    //////////////////////////////////////////////////////////////////////////////////

    __global__ void col_boot_dec_mul_with_sk(const Data64* ct1, const Data64* a,
                                             const Data64* sk, Data64* output,
                                             const Modulus64* modulus,
                                             int n_power, int decomp_mod_count);

    __global__ void col_boot_add_random_and_errors(
        Data64* ct, const Data64* errors, const Data64* random_plain,
        const Modulus64* modulus, Modulus64 plain_mod, Data64 Q_mod_t,
        Data64 upper_threshold, Data64* coeffdiv_plain, int n_power,
        int decomp_mod_count);

    __global__ void col_boot_enc(Data64* ct, const Data64* h,
                                 const Data64* random_plain,
                                 const Modulus64* modulus, Modulus64 plain_mod,
                                 Data64 Q_mod_t, Data64 upper_threshold,
                                 Data64* coeffdiv_plain, int n_power,
                                 int decomp_mod_count);

    __global__ void col_boot_dec_mul_with_sk_ckks(
        const Data64* ct1, const Data64* a, const Data64* sk, Data64* output,
        const Modulus64* modulus, int n_power, int decomp_mod_count,
        int current_decomp_mod_count);

    __global__ void col_boot_add_random_and_errors_ckks(
        Data64* ct, const Data64* error0, const Data64* error1,
        const Data64* random_plain, const Modulus64* modulus, int n_power,
        int decomp_mod_count, int current_decomp_mod_count);

} // namespace heongpu
#endif // HEONGPU_DECRYPTION_H