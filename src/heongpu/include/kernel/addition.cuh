// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HE_ADDITION_H
#define HE_ADDITION_H

#include <curand_kernel.h>
#include "context.cuh"

namespace heongpu
{
    // Homomorphic Addition Kernel
    __global__ void addition(Data64* in1, Data64* in2, Data64* out,
                             Modulus64* modulus, int n_power);

    // Homomorphic Substraction Kernel
    __global__ void substraction(Data64* in1, Data64* in2, Data64* out,
                                 Modulus64* modulus, int n_power);

    // Homomorphic Negation Kernel
    __global__ void negation(Data64* in1, Data64* out, Modulus64* modulus,
                             int n_power);

    // Homomorphic Plaintext Addition Kernel(BFV)
    __global__ void addition_plain_bfv_poly(Data64* cipher, Data64* plain,
                                            Data64* output, Modulus64* modulus,
                                            Modulus64 plain_mod, Data64 Q_mod_t,
                                            Data64 upper_threshold,
                                            Data64* coeffdiv_plain,
                                            int n_power);

    // Homomorphic Plaintext Substraction Kernel(BFV)
    __global__ void
    substraction_plain_bfv_poly(Data64* cipher, Data64* plain, Data64* output,
                                Modulus64* modulus, Modulus64 plain_mod,
                                Data64 Q_mod_t, Data64 upper_threshold,
                                Data64* coeffdiv_plain, int n_power);

    // Homomorphic Plaintext Addition Kernel(BFV)
    __global__ void addition_plain_bfv_poly_inplace(
        Data64* cipher, Data64* plain, Data64* output, Modulus64* modulus,
        Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
        Data64* coeffdiv_plain, int n_power);

    // Homomorphic Plaintext Substraction Kernel(BFV)
    __global__ void substraction_plain_bfv_poly_inplace(
        Data64* cipher, Data64* plain, Data64* output, Modulus64* modulus,
        Modulus64 plain_mod, Data64 Q_mod_t, Data64 upper_threshold,
        Data64* coeffdiv_plain, int n_power);

    // Homomorphic Plaintext Addition Kernel(CKKS)
    __global__ void addition_plain_ckks_poly(Data64* in1, Data64* in2,
                                             Data64* out, Modulus64* modulus,
                                             int n_power);

    // Homomorphic Plaintext Substraction Kernel(CKKS)
    __global__ void substraction_plain_ckks_poly(Data64* in1, Data64* in2,
                                                 Data64* out,
                                                 Modulus64* modulus,
                                                 int n_power);

} // namespace heongpu

#endif // HE_ADDITION_H