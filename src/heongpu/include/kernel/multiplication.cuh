// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HE_MULTIPLICATION_H
#define HE_MULTIPLICATION_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "context.cuh"

namespace heongpu
{
    // Homomorphic Multiplication Kernels

    __global__ void cross_multiplication(Data* in1, Data* in2, Data* out,
                                         Modulus* modulus, int n_power,
                                         int decomp_size);

    __global__ void
    fast_convertion(Data* in1, Data* in2, Data* out1, Modulus* ibase,
                    Modulus* obase, Modulus m_tilde,
                    Data inv_prod_q_mod_m_tilde, Data* inv_m_tilde_mod_Bsk,
                    Data* prod_q_mod_Bsk, Data* base_change_matrix_Bsk,
                    Data* base_change_matrix_m_tilde,
                    Data* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size);

    __global__ void
    fast_floor(Data* in_baseq_Bsk, Data* out1, Modulus* ibase, Modulus* obase,
               Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array,
               Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk,
               Data* inv_punctured_prod_mod_B_array, Data* base_change_matrix_q,
               Data* base_change_matrix_msk, Data inv_prod_B_mod_m_sk,
               Data* prod_B_mod_q, int n_power, int ibase_size, int obase_size);

    __global__ void threshold_kernel(Data* plain_in, Data* output,
                                     Modulus* modulus,
                                     Data* plain_upper_half_increment,
                                     Data plain_upper_half_threshold,
                                     int n_power, int decomp_size);

    __global__ void cipherplain_kernel(Data* cipher, Data* plain_in,
                                       Data* output, Modulus* modulus,
                                       int n_power, int decomp_size);

    __global__ void cipherplain_multiplication_kernel(Data* in1, Data* in2,
                                                      Data* out,
                                                      Modulus* modulus,
                                                      int n_power);

} // namespace heongpu
#endif // HE_MULTIPLICATION_H