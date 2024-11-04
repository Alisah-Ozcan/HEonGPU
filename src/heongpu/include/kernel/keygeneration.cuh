// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef KEYGENERATION_H
#define KEYGENERATION_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "context.cuh"

namespace heongpu
{

    // Secret Key Generation

    __global__ void sk_gen_kernel(int* secret_key, int hamming_weight,  int n_power, int seed);

    __global__ void sk_rns_kernel(int* input, Data* output, Modulus* modulus, int n_power,
                              int rns_mod_count, int seed);

    // Public Key Generation

    __global__ void error_kernel(Data* a_e, Modulus* modulus, int n_power,
                                 int rns_mod_count, int seed);

    __global__ void error_kernel_leveled(Data* a_e, Modulus* modulus,
                                         int n_power, int mod_count,
                                         int* mod_index, int seed);

    __global__ void pk_kernel(Data* public_key, Data* secret_key, Data* e_a,
                              Modulus* modulus, int n_power, int rns_mod_count);

    // Relinearization Key Generation

    __global__ void relinkey_kernel(Data* relin_key, Data* secret_key,
                                    Data* e_a, Modulus* modulus, Data* factor,
                                    int n_power, int rns_mod_count);

    __global__ void
    relinkey_kernel_externel_product(Data* relin_key_temp, Data* secret_key,
                                     Data* e_a, Modulus* modulus, Data* factor,
                                     int* Sk_pair, int n_power, int l_tilda,
                                     int d, int Q_size, int P_size);

    __global__ void relinkey_kernel_externel_product_leveled(
        Data* relin_key_temp, Data* secret_key, Data* e_a, Modulus* modulus,
        Data* factor, int* Sk_pair, int n_power, int l_tilda, int d, int Q_size,
        int P_size, int* mod_index);

    __global__ void relinkey_DtoB_kernel(Data* relin_key_temp, Data* relin_key,
                                         Modulus* modulus, Modulus* B_base,
                                         Data* base_change_matrix_D_to_B,
                                         Data* Mi_inv_D_to_B, Data* prod_D_to_B,
                                         int* I_j_, int* I_location_,
                                         int n_power, int l_tilda, int d_tilda,
                                         int d, int r_prime);

    __global__ void relinkey_DtoB_kernel_leveled2(
        Data* relin_key_temp, Data* relin_key, Modulus* modulus,
        Modulus* B_base, Data* base_change_matrix_D_to_B, Data* Mi_inv_D_to_B,
        Data* prod_D_to_B, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index);

    // Galois Key Generation

    int steps_to_galois_elt(int steps, int coeff_count);

    __device__ int bitreverse_gpu(int index, int n_power);

    __device__ int permutation(int index, int galois_elt, int coeff_count,
                               int n_power);

    __global__ void galoiskey_method_I_kernel(Data* galois_key,
                                              Data* secret_key, Data* e_a,
                                              Modulus* modulus, Data* factor,
                                              int galois_elt, int n_power,
                                              int rns_mod_count);

    __global__ void galoiskey_method_II_kernel(Data* galois_key_temp,
                                               Data* secret_key, Data* e_a,
                                               Modulus* modulus, Data* factor,
                                               int galois_elt, int* Sk_pair,
                                               int n_power, int l_tilda, int d,
                                               int Q_size, int P_size);

    // witch Key Generation

    __global__ void switchkey_kernel(Data* switch_key, Data* new_secret_key,
                                     Data* old_secret_key, Data* e_a,
                                     Modulus* modulus, Data* factor,
                                     int n_power, int rns_mod_count);

    __global__ void switchkey_kernel_method_II(
        Data* switch_key, Data* new_secret_key, Data* old_secret_key, Data* e_a,
        Modulus* modulus, Data* factor, int* Sk_pair, int n_power, int l_tilda,
        int d, int Q_size, int P_size);

} // namespace heongpu
#endif // KEYGENERATION_H