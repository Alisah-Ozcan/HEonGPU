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

    __global__ void secretkey_gen_kernel(int* secret_key, int hamming_weight,
                                         int n_power, int seed);

    __global__ void secretkey_rns_kernel(int* input, Data* output,
                                         Modulus* modulus, int n_power,
                                         int rns_mod_count, int seed);

    // Public Key Generation

    __global__ void publickey_gen_kernel(Data* public_key, Data* secret_key,
                                         Data* error_poly, Data* a_poly,
                                         Modulus* modulus, int n_power,
                                         int rns_mod_count);

    __global__ void threshold_pk_addition(Data* pk1, Data* pk2, Data* pkout,
                                          Modulus* modulus, int n_power,
                                          bool first);

    // Relinearization Key Generation

    __global__ void relinkey_gen_kernel(Data* relin_key, Data* secret_key,
                                        Data* error_poly, Data* a_poly,
                                        Modulus* modulus, Data* factor,
                                        int n_power, int rns_mod_count);

    __global__ void multi_party_relinkey_piece_method_I_stage_I_kernel(
        Data* rk, Data* sk, Data* a, Data* u, Data* e0, Data* e1,
        Modulus* modulus, Data* factor, int n_power, int rns_mod_count);

    __global__ void multi_party_relinkey_piece_method_I_II_stage_II_kernel(
        Data* rk_1, Data* rk_2, Data* sk, Data* u, Data* e0, Data* e1,
        Modulus* modulus, int n_power, int rns_mod_count, int decomp_mod_count);

    __global__ void multi_party_relinkey_piece_method_II_stage_I_kernel(
        Data* rk, Data* sk, Data* a, Data* u, Data* e0, Data* e1,
        Modulus* modulus, Data* factor, int* Sk_pair, int n_power, int l_tilda,
        int d, int Q_size, int P_size);

    __global__ void multi_party_relinkey_method_I_stage_I_kernel(
        Data* rk_in, Data* rk_out, Modulus* modulus, int n_power,
        int Q_prime_size, int l, bool first);

    __global__ void multi_party_relinkey_method_I_stage_II_kernel(
        Data* rk1, Data* rk2, Data* rk_out, Modulus* modulus, int n_power,
        int Q_prime_size, int l);

    __global__ void
    multi_party_relinkey_method_I_stage_II_kernel(Data* rk_in, Data* rk_out,
                                                  Modulus* modulus, int n_power,
                                                  int Q_prime_size, int l);

    __global__ void relinkey_gen_II_kernel(Data* relin_key_temp,
                                           Data* secret_key, Data* error_poly,
                                           Data* a_poly, Modulus* modulus,
                                           Data* factor, int* Sk_pair,
                                           int n_power, int l_tilda, int d,
                                           int Q_size, int P_size);

    __global__ void relinkey_gen_II_leveled_kernel(
        Data* relin_key_temp, Data* secret_key, Data* error_poly, Data* a_poly,
        Modulus* modulus, Data* factor, int* Sk_pair, int n_power, int l_tilda,
        int d, int Q_size, int P_size, int* mod_index);

    __global__ void relinkey_DtoB_kernel(Data* relin_key_temp, Data* relin_key,
                                         Modulus* modulus, Modulus* B_base,
                                         Data* base_change_matrix_D_to_B,
                                         Data* Mi_inv_D_to_B, Data* prod_D_to_B,
                                         int* I_j_, int* I_location_,
                                         int n_power, int l_tilda, int d_tilda,
                                         int d, int r_prime);

    __global__ void relinkey_DtoB_leveled_kernel(
        Data* relin_key_temp, Data* relin_key, Modulus* modulus,
        Modulus* B_base, Data* base_change_matrix_D_to_B, Data* Mi_inv_D_to_B,
        Data* prod_D_to_B, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index);

    // Galois Key Generation

    int steps_to_galois_elt(int steps, int coeff_count, int group_order);

    __device__ int bitreverse_gpu(int index, int n_power);

    __device__ int permutation(int index, int galois_elt, int coeff_count,
                               int n_power);

    __global__ void galoiskey_gen_kernel(Data* galois_key, Data* secret_key,
                                         Data* error_poly, Data* a_poly,
                                         Modulus* modulus, Data* factor,
                                         int galois_elt, int n_power,
                                         int rns_mod_count);

    __global__ void galoiskey_gen_II_kernel(
        Data* galois_key_temp, Data* secret_key, Data* error_poly, Data* a_poly,
        Modulus* modulus, Data* factor, int galois_elt, int* Sk_pair,
        int n_power, int l_tilda, int d, int Q_size, int P_size);

    __global__ void multi_party_galoiskey_gen_method_I_II_kernel(
        Data* gk_1, Data* gk_2, Modulus* modulus, int n_power,
        int rns_mod_count, int decomp_mod_count, bool first);

    // Switch Key Generation

    __global__ void switchkey_gen_kernel(Data* switch_key, Data* new_secret_key,
                                         Data* old_secret_key, Data* error_poly,
                                         Data* a_poly, Modulus* modulus,
                                         Data* factor, int n_power,
                                         int rns_mod_count);

    __global__ void switchkey_gen_II_kernel(
        Data* switch_key, Data* new_secret_key, Data* old_secret_key,
        Data* error_poly, Data* a_poly, Modulus* modulus, Data* factor,
        int* Sk_pair, int n_power, int l_tilda, int d, int Q_size, int P_size);

    __global__ void switchkey_kernel(Data* switch_key, Data* new_secret_key,
                                     Data* old_secret_key, Data* e_a,
                                     Modulus* modulus, Data* factor,
                                     int n_power, int rns_mod_count);

} // namespace heongpu
#endif // KEYGENERATION_H