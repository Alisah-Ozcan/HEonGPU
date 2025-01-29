// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef SWITCHKEY_H
#define SWITCHKEY_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "ntt.cuh"
#include "context.cuh"

namespace heongpu
{

    __global__ void cipher_broadcast_kernel(Data64* input, Data64* output,
                                            Modulus64* modulus, int n_power,
                                            int rns_mod_count);
    __global__ void
    cipher_broadcast_leveled_kernel(Data64* input, Data64* output,
                                    Modulus64* modulus, int first_rns_mod_count,
                                    int current_rns_mod_count, int n_power);

    __global__ void multiply_accumulate_kernel(Data64* input, Data64* relinkey,
                                               Data64* output,
                                               Modulus64* modulus, int n_power,
                                               int decomp_mod_count);

    __global__ void
    multiply_accumulate_method_II_kernel(Data64* input, Data64* relinkey,
                                         Data64* output, Modulus64* modulus,
                                         int n_power, int Q_tilda_size, int d);

    __global__ void multiply_accumulate_leveled_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* modulus,
        int first_rns_mod_count, int current_decomp_mod_count, int n_power);

    __global__ void multiply_accumulate_leveled_method_II_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* modulus,
        int first_rns_mod_count, int current_decomp_mod_count,
        int current_rns_mod_count, int d, int level, int n_power);

    __global__ void
    divide_round_lastq_kernel(Data64* input, Data64* ct, Data64* output,
                              Modulus64* modulus, Data64* half,
                              Data64* half_mod, Data64* last_q_modinv,
                              int n_power, int decomp_mod_count);

    __global__ void divide_round_lastq_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int decomp_mod_count);

    __global__ void divide_round_lastq_leveled_stage_one_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, int n_power, int first_decomp_count,
        int current_decomp_count);

    __global__ void divide_round_lastq_leveled_stage_two_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count);

    __global__ void divide_round_lastq_leveled_stage_two_switchkey_kernel(
        Data64* input_last, Data64* input, Data64* ct, Data64* output,
        Modulus64* modulus, Data64* last_q_modinv, int n_power,
        int current_decomp_count);

    __global__ void move_cipher_leveled_kernel(Data64* input, Data64* output,
                                               int n_power,
                                               int current_decomp_count);

    __global__ void divide_round_lastq_rescale_kernel(
        Data64* input_last, Data64* input, Data64* output, Modulus64* modulus,
        Data64* last_q_modinv, int n_power, int current_decomp_count);

    __global__ void base_conversion_DtoB_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int l, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_DtoQtilde_relin_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int l, int Q_tilda, int d);

    __global__ void base_conversion_DtoB_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Modulus64* B_base, Data64* base_change_matrix_D_to_B,
        Data64* Mi_inv_D_to_B, Data64* prod_D_to_B, int* I_j_, int* I_location_,
        int n_power, int d_tilda, int d, int r_prime, int* mod_index);

    __global__ void base_conversion_DtoQtilde_relin_leveled_kernel(
        Data64* ciphertext, Data64* output, Modulus64* modulus,
        Data64* base_change_matrix_D_to_Qtilda, Data64* Mi_inv_D_to_Qtilda,
        Data64* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power,
        int d, int current_Qtilda_size, int current_Q_size, int level,
        int* mod_index);

    __global__ void multiply_accumulate_extended_kernel(
        Data64* input, Data64* relinkey, Data64* output, Modulus64* B_prime,
        int n_power, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_BtoD_relin_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime);

    __global__ void base_conversion_BtoD_relin_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Modulus64* B_base,
        Data64* base_change_matrix_B_to_D, Data64* Mi_inv_B_to_D,
        Data64* prod_B_to_D, int* I_j_, int* I_location_, int n_power,
        int l_tilda, int d_tilda, int d, int r_prime, int* mod_index);

    __global__ void divide_round_lastq_extended_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size);

    __global__ void divide_round_lastq_extended_switchkey_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int n_power,
        int Q_prime_size, int Q_size, int P_size);

    __global__ void divide_round_lastq_extended_leveled_kernel(
        Data64* input, Data64* output, Modulus64* modulus, Data64* half,
        Data64* half_mod, Data64* last_q_modinv, int n_power, int Q_prime_size,
        int Q_size, int first_Q_prime_size, int first_Q_size, int P_size);

    // TODO: Find efficient way!
    __global__ void global_memory_replace_kernel(Data64* input, Data64* output,
                                                 int n_power);

    __global__ void
    global_memory_replace_offset_kernel(Data64* input, Data64* output,
                                        int current_decomposition_count,
                                        int n_power);

    __global__ void
    cipher_broadcast_switchkey_kernel(Data64* cipher, Data64* out0,
                                      Data64* out1, Modulus64* modulus,
                                      int n_power, int decomp_mod_count);

    __global__ void cipher_broadcast_switchkey_method_II_kernel(
        Data64* cipher, Data64* out0, Data64* out1, Modulus64* modulus,
        int n_power, int decomp_mod_count);

    __global__ void addition_switchkey(Data64* in1, Data64* in2, Data64* out,
                                       Modulus64* modulus, int n_power);

    __global__ void cipher_broadcast_switchkey_leveled_kernel(
        Data64* cipher, Data64* out0, Data64* out1, Modulus64* modulus,
        int n_power, int first_rns_mod_count, int current_rns_mod_count,
        int current_decomp_mod_count);

    __global__ void negacyclic_shift_poly_coeffmod_kernel(Data64* cipher_in,
                                                          Data64* cipher_out,
                                                          Modulus64* modulus,
                                                          int shift,
                                                          int n_power);

    ////////////////////////////////////////
    // Optimized Hoisting-Rotations

    __global__ void ckks_duplicate_kernel(Data64* cipher, Data64* output,
                                          Modulus64* modulus, int n_power,
                                          int first_rns_mod_count,
                                          int current_rns_mod_count,
                                          int current_decomp_mod_count);

    __global__ void bfv_duplicate_kernel(Data64* cipher, Data64* output1,
                                         Data64* output2, Modulus64* modulus,
                                         int n_power, int rns_mod_count);

    __global__ void divide_round_lastq_permute_ckks_kernel(
        Data64* input, Data64* input2, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int first_Q_prime_size,
        int first_Q_size, int P_size);

    __global__ void divide_round_lastq_permute_bfv_kernel(
        Data64* input, Data64* ct, Data64* output, Modulus64* modulus,
        Data64* half, Data64* half_mod, Data64* last_q_modinv, int galois_elt,
        int n_power, int Q_prime_size, int Q_size, int P_size);

} // namespace heongpu
#endif // SWITCHKEY_H