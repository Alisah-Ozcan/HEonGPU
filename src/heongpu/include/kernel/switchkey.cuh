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

    __global__ void CipherBroadcast2(Data* input, Data* output,
                                     Modulus* modulus, int n_power,
                                     int rns_mod_count);
    __global__ void CipherBroadcast2_leveled(Data* input, Data* output,
                                             Modulus* modulus,
                                             int first_rns_mod_count,
                                             int current_rns_mod_count,
                                             int n_power);

    __global__ void MultiplyAcc(Data* input, Data* relinkey, Data* output,
                                Modulus* modulus, int n_power,
                                int decomp_mod_count);

    __global__ void MultiplyAcc_method2(Data* input, Data* relinkey,
                                        Data* output, Modulus* modulus,
                                        int n_power, int Q_tilda_size, int d);

    __global__ void MultiplyAcc2_leveled(Data* input, Data* relinkey,
                                         Data* output, Modulus* modulus,
                                         int first_rns_mod_count,
                                         int current_decomp_mod_count,
                                         int n_power);

    __global__ void MultiplyAcc2_leveled_method2(Data* input, Data* relinkey,
                                                 Data* output, Modulus* modulus,
                                                 int first_rns_mod_count,
                                                 int current_decomp_mod_count,
                                                 int current_rns_mod_count,
                                                 int d, int level, int n_power);

    __global__ void DivideRoundLastq(Data* input, Data* ct, Data* output,
                                     Modulus* modulus, Data half,
                                     Data* half_mod, Data* last_q_modinv,
                                     int n_power, int decomp_mod_count);

    __global__ void DivideRoundLastq_(Data* input, Data* ct, Data* output,
                                      Modulus* modulus, Data* half,
                                      Data* half_mod, Data* last_q_modinv,
                                      int n_power, int decomp_mod_count);

    __global__ void DivideRoundLastq_ckks1_leveled(Data* input, Data* output,
                                                   Modulus* modulus, Data* half,
                                                   Data* half_mod, int n_power,
                                                   int first_decomp_count,
                                                   int current_decomp_count);
    __global__ void DivideRoundLastq_ckks2_leveled(
        Data* input_last, Data* input, Data* ct, Data* output, Modulus* modulus,
        Data* last_q_modinv, int n_power, int current_decomp_count);

    // For rescale
    __global__ void move_cipher_ckks_leveled(Data* input, Data* output,
                                             int n_power,
                                             int current_decomp_count);
    __global__ void DivideRoundLastq_rescale_ckks2_leveled(
        Data* input_last, Data* input, Data* output, Modulus* modulus,
        Data* last_q_modinv, int n_power, int current_decomp_count);

    __global__ void apply_galois_kernel(Data* cipher, Data* out0, Data* out1,
                                        Modulus* modulus, int galois_elt,
                                        int n_power, int decomp_mod_count);

    __global__ void apply_galois_method_II_kernel(Data* cipher, Data* output0,
                                                  Data* output1,
                                                  Modulus* modulus,
                                                  int galois_elt, int n_power,
                                                  int decomp_mod_count);

    __global__ void apply_galois_ckks_kernel(Data* cipher, Data* out0,
                                             Data* out1, Modulus* modulus,
                                             int galois_elt, int n_power,
                                             int first_rns_mod_count,
                                             int current_rns_mod_count,
                                             int current_decomp_mod_count);

    ////////////////////////////
    __global__ void relin_DtoB_kernel(Data* ciphertext, Data* output,
                                      Modulus* modulus, Modulus* B_base,
                                      Data* base_change_matrix_D_to_B,
                                      Data* Mi_inv_D_to_B, Data* prod_D_to_B,
                                      int* I_j_, int* I_location_, int n_power,
                                      int l, int d_tilda, int d, int r_prime);

    __global__ void relin_DtoQtilde_kernel(Data* ciphertext, Data* output,
                                           Modulus* modulus,
                                           Data* base_change_matrix_D_to_Qtilda,
                                           Data* Mi_inv_D_to_Qtilda,
                                           Data* prod_D_to_Qtilda, int* I_j_,
                                           int* I_location_, int n_power, int l,
                                           int Q_tilda, int d);

    __global__ void
    relin_DtoB_kernel_leveled2(Data* ciphertext, Data* output, Modulus* modulus,
                               Modulus* B_base, Data* base_change_matrix_D_to_B,
                               Data* Mi_inv_D_to_B, Data* prod_D_to_B,
                               int* I_j_, int* I_location_, int n_power,
                               int d_tilda, int d, int r_prime, int* mod_index);

    __global__ void relin_DtoQtilda_kernel_leveled2(
        Data* ciphertext, Data* output, Modulus* modulus,
        Data* base_change_matrix_D_to_Qtilda, Data* Mi_inv_D_to_Qtilda,
        Data* prod_D_to_Qtilda, int* I_j_, int* I_location_, int n_power, int d,
        int current_Qtilda_size, int current_Q_size, int level, int* mod_index);

    __global__ void MultiplyAcc_new(Data* input, Data* relinkey, Data* output,
                                    Modulus* B_prime, int n_power, int d_tilda,
                                    int d, int r_prime);

    __global__ void relin_BtoD_kernel(Data* input, Data* output,
                                      Modulus* modulus, Modulus* B_base,
                                      Data* base_change_matrix_B_to_D,
                                      Data* Mi_inv_B_to_D, Data* prod_B_to_D,
                                      int* I_j_, int* I_location_, int n_power,
                                      int l, int d_tilda, int d, int r_prime);

    __global__ void
    relin_BtoD_kernelNewP(Data* input, Data* output, Modulus* modulus,
                          Modulus* B_base, Data* base_change_matrix_B_to_D,
                          Data* Mi_inv_B_to_D, Data* prod_B_to_D, int* I_j_,
                          int* I_location_, int n_power, int l_tilda,
                          int d_tilda, int d, int r_prime);

    __global__ void relin_BtoD_kernelNewP_leveled2(
        Data* input, Data* output, Modulus* modulus, Modulus* B_base,
        Data* base_change_matrix_B_to_D, Data* Mi_inv_B_to_D, Data* prod_B_to_D,
        int* I_j_, int* I_location_, int n_power, int l_tilda, int d_tilda,
        int d, int r_prime, int* mod_index);

    __global__ void DivideRoundLastqNewP(Data* input, Data* ct, Data* output,
                                         Modulus* modulus, Data* half,
                                         Data* half_mod, Data* last_q_modinv,
                                         int n_power, int Q_prime_size,
                                         int Q_size, int P_size);

    __global__ void DivideRoundLastq_ckks(Data* input, Data* output,
                                          Modulus* modulus, Data half,
                                          Data* half_mod, Data* last_q_modinv,
                                          int n_power, int decomp_mod_count);

    __global__ void cipher_temp_add(Data* input1, Data* input2, Data* output,
                                    Modulus* modulus, int n_power);

    __global__ void DivideRoundLastqNewP_external_ckks(
        Data* input, Data* output, Modulus* modulus, Data* half, Data* half_mod,
        Data* last_q_modinv, int n_power, int Q_prime_size, int Q_size,
        int first_Q_prime_size, int first_Q_size, int P_size);

    // TODO: Find efficient way!
    __global__ void global_memory_replace(Data* input, Data* output,
                                          int n_power);

    __global__ void global_memory_replace_2(Data* input, Data* output,
                                            int current_decomposition_count,
                                            int n_power);

    __global__ void
    CipherBroadcast_switchkey_bfv_method_I(Data* cipher, Data* out0, Data* out1,
                                           Modulus* modulus, int n_power,
                                           int decomp_mod_count);

    __global__ void
    CipherBroadcast_switchkey_bfv_method_II(Data* cipher, Data* out0,
                                            Data* out1, Modulus* modulus,
                                            int n_power, int decomp_mod_count);

    __global__ void CipherBroadcast_switchkey_ckks_method_I(
        Data* cipher, Data* out0, Data* out1, Modulus* modulus, int n_power,
        int first_rns_mod_count, int current_rns_mod_count,
        int current_decomp_mod_count);

    __global__ void negacyclic_shift_poly_coeffmod_kernel(Data* cipher_in,
                                                          Data* cipher_out,
                                                          Modulus* modulus,
                                                          int shift,
                                                          int n_power);

} // namespace heongpu
#endif // SWITCHKEY_H