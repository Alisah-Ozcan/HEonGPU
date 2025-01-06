// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef ENCRYPTION_H
#define ENCRYPTION_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "context.cuh"

namespace heongpu
{
    __global__ void pk_u_kernel(Data* pk, Data* u, Data* pk_u, Modulus* modulus,
                                int n_power, int rns_mod_count);

    __global__ void enc_div_lastq_bfv_kernel(
        Data* pk, Data* e, Data* plain, Data* ct, Modulus* modulus, Data* half,
        Data* half_mod, Data* last_q_modinv, Modulus plain_mod, Data Q_mod_t,
        Data upper_threshold, Data* coeffdiv_plain, int n_power,
        int Q_prime_size, int Q_size, int P_size);

    __global__ void enc_div_lastq_ckks_kernel(Data* pk, Data* e, Data* ct,
                                              Modulus* modulus, Data* half,
                                              Data* half_mod,
                                              Data* last_q_modinv, int n_power,
                                              int Q_prime_size, int Q_size,
                                              int P_size);

    __global__ void cipher_message_add_kernel(Data* ciphertext, Data* plaintext,
                                              Modulus* modulus, int n_power);

} // namespace heongpu

#endif // ENCRYPTION_H