// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef RANDOM_GENERATOR_H
#define RANDOM_GENERATOR_H

#include <curand_kernel.h>
#include "context.cuh"

namespace heongpu
{
    __global__ void modular_uniform_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset);

    __global__ void modular_uniform_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index);

    __global__ void modular_gaussian_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset);

    __global__ void modular_gaussian_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index);

    __global__ void modular_ternary_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset);

    __global__ void modular_ternary_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index);

} // namespace heongpu
#endif // RANDOM_GENERATOR_H
