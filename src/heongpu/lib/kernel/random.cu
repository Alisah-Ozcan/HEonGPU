// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "random.cuh"

namespace heongpu
{
    // Not cryptographically secure, will be fixed later.
    __global__ void modular_uniform_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y;

        int subsequence = idx + (block_y << n_power);
        curandState_t state;
        curand_init(seed, subsequence, offset, &state);

        int out_offset = (block_y * rns_mod_count) << n_power;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int in_offset = i << n_power;

            uint32_t rn_lo = curand(&state);
            uint32_t rn_hi = curand(&state);

            uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
                                static_cast<uint64_t>(rn_lo);
            Data64 rn_ULL = static_cast<Data64>(combined);
            rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[i]);

            output[idx + in_offset + out_offset] = rn_ULL;
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void modular_uniform_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y;

        int subsequence = idx + (block_y << n_power);
        curandState_t state;
        curand_init(seed, subsequence, offset, &state);

        int out_offset = (block_y * rns_mod_count) << n_power;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int in_offset = i << n_power;
            int index_mod = mod_index[i];

            uint32_t rn_lo = curand(&state);
            uint32_t rn_hi = curand(&state);

            uint64_t combined = (static_cast<uint64_t>(rn_hi) << 32) |
                                static_cast<uint64_t>(rn_lo);
            Data64 rn_ULL = static_cast<Data64>(combined);
            rn_ULL = OPERATOR_GPU_64::reduce_forced(rn_ULL, modulus[index_mod]);

            output[idx + in_offset + out_offset] = rn_ULL;
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void modular_gaussian_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y;

        int subsequence = idx + (block_y << n_power);
        curandState_t state;
        curand_init(seed, subsequence, offset, &state);

        float noise = curand_normal(&state);
        noise = noise * error_std_dev; // SIGMA

        uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

        int out_offset = (block_y * rns_mod_count) << n_power;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            Data64 rn_ULL =
                static_cast<Data64>(noise) + (flag & modulus[i].value);
            int in_offset = i << n_power;
            output[idx + in_offset + out_offset] = rn_ULL;
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void modular_gaussian_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes
        int block_y = blockIdx.y;

        int subsequence = idx + (block_y << n_power);
        curandState_t state;
        curand_init(seed, subsequence, offset, &state);

        float noise = curand_normal(&state);
        noise = noise * error_std_dev; // SIGMA

        uint64_t flag = static_cast<uint64_t>(-static_cast<int64_t>(noise < 0));

        int out_offset = (block_y * rns_mod_count) << n_power;
#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int index_mod = mod_index[i];
            Data64 rn_ULL =
                static_cast<Data64>(noise) + (flag & modulus[index_mod].value);
            int in_offset = i << n_power;
            output[idx + in_offset + out_offset] = rn_ULL;
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void modular_ternary_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        curandState_t state;
        curand_init(seed, idx, offset, &state);

        // TODO: make it efficient
        Data64 random_number = curand(&state) & 3; // 0,1,2,3
        if (random_number == 3)
        {
            random_number -= 3; // 0,1,2
        }

        uint64_t flag =
            static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int location = i << n_power;
            Data64 result = random_number;
            result = result + (flag & modulus[i].value) - 1;
            output[idx + location] = result;
        }
    }

    // Not cryptographically secure, will be fixed later.
    __global__ void modular_ternary_random_number_generation_kernel(
        Data64* output, Modulus64* modulus, int n_power, int rns_mod_count,
        int seed, int offset, int* mod_index)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // Ring Sizes

        curandState_t state;
        curand_init(seed, idx, offset, &state);

        // TODO: make it efficient
        Data64 random_number = curand(&state) & 3; // 0,1,2,3
        if (random_number == 3)
        {
            random_number -= 3; // 0,1,2
        }

        uint64_t flag =
            static_cast<uint64_t>(-static_cast<int64_t>(random_number == 0));

#pragma unroll
        for (int i = 0; i < rns_mod_count; i++)
        {
            int index_mod = mod_index[i];
            int location = i << n_power;
            Data64 result = random_number;
            result = result + (flag & modulus[index_mod].value) - 1;
            output[idx + location] = result;
        }
    }
} // namespace heongpu
