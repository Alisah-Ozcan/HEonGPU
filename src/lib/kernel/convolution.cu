// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/convolution.cuh>

namespace heongpu
{
    __global__ void rns_pointwise_multiply_kernel(Data64* in1, Data64* in2,
                                                 Data64* out,
                                                 Modulus64* modulus,
                                                 int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns id
        int block_z = blockIdx.z; // poly id

        int n = 1 << n_power;
        if (idx >= n)
        {
            return;
        }

        int location =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);

        out[location] =
            OPERATOR_GPU_64::mult(in1[location], in2[location], modulus[block_y]);
    }

} // namespace heongpu
