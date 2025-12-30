// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/kernel/transform.cuh>

namespace heongpu
{
    __global__ void negacyclic_shift_rns_kernel(const Data64* in, Data64* out,
                                                Modulus64* modulus, int shift,
                                                int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring_size
        int rns_id = blockIdx.y;

        int n = 1 << n_power;
        if (idx >= n)
        {
            return;
        }

        int dst = idx + shift;
        bool negate = false;
        if (dst >= n)
        {
            dst -= n;
            negate = true;
        }
        else if (dst < 0)
        {
            dst += n;
            negate = true;
        }

        const int in_loc = idx + (rns_id << n_power);
        const int out_loc = dst + (rns_id << n_power);

        Data64 val = in[in_loc];
        if (negate && (val != 0))
        {
            out[out_loc] = OPERATOR_GPU_64::sub(modulus[rns_id].value, val,
                                                modulus[rns_id]);
        }
        else
        {
            out[out_loc] = val;
        }
    }

} // namespace heongpu

