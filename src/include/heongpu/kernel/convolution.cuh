// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CONVOLUTION_H
#define HEONGPU_CONVOLUTION_H

#include "cuda_runtime.h"
#include "gpuntt/common/modular_arith.cuh"
#include <heongpu/kernel/defines.h>

namespace heongpu
{
    /**
     * @brief Pointwise multiply two RNS polynomials in NTT domain.
     *
     * Layout: [poly_id][rns_id][coeff_id], contiguous with stride (1<<n_power)
     * for coeff dimension.
     *
     * @param in1 First input, size = poly_count * rns_count * n.
     * @param in2 Second input, size = poly_count * rns_count * n.
     * @param out Output, size = poly_count * rns_count * n.
     * @param modulus RNS base, size = rns_count.
     */
    __global__ void rns_pointwise_multiply_kernel(Data64* in1, Data64* in2,
                                                 Data64* out,
                                                 Modulus64* modulus,
                                                 int n_power);

} // namespace heongpu

#endif // HEONGPU_CONVOLUTION_H

