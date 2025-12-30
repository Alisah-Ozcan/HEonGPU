// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TRANSFORM_H
#define HEONGPU_TRANSFORM_H

#include "cuda_runtime.h"
#include "gpuntt/common/modular_arith.cuh"
#include <heongpu/kernel/defines.h>

namespace heongpu
{
    /**
     * @brief Negacyclic monomial multiplication by X^shift in coefficient
     * domain: out(X) = in(X) * X^shift mod (X^N + 1).
     *
     * Layout: [rns][coeff], contiguous with stride (1<<n_power).
     */
    __global__ void negacyclic_shift_rns_kernel(const Data64* in, Data64* out,
                                                Modulus64* modulus, int shift,
                                                int n_power);

} // namespace heongpu

#endif // HEONGPU_TRANSFORM_H

