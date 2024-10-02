// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "publickey.cuh"

namespace heongpu
{
    __host__ Publickey::Publickey(Parameters& context)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n

        locations_ = DeviceVector<Data>(2 * coeff_modulus_count_ * ring_size_);
    }

    Data* Publickey::data()
    {
        return locations_.data();
    }
} // namespace heongpu