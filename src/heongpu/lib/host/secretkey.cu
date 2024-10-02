// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "secretkey.cuh"

namespace heongpu
{
    __host__ Secretkey::Secretkey(Parameters& context)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n

        location = DeviceVector<Data>(coeff_modulus_count_ * ring_size_);
    }

    Data* Secretkey::data()
    {
        return location.data();
    }
} // namespace heongpu