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
        in_ntt_domain_ = false;
    }

    Data* Publickey::data()
    {
        return locations_.data();
    }

    __host__ MultipartyPublickey::MultipartyPublickey(Parameters& context,
                                                      int seed)
        : Publickey(context), seed_(seed)
    {
    }

} // namespace heongpu