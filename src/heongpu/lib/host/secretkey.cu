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

        hamming_weight_ = ring_size_ >> 1; // default
        in_ntt_domain_ = false;
    }

    __host__ Secretkey::Secretkey(Parameters& context, int hamming_weight)
    {
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n

        hamming_weight_ = hamming_weight;
        if ((hamming_weight_ <= 0) || (hamming_weight_ > ring_size_))
        {
            throw std::invalid_argument(
                "hamming weight has to be in range 0 to ring size.");
        }

        in_ntt_domain_ = false;
    }

    Data* Secretkey::data()
    {
        return location_.data();
    }
} // namespace heongpu