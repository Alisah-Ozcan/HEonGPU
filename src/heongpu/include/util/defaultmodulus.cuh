// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef DEFAULT_MODULUS_H
#define DEFAULT_MODULUS_H

#include "common.cuh"
#include "nttparameters.cuh"
#include <unordered_map>
#include <vector>

namespace heongpu
{
    /*
    The default modulus for different poly_modulus_degree values was determined
    based on security estimates from the lattice-estimator tool, with parameters
    selected to align with the desired security level.
    lattice-estimator: https://github.com/malb/lattice-estimator
    Reference: https://eprint.iacr.org/2015/046
    */
    namespace defaultparams
    {

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_128bit_sec_modulus();

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_192bit_sec_modulus();

        const std::unordered_map<std::size_t, std::vector<Modulus64>>&
        get_256bit_sec_modulus();

    } // namespace defaultparams
} // namespace heongpu
#endif // DEFAULT_MODULUS_H
