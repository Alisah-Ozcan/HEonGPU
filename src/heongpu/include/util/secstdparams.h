// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef SECURITY_STANDART_PARAMETERS_H
#define SECURITY_STANDART_PARAMETERS_H

#include "common.cuh"

namespace heongpu
{
    /*
    The maximum bit size for Q_tilda​ based on the security estimates from the
    lattice-estimator tool is determined by considering the parameters that
    align with the desired security level. lattice-estimator:
    https://github.com/malb/lattice-estimator Reference:
    https://eprint.iacr.org/2015/046
    */

    // Standard deviation of error distribution
    constexpr double error_std_dev = 3.2;

    // 128 bit security (Ternary Secret, Standart Deviation = 3.2);
    constexpr int heongpu_128bit_std_parms(size_t poly_modulus_degree) noexcept
    {
        switch (poly_modulus_degree)
        {
            case 4096:
                return 109;
            case 8192:
                return 218;
            case 16384:
                return 438;
            case 32768:
                return 881;
            case 65536:
                return 1761;
        }
        return 0;
    }

    // 192 bit security (Ternary Secret, Standart Deviation = 3.2);
    constexpr int heongpu_192bit_std_parms(size_t poly_modulus_degree) noexcept
    {
        switch (poly_modulus_degree)
        {
            case 4096:
                return 74;
            case 8192:
                return 149;
            case 16384:
                return 300;
            case 32768:
                return 605;
            case 65536:
                return 1212;
        }
        return 0;
    }

    // 256 bit security (Ternary Secret, Standart Deviation = 3.2);
    constexpr int heongpu_256bit_std_parms(size_t poly_modulus_degree) noexcept
    {
        switch (poly_modulus_degree)
        {
            case 4096:
                return 57;
            case 8192:
                return 115;
            case 16384:
                return 232;
            case 32768:
                return 465;
            case 65536:
                return 930;
        }
        return 0;
    }

} // namespace heongpu
#endif // SECURITY_STANDART_PARAMETERS_H
