// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_SCHEMES_H
#define HEONGPU_SCHEMES_H

#include "defines.h"

namespace heongpu
{
    enum class Scheme
    {
        BFV,
        CKKS,
        TFHE
    };

    template <Scheme S> class Ciphertext;

    template <Scheme S> class HEContext;

    template <Scheme S> class HEDecryptor;

    template <Scheme S> class HEEncoder;

    template <Scheme S> class HEEncryptor;

    template <Scheme S> class Relinkey;

    template <Scheme S> class MultipartyRelinkey;

    template <Scheme S> class Galoiskey;

    template <Scheme S> class MultipartyGaloiskey;

    template <Scheme S> class Switchkey;

    template <Scheme S> class Bootstrappingkey;

    template <Scheme S> class HEKeyGenerator;

    template <Scheme S> class HEOperator;

    template <Scheme S> class HEArithmeticOperator;

    template <Scheme S> class HELogicOperator;

    template <Scheme S> class Plaintext;

    template <Scheme S> class Publickey;

    template <Scheme S> class MultipartyPublickey;

    template <Scheme S> class Secretkey;

    template <Scheme S> class HEMultiPartyManager;

    // Describes the type of encryption scheme to be used.
    enum class scheme_type : std::uint8_t
    {
        // No scheme set; cannot be used for encryption
        none = 0x0,

        // Brakerski/Fan-Vercauteren scheme
        bfv = 0x1,

        // Cheon-Kim-Kim-Song scheme
        ckks = 0x2,

        // Brakerski-Gentry-Vaikuntanathan scheme
        bgv = 0x3,

        // Fast Fully Homomorphic Encryption over the Torus
        tfhe = 0x4
    };

    enum class sec_level_type : std::uint8_t
    {
        // No security level specified.
        none = 0x0,

        // 128 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec128 = 0x1,

        // 192 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec192 = 0x2,

        // 256 bits security level specified according to lattice-estimator:
        // https://github.com/malb/lattice-estimator.
        sec256 = 0x3
    };

    enum class keyswitching_type : std::uint8_t
    {
        NONE = 0x0,
        KEYSWITCHING_METHOD_I = 0x1, // SEALMETHOD = 0x1,
        KEYSWITCHING_METHOD_II = 0x2, // EXTERNALPRODUCT = 0x2,
        KEYSWITCHING_METHOD_III = 0x3, // EXTERNALPRODUCT_2 = 0x3
    };

    enum class arithmetic_bootstrapping_type : std::uint8_t
    {
        NONE = 0x0,
        REGULAR_BOOTSTRAPPING = 0x1,
        SLIM_BOOTSTRAPPING = 0x2,
    };

    enum class logic_bootstrapping_type : std::uint8_t
    {
        NONE = 0x0,
        BIT_BOOTSTRAPPING = 0x1, // scale = q0 / 2. More detail:
                                 // https://eprint.iacr.org/2024/767.pdf
        GATE_BOOTSTRAPPING = 0x2, // scale = q0 / 3. More detail:
                                  // https://eprint.iacr.org/2024/767.pdf
    };

} // namespace heongpu
#endif // HEONGPU_SCHEMES_H
