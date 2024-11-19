﻿// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef UTIL_H
#define UTIL_H

#include "common.cuh"
#include "nttparameters.cuh"
#include <string>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "defines.h"
#include <unordered_map>
#include <stdexcept>

class CudaException_ : public std::exception
{
  public:
    CudaException_(const std::string& file, int line, cudaError_t error)
        : file_(file), line_(line), error_(error)
    {
    }

    const char* what() const noexcept override
    {
        return m_error_string.c_str();
    }

  private:
    std::string file_;
    int line_;
    cudaError_t error_;
    std::string m_error_string = "CUDA Error in " + file_ + " at line " +
                                 std::to_string(line_) + ": " +
                                 cudaGetErrorString(error_);
};

#define HEONGPU_CUDA_CHECK(err)                                                \
    do                                                                         \
    {                                                                          \
        cudaError_t error = err;                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            throw CudaException_(__FILE__, __LINE__, error);                   \
        }                                                                      \
    } while (0)

namespace heongpu
{
    //////////////////////////////////////////////////////////////////////////////////

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
        bgv = 0x3
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

    enum class storage_type : std::uint8_t
    {
        HOST = 0x1,
        DEVICE = 0x2
    };

    Data extendedGCD(Data a, Data b, Data& x, Data& y);
    Data modInverse(Data a, Data m);
    int countBits(Data number);

    bool is_power_of_two(size_t number);
    int calculate_bit_count(Data number);
    int calculate_big_integer_bit_count(Data* number, int word_count);

    bool miller_rabin(const Data& value, size_t num_rounds);

    bool is_prime(const Data& value);

    std::vector<Data> generate_proper_primes(Data factor, int bit_size,
                                             size_t count);

    std::vector<Modulus>
    generate_primes(size_t poly_modulus_degree,
                    const std::vector<int> prime_bit_sizes);

    std::vector<Modulus> generate_internal_primes(size_t poly_modulus_degree,
                                                  const int prime_count);

    bool is_primitive_root(Data root, size_t degree, Modulus& modulus);

    bool find_primitive_root(size_t degree, Modulus& modulus,
                             Data& destination);

    Data find_minimal_primitive_root(size_t degree, Modulus& modulus);

    std::vector<Data>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus> primes);

    std::vector<Root>
    generate_ntt_table(std::vector<Data> psi, std::vector<Modulus> primes,
                       int n_power); // bit reverse order for GPU-NTT

    std::vector<Root>
    generate_intt_table(std::vector<Data> psi, std::vector<Modulus> primes,
                        int n_power); // bit reverse order for GPU-NTT

    std::vector<Ninverse> generate_n_inverse(size_t poly_modulus_degree,
                                             std::vector<Modulus> primes);

    __global__ void unsigned_signed_convert(Data* input, Data* output,
                                            Modulus* modulus);

} // namespace heongpu
#endif // UTIL_H
