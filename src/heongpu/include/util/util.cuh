// Copyright 2024 Alişah Özcan
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
#include <set>
#include "storagemanager.cuh"

namespace heongpu
{

    class CudaException : public std::exception
    {
      public:
        CudaException(const std::string& file, int line, cudaError_t error)
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
            throw CudaException(__FILE__, __LINE__, error);                    \
        }                                                                      \
    } while (0)

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

    struct BootstrappingConfig
    {
        int CtoS_piece_; // Default: 3
        int StoC_piece_; // Default: 3
        int taylor_number_; // Default: 11
        bool less_key_mode_; // Default: false

        BootstrappingConfig(int CtoS = 3, int StoC = 3, int taylor = 11,
                            bool less_key_mode = false);

      private:
        void validate(); // Validates the configuration input values
    };

    enum class logic_bootstrapping_type : std::uint8_t
    {
        NONE = 0x0,
        BIT_BOOTSTRAPPING = 0x1, // scale = q0 / 2. More detail:
                                 // https://eprint.iacr.org/2024/767.pdf
        GATE_BOOTSTRAPPING = 0x2, // scale = q0 / 3. More detail:
                                  // https://eprint.iacr.org/2024/767.pdf
    };

    Data64 extendedGCD(Data64 a, Data64 b, Data64& x, Data64& y);
    Data64 modInverse(Data64 a, Data64 m);
    int countBits(Data64 number);

    bool is_power_of_two(size_t number);
    int calculate_bit_count(Data64 number);
    int calculate_big_integer_bit_count(Data64* number, int word_count);

    bool miller_rabin(const Data64& value, size_t num_rounds);

    bool is_prime(const Data64& value);

    std::vector<Data64> generate_proper_primes(Data64 factor, int bit_size,
                                               size_t count);

    std::vector<Modulus64>
    generate_primes(size_t poly_modulus_degree,
                    const std::vector<int> prime_bit_sizes);

    std::vector<Modulus64> generate_internal_primes(size_t poly_modulus_degree,
                                                    const int prime_count);

    bool is_primitive_root(Data64 root, size_t degree, Modulus64& modulus);

    bool find_primitive_root(size_t degree, Modulus64& modulus,
                             Data64& destination);

    Data64 find_minimal_primitive_root(size_t degree, Modulus64& modulus);

    std::vector<Data64>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus64> primes);

    std::vector<Root64>
    generate_ntt_table(std::vector<Data64> psi, std::vector<Modulus64> primes,
                       int n_power); // bit reverse order for GPU-NTT

    std::vector<Root64>
    generate_intt_table(std::vector<Data64> psi, std::vector<Modulus64> primes,
                        int n_power); // bit reverse order for GPU-NTT

    std::vector<Ninverse64> generate_n_inverse(size_t poly_modulus_degree,
                                               std::vector<Modulus64> primes);

    __global__ void unsigned_signed_convert(Data64* input, Data64* output,
                                            Modulus64* modulus);

    __global__ void fill_device_vector(Data64* vector, Data64 number, int size);

    int find_closest_divisor(int N);

    std::vector<std::vector<int>> split_array(const std::vector<int>& array,
                                              int chunk_size);

    std::vector<std::vector<int>> seperate_func(const std::vector<int>& A);

    std::vector<int> unique_sort(const std::vector<int>& input);

} // namespace heongpu
#endif // UTIL_H
