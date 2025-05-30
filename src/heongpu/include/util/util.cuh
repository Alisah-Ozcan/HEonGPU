// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_UTIL_H
#define HEONGPU_UTIL_H

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
#include <gmp.h>

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

    bool coefficient_validator(const std::vector<int>& log_Q_bases_bit_sizes,
                               const std::vector<int>& log_P_bases_bit_sizes);

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

    Data64 extendedGCD(Data64 a, Data64 b, Data64& x, Data64& y);
    Data64 modInverse(Data64 a, Data64 m);
    int calculate_bit_size(Data64 number);

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

    std::vector<Data64>
    calculate_last_q_modinv(const std::vector<Modulus64>& prime_vector,
                            const int Q_prime_size, const int P_size);

    std::vector<Data64>
    calculate_half(const std::vector<Modulus64>& prime_vector,
                   const int P_size);

    std::vector<Data64>
    calculate_half_mod(const std::vector<Modulus64>& prime_vector,
                       const std::vector<Data64>& half, const int Q_prime_size,
                       const int P_size);

    std::vector<Data64>
    calculate_factor(const std::vector<Modulus64>& prime_vector,
                     const int Q_size, const int P_size);

    std::vector<Data64> calculate_Mi(const std::vector<Modulus64>& prime_vector,
                                     const int size);

    std::vector<Data64>
    calculate_Mi_inv(const std::vector<Modulus64>& prime_vector,
                     const int size);

    std::vector<Data64> calculate_M(const std::vector<Modulus64>& prime_vector,
                                    const int size);

    std::vector<Data64>
    calculate_upper_half_threshold(const std::vector<Modulus64>& prime_vector,
                                   const int size);

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////

    static __forceinline__ __device__ uint32_t warp_reduce(uint32_t input)
    {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        {
            input += __shfl_down_sync(0xFFFFFFFF, input, offset);
        }
        return input;
    }

} // namespace heongpu
#endif // HEONGPU_UTIL_H
