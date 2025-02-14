// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HE_BOOTSTRAPPING_H
#define HE_BOOTSTRAPPING_H

#include <curand_kernel.h>
#include "context.cuh"

namespace heongpu
{

    template <typename T>
    __device__ T rotated_access(T* data, int& rotate, int& idx, int& n_power)
    {
        int n = 1 << n_power;
        int new_location = idx + rotate;

        if (new_location < 0)
        {
            new_location = new_location + n;
        }
        else
        {
            int mask = n - 1;
            new_location = new_location & mask;
        }

        return data[new_location];
    }

    __device__ int exponent_calculation(int& index, int& n);

    __device__ int matrix_location(int& index);

    __device__ int matrix_reverse_location(int& index);

    __global__ void E_diagonal_generate_kernel(Complex64* output, int n_power);

    __global__ void E_diagonal_inverse_generate_kernel(Complex64* output,
                                                       int n_power);

    __global__ void E_diagonal_inverse_matrix_mult_single_kernel(
        Complex64* input, Complex64* output, bool last, int n_power);

    __global__ void E_diagonal_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first1, bool first2, int n_power);

    __global__ void E_diagonal_inverse_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first, bool last, int n_power);

    __global__ void vector_rotate_kernel(Complex64* input, Complex64* output,
                                         int rotate_index, int n_power);

    // TODO: work it for multiple RNS prime (currently it only works for single
    // prime)
    __global__ void mod_raise_kernel(Data64* input, Data64* output,
                                     Modulus64* modulus, int n_power);

} // namespace heongpu

#endif // HE_BOOTSTRAPPING_H