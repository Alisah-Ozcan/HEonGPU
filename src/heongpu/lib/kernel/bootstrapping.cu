// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "bootstrapping.cuh"

namespace heongpu
{

    __device__ int exponent_calculation(int& index, int& n)
    {
        Data64 result = 1ULL;
        Data64 five = 5ULL;
        Data64 mod = (n << 2) - 1;

        int bits = 32 - __clz(index);
        for (int i = bits - 1; i > -1; i--)
        {
            result = (result * result) & mod;

            if (((index >> i) & 1u))
            {
                result = (result * five) & mod;
            }
        }

        return result;
    }

    __device__ int matrix_location(int& index)
    {
        if (index == 0)
        {
            return 0;
        }

        return (3 * index) - 1;
    }

    __device__ int matrix_reverse_location(int& index)
    {
        int total = (gridDim.y - 1) * 3;
        if (index == 0)
        {
            return total;
        }

        return total - (3 * index);
    }

    __global__ void E_diagonal_generate_kernel(Complex64* output, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int block_y = blockIdx.y; // matrix index
        int logk = block_y + 1;
        int output_location = matrix_location(block_y);

        int n = 1 << n_power;
        int v_size = 1 << (n_power - logk);

        int index1 = idx & ((v_size << 1) - 1);
        int index2 = index1 >> (n_power - logk);
        Complex64 W1(1.0, 0.0);
        Complex64 W2(0.0, 0.0);
        Complex64 W3(0.0, 0.0);

        if (block_y == 0)
        {
            double angle = M_PI / (v_size << 2);
            Complex64 omega_4n(cos(angle), sin(angle));
            int expo = exponent_calculation(index1, n);

            Complex64 W = omega_4n.exp(expo);
            Complex64 W_neg = W; // W.negate();

            if (index2 == 1)
            {
                W1 = W_neg;
                W2 = Complex64(1.0, 0.0);
            }
            else
            {
                W2 = W;
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
        }
        else
        {
            double angle = M_PI / (v_size << 2);
            Complex64 omega_4n(cos(angle), sin(angle));
            int expo = exponent_calculation(index1, n);

            Complex64 W = omega_4n.exp(expo);
            Complex64 W_neg = W; // W.negate();

            if (index2 == 1)
            {
                W1 = W_neg;
                W3 = Complex64(1.0, 0.0);
            }
            else
            {
                W2 = W;
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
            output[((output_location + 2) << n_power) + idx] = W3;
        }
    }

    __global__ void E_diagonal_inverse_generate_kernel(Complex64* output,
                                                       int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int block_y = blockIdx.y; // matrix index
        int logk = block_y + 1;
        int output_location = matrix_reverse_location(block_y);

        int n = 1 << n_power;
        int v_size = 1 << (n_power - logk);

        int index1 = idx & ((v_size << 1) - 1);
        int index2 = index1 >> (n_power - logk);
        Complex64 W1(0.5, 0.0);
        Complex64 W2(0.5, 0.0);
        Complex64 W3(0.0, 0.0);

        if (block_y == 0)
        {
            if (index2 == 1)
            {
                double angle = M_PI / (v_size << 2);
                Complex64 omega_4n(cos(angle), sin(angle));
                int expo = exponent_calculation(index1, n);
                W1 = omega_4n.inverse();
                W1 = W1.exp(expo);
                W1 = W1 / Complex64(2.0, 0.0);
                W2 = W1.negate();
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
        }
        else
        {
            if (index2 == 1)
            {
                double angle = M_PI / (v_size << 2);
                Complex64 omega_4n(cos(angle), sin(angle));
                int expo = exponent_calculation(index1, n);
                W1 = omega_4n.inverse();
                W1 = W1.exp(expo);
                W1 = W1 / Complex64(2.0, 0.0);
                W2 = Complex64(0.0, 0.0);
                W3 = W1.negate();
            }

            output[(output_location << n_power) + idx] = W1;
            output[((output_location + 1) << n_power) + idx] = W2;
            output[((output_location + 2) << n_power) + idx] = W3;
        }
    }

    __global__ void E_diagonal_inverse_matrix_mult_single_kernel(
        Complex64* input, Complex64* output, bool last, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (last)
        {
            for (int i = 0; i < 2; i++)
            {
                output[idx + (i << n_power)] = input[idx + (i << n_power)];
            }
        }
        else
        {
            for (int i = 0; i < 3; i++)
            {
                output[idx + (i << n_power)] = input[idx + (i << n_power)];
            }
        }
    }

    __global__ void E_diagonal_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first1, bool first2, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int offset = first1 ? 2 : 3;
        int L_matrix_loc_ = offset + (3 * mul_index);
        int L_matrix_size = 3;

        int R_matrix_counter_ = R_matrix_counter;
        int output_index_counter_ = output_index_counter;
        int iter_R_m = iteration_count;
        if (first2)
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m = input[idx + (i << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];

                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
        else
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m =
                    temp[idx +
                         (input_index[R_matrix_counter_ - offset] << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];

                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
    }

    __global__ void E_diagonal_inverse_matrix_mult_kernel(
        Complex64* input, Complex64* output, Complex64* temp, int* diag_index,
        int* input_index, int* output_index, int iteration_count,
        int R_matrix_counter, int output_index_counter, int mul_index,
        bool first, bool last, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int L_matrix_loc_ = 3 + (3 * mul_index);
        int L_matrix_size = (last) ? 2 : 3;

        int R_matrix_counter_ = R_matrix_counter;
        int output_index_counter_ = output_index_counter;
        int iter_R_m = iteration_count;
        if (first)
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m = input[idx + (i << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];
                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
        else
        {
            for (int i = 0; i < iter_R_m; i++)
            {
                int diag_index_ = diag_index[R_matrix_counter_];
                Complex64 R_m =
                    temp[idx + (input_index[R_matrix_counter_ - 3] << n_power)];
                for (int j = 0; j < L_matrix_size; j++)
                {
                    Complex64 L_m =
                        rotated_access(input + ((L_matrix_loc_ + j) << n_power),
                                       diag_index_, idx, n_power);

                    int output_location = output_index[output_index_counter_];
                    Complex64 res = output[(output_location << n_power) + idx];
                    res = res + (L_m * R_m);
                    output[(output_location << n_power) + idx] = res;

                    output_index_counter_++;
                }
                R_matrix_counter_++;
            }
        }
    }

    __global__ void vector_rotate_kernel(Complex64* input, Complex64* output,
                                         int rotate_index, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        Complex64 rotated = rotated_access(input, rotate_index, idx, n_power);

        output[idx] = rotated;
    }

    // TODO: implement it for multiple RNS prime (currently it only works for
    // single prime)
    __global__ void mod_raise_kernel(Data64* input, Data64* output,
                                     Modulus64* modulus, int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // rns count
        int idz = blockIdx.z; // cipher count

        int location_input = idx + (idz << n_power);
        int location_output =
            idx + (idy << n_power) + ((gridDim.y * idz) << n_power);

        Data64 input_r = input[location_input];
        Data64 result = OPERATOR_GPU_64::reduce_forced(input_r, modulus[idy]);

        output[location_output] = result;
    }

} // namespace heongpu