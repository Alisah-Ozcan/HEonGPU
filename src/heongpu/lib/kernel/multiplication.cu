// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "multiplication.cuh"

namespace heongpu
{
    __global__ void
    fast_convertion(Data* in1, Data* in2, Data* out1, Modulus* ibase,
                    Modulus* obase, Modulus m_tilde,
                    Data inv_prod_q_mod_m_tilde, Data* inv_m_tilde_mod_Bsk,
                    Data* prod_q_mod_Bsk, Data* base_change_matrix_Bsk,
                    Data* base_change_matrix_m_tilde,
                    Data* inv_punctured_prod_mod_base_array, int n_power,
                    int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // cipher count * 2 // for input

        int location = idx + (((idy % 2) * ibase_size)
                              << n_power); // ibase_size = decomp_modulus_count
        Data* input = ((idy >> 1) == 0) ? in1 : in2;

        Data temp[MAX_BSK_SIZE];
        Data temp_[MAX_BSK_SIZE];

        // taking input from global and mult with m_tilde
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp_[i] = input[location + (i << n_power)];
            temp[i] = VALUE_GPU::mult(temp_[i], m_tilde.value, ibase[i]);
            temp[i] = VALUE_GPU::mult(
                temp[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        // for Bsk
        Data temp2[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp2[i] = 0;
#pragma unroll
            for (int j = 0; j < ibase_size; j++)
            {
                Data mult = VALUE_GPU::mult(
                    temp[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp2[i] = VALUE_GPU::add(temp2[i], mult, obase[i]);
            }
        }

        // for m_tilde
        temp2[obase_size] = 0;
#pragma unroll
        for (int j = 0; j < ibase_size; j++)
        {
            Data temp_in = VALUE_GPU::reduce_forced(temp[j], m_tilde);
            Data mult = VALUE_GPU::mult(temp_in, base_change_matrix_m_tilde[j],
                                        m_tilde);
            temp2[obase_size] =
                VALUE_GPU::add(temp2[obase_size], mult, m_tilde);
        }

        // sm_mrq
        Data m_tilde_div_2 = m_tilde.value >> 1;
        Data r_m_tilde =
            VALUE_GPU::mult(temp2[obase_size], inv_prod_q_mod_m_tilde, m_tilde);
        r_m_tilde = m_tilde.value - r_m_tilde;

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data temp3 = r_m_tilde;
            if (temp3 >= m_tilde_div_2)
            {
                temp3 = obase[i].value - m_tilde.value;
                temp3 = VALUE_GPU::add(temp3, r_m_tilde, obase[i]);
            }

            temp3 = VALUE_GPU::mult(temp3, prod_q_mod_Bsk[i], obase[i]);
            temp3 = VALUE_GPU::add(temp2[i], temp3, obase[i]);
            temp2[i] = VALUE_GPU::mult(temp3, inv_m_tilde_mod_Bsk[i], obase[i]);
        }

        int location2 = idx + ((idy * (obase_size + ibase_size)) << n_power);
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location2 + (i << n_power)] = temp_[i];
        }
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            out1[location2 + ((i + ibase_size) << n_power)] = temp2[i];
        }
    }

    __global__ void cross_multiplication(Data* in1, Data* in2, Data* out,
                                         Modulus* modulus, int n_power,
                                         int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // decomp size + bsk size

        int location = idx + (idy << n_power);

        Data ct0_0 = in1[location];
        Data ct0_1 = in1[location + (decomp_size << n_power)];

        Data ct1_0 = in2[location];
        Data ct1_1 = in2[location + (decomp_size << n_power)];

        Data out_0 = VALUE_GPU::mult(ct0_0, ct1_0, modulus[idy]);
        Data out_1_0 = VALUE_GPU::mult(ct0_0, ct1_1, modulus[idy]);
        Data out_1_1 = VALUE_GPU::mult(ct0_1, ct1_0, modulus[idy]);
        Data out_2 = VALUE_GPU::mult(ct0_1, ct1_1, modulus[idy]);
        Data out_1 = VALUE_GPU::add(out_1_0, out_1_1, modulus[idy]);

        out[location] = out_0;
        out[location + (decomp_size << n_power)] = out_1;
        out[location + (decomp_size << (n_power + 1))] = out_2;
    }

    __global__ void
    fast_floor(Data* in_baseq_Bsk, Data* out1, Modulus* ibase, Modulus* obase,
               Modulus plain_modulus, Data* inv_punctured_prod_mod_base_array,
               Data* base_change_matrix_Bsk, Data* inv_prod_q_mod_Bsk,
               Data* inv_punctured_prod_mod_B_array, Data* base_change_matrix_q,
               Data* base_change_matrix_msk, Data inv_prod_B_mod_m_sk,
               Data* prod_B_mod_q, int n_power, int ibase_size, int obase_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int idy = blockIdx.y; // 3

        int location_q =
            idx + ((idy * (ibase_size + obase_size))
                   << n_power); // ibase_size = decomp_modulus_count
        int location_Bsk =
            idx + ((idy * (ibase_size + obase_size)) << n_power) +
            (ibase_size << n_power); // ibase_size = decomp_modulus_count

        Data reg_q[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            reg_q[i] =
                VALUE_GPU::mult(in_baseq_Bsk[location_q + (i << n_power)],
                                plain_modulus.value, ibase[i]);
            reg_q[i] = VALUE_GPU::mult(
                reg_q[i], inv_punctured_prod_mod_base_array[i], ibase[i]);
        }

        Data reg_Bsk[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            reg_Bsk[i] =
                VALUE_GPU::mult(in_baseq_Bsk[location_Bsk + (i << n_power)],
                                plain_modulus.value, obase[i]);
        }

        // for Bsk
        Data temp[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            temp[i] = 0;
            for (int j = 0; j < ibase_size; j++)
            {
                Data mult = VALUE_GPU::mult(
                    reg_q[j], base_change_matrix_Bsk[j + (i * ibase_size)],
                    obase[i]);
                temp[i] = VALUE_GPU::add(temp[i], mult, obase[i]);
            }
        }

#pragma unroll
        for (int i = 0; i < obase_size; i++)
        {
            Data temp2 = VALUE_GPU::sub(obase[i].value, temp[i], obase[i]);
            temp2 = VALUE_GPU::add(temp2, reg_Bsk[i], obase[i]);
            reg_Bsk[i] =
                VALUE_GPU::mult(temp2, inv_prod_q_mod_Bsk[i], obase[i]);
        }

        Data temp3[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < obase_size - 1; i++)
        { // only B bases
            temp3[i] = VALUE_GPU::mult(
                reg_Bsk[i], inv_punctured_prod_mod_B_array[i], obase[i]);
        }

        Data temp4[MAX_BSK_SIZE];
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            temp4[i] = 0;
#pragma unroll
            for (int j = 0; j < obase_size - 1; j++)
            {
                Data temp3_ =
                    VALUE_GPU::reduce_forced(temp3[j], ibase[i]); // extra

                Data mult = VALUE_GPU::mult(
                    temp3_, base_change_matrix_q[j + (i * (obase_size - 1))],
                    ibase[i]); // extra
                mult = VALUE_GPU::reduce_forced(mult, ibase[i]); // extra
                temp4[i] = VALUE_GPU::add(temp4[i], mult, ibase[i]);
            }
        }

        // for m_sk
        temp4[ibase_size] = 0;
#pragma unroll
        for (int j = 0; j < obase_size - 1; j++)
        {
            Data mult = VALUE_GPU::mult(temp3[j], base_change_matrix_msk[j],
                                        obase[obase_size - 1]);
            temp4[ibase_size] =
                VALUE_GPU::add(temp4[ibase_size], mult, obase[obase_size - 1]);
        }

        Data alpha_sk =
            VALUE_GPU::sub(obase[obase_size - 1].value, reg_Bsk[obase_size - 1],
                           obase[obase_size - 1]);
        alpha_sk =
            VALUE_GPU::add(alpha_sk, temp4[ibase_size], obase[obase_size - 1]);
        alpha_sk = VALUE_GPU::mult(alpha_sk, inv_prod_B_mod_m_sk,
                                   obase[obase_size - 1]);

        Data m_sk_div_2 = obase[obase_size - 1].value >> 1;

#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            Data obase_ =
                VALUE_GPU::reduce_forced(obase[obase_size - 1].value, ibase[i]);
            Data temp4_ = VALUE_GPU::reduce_forced(temp4[i], ibase[i]);
            Data alpha_sk_ = VALUE_GPU::reduce_forced(alpha_sk, ibase[i]);
            if (alpha_sk > m_sk_div_2)
            {
                Data inner =
                    VALUE_GPU::sub(obase_, alpha_sk_, ibase[i]); // extra
                inner = VALUE_GPU::mult(inner, prod_B_mod_q[i], ibase[i]);
                temp4[i] = VALUE_GPU::add(temp4_, inner, ibase[i]);
            }
            else
            {
                Data inner =
                    VALUE_GPU::sub(ibase[i].value, prod_B_mod_q[i], ibase[i]);
                inner = VALUE_GPU::mult(inner, alpha_sk_, ibase[i]); // extra
                temp4[i] = VALUE_GPU::add(temp4_, inner, ibase[i]);
            }
        }

        int location_out =
            idx + ((idy * ibase_size)
                   << n_power); // ibase_size = decomp_modulus_count
#pragma unroll
        for (int i = 0; i < ibase_size; i++)
        {
            out1[location_out + (i << n_power)] = temp4[i];
        }
    }

    __global__ void threshold_kernel(Data* plain_in, Data* output,
                                     Modulus* modulus,
                                     Data* plain_upper_half_increment,
                                     Data plain_upper_half_threshold,
                                     int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod

        Data plain_reg = plain_in[idx];

        if (plain_reg >= plain_upper_half_threshold)
        {
            output[idx + (block_y << n_power)] = VALUE_GPU::add(
                plain_reg, plain_upper_half_increment[block_y],
                modulus[block_y]); // plain_reg +
                                   // plain_upper_half_increment[block_y];
        }
        else
        {
            output[idx + (block_y << n_power)] = plain_reg;
        }
    }

    __global__ void cipherplain_kernel(Data* cipher, Data* plain_in,
                                       Data* output, Modulus* modulus,
                                       int n_power, int decomp_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // decomp_mod
        int block_z = blockIdx.z; // cipher size

        int index1 = idx + (block_y << n_power);
        int index2 = index1 + ((decomp_size << n_power) * block_z);

        output[index2] =
            VALUE_GPU::mult(cipher[index2], plain_in[index1], modulus[block_y]);
    }

    __global__ void cipherplain_multiplication_kernel(Data* in1, Data* in2,
                                                      Data* out,
                                                      Modulus* modulus,
                                                      int n_power)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size
        int block_y = blockIdx.y; // rns count
        int block_z = blockIdx.z; // cipher count

        int location_ct =
            idx + (block_y << n_power) + ((gridDim.y * block_z) << n_power);
        int location_pt = idx + (block_y << n_power);

        Data ct = in1[location_ct];
        Data pt = in2[location_pt];

        ct = VALUE_GPU::mult(ct, pt, modulus[block_y]);
        out[location_ct] = ct;
    }

} // namespace heongpu