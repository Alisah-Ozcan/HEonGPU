// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include <gtest/gtest.h>

template <typename T>
bool fix_point_equal(T input1, T input2, T epsilon = static_cast<T>(1e-4))
{
    return std::fabs(input1 - input2) < epsilon;
}

template <typename T>
bool fix_point_array_check(const std::vector<T>& array1,
                           const std::vector<T>& array2,
                           T epsilon = static_cast<T>(1e-4))
{
    if (array1.size() != array2.size())
    {
        return false;
    }

    for (size_t i = 0; i < array1.size(); ++i)
    {
        if (!fix_point_equal(array1[i], array2[i], epsilon))
        {
            return false;
        }
    }

    return true;
}

TEST(HEonGPU, BFV_Ciphertext_Rotation_Keyswitching_Method_II_Part_I)
{
    cudaSetDevice(0);
    {
        size_t poly_modulus_degree = 32768;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({59, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                   40, 40, 40, 40, 40, 40, 40, 40},
                                  {59, 59});
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::Galoiskey galois_key(context);
        keygen.generate_galois_key(galois_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEOperator operators(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
        }

        int shift_count = 2;
        std::vector<double> message_rotation_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_rotation_result[i] = message1[(i + shift_count) % row_size];
        }

        double scale = pow(2.0, 40);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        operators.rotate_rows(C1, C1, galois_key, shift_count);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message_rotation_result, gpu_result,
                                        static_cast<double>(1e-1)),
                  true);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}