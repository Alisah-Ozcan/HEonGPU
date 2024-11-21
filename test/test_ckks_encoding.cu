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

TEST(HEonGPU, CKKS_Encoding_Decoding)
{
    cudaSetDevice(0);
    {
        size_t poly_modulus_degree = 4096;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({40, 30, 30}, {40});
        context.generate();

        heongpu::HEEncoder encoder(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        double scale = pow(2.0, 30);
        encoder.encode(P1, message, scale);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P1);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);

        heongpu::Plaintext P2(context);
        double number = static_cast<double>(dis(gen));
        encoder.encode(P2, number, scale);

        std::vector<double> gpu_result2;
        encoder.decode(gpu_result2, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_equal(number, gpu_result2[0]), true);
    }

    {
        size_t poly_modulus_degree = 8192;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({40, 30, 30, 30, 30}, {40});
        context.generate();

        heongpu::HEEncoder encoder(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        double scale = pow(2.0, 30);
        encoder.encode(P1, message, scale);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P1);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);

        heongpu::Plaintext P2(context);
        double number = static_cast<double>(dis(gen));
        encoder.encode(P2, number, scale);

        std::vector<double> gpu_result2;
        encoder.decode(gpu_result2, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_equal(number, gpu_result2[0]), true);
    }

    {
        size_t poly_modulus_degree = 16384;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus(
            {45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35}, {45});
        context.generate();

        heongpu::HEEncoder encoder(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        double scale = pow(2.0, 30);
        encoder.encode(P1, message, scale);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P1);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);

        heongpu::Plaintext P2(context);
        double number = static_cast<double>(dis(gen));
        encoder.encode(P2, number, scale);

        std::vector<double> gpu_result2;
        encoder.decode(gpu_result2, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_equal(number, gpu_result2[0]), true);
    }

    {
        size_t poly_modulus_degree = 32768;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({59, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                   40, 40, 40, 40, 40, 40, 40, 40},
                                  {59});
        context.generate();

        heongpu::HEEncoder encoder(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        double scale = pow(2.0, 40);
        encoder.encode(P1, message, scale);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P1);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);

        heongpu::Plaintext P2(context);
        double number = static_cast<double>(dis(gen));
        encoder.encode(P2, number, scale);

        std::vector<double> gpu_result2;
        encoder.decode(gpu_result2, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_equal(number, gpu_result2[0]), true);
    }

    {
        size_t poly_modulus_degree = 65536;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({59, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45},
                                  {59});
        context.generate();

        heongpu::HEEncoder encoder(context);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        double scale = pow(2.0, 45);
        encoder.encode(P1, message, scale);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P1);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_array_check(message, gpu_result), true);

        heongpu::Plaintext P2(context);
        double number = static_cast<double>(dis(gen));
        encoder.encode(P2, number, scale);

        std::vector<double> gpu_result2;
        encoder.decode(gpu_result2, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(fix_point_equal(number, gpu_result2[0]), true);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}