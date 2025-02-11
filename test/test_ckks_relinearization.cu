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

TEST(HEonGPU, CKKS_Relinearization_Keyswitching_Method_I)
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

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 30);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
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

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 30);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
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

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 35);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
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

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 40);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
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

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 45);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }
}

TEST(HEonGPU, CKKS_Relinearization_Keyswitching_Method_II)
{
    cudaSetDevice(0);
    {
        size_t poly_modulus_degree = 4096;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({40, 30, 30}, {40, 40});
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 30);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }

    {
        size_t poly_modulus_degree = 8192;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({40, 30, 30, 30, 30}, {40, 40});
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 30);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }

    {
        size_t poly_modulus_degree = 16384;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus(
            {45, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35}, {45, 45});
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 35);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }

    {
        size_t poly_modulus_degree = 32768;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
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

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 40);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }

    {
        size_t poly_modulus_degree = 65536;
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({59, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
                                   45, 45, 45, 45, 45, 45, 45},
                                  {59, 59});
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::Relinkey relin_key(context);
        keygen.generate_relin_key(relin_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const int row_size = poly_modulus_degree / 2;
        std::vector<double> message1(row_size, 0);
        std::vector<double> message2(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<double> message_multiplication_result(row_size, 0);
        for (int i = 0; i < row_size; i++)
        {
            message_multiplication_result[i] = message1[i] * message2[i];
        }

        double scale = pow(2.0, 45);
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1, scale);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        operators.multiply_inplace(C1, C2);
        operators.relinearize_inplace(C1, relin_key);
        operators.rescale_inplace(C1);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C1);

        std::vector<double> gpu_result;
        encoder.decode(gpu_result, P3);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            fix_point_array_check(message_multiplication_result, gpu_result),
            true);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
