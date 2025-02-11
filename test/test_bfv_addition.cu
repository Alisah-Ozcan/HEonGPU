// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include <gtest/gtest.h>

TEST(HEonGPU, BFV_Ciphertext_Ciphertext_Addition_Subtraction)
{
    cudaSetDevice(0);
    {
        size_t poly_modulus_degree = 4096;
        int plain_modulus = 1032193;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({36, 36}, {37});
        context.set_plain_modulus(plain_modulus);
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
        std::vector<uint64_t> message2(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<uint64_t> message_addition_result(poly_modulus_degree,
                                                      0ULL);
        std::vector<uint64_t> message_subtraction_result(poly_modulus_degree,
                                                         0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            uint64_t addition_result = message1[i] + message2[i];
            message_addition_result[i] = (addition_result > plain_modulus)
                                             ? (addition_result - plain_modulus)
                                             : addition_result;

            uint64_t subtraction_result = (message2[i] > message1[i])
                                              ? (message1[i] + plain_modulus)
                                              : message1[i];
            message_subtraction_result[i] = subtraction_result - message2[i];
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        heongpu::Ciphertext C_addition(context);
        operators.add(C1, C2, C_addition);

        heongpu::Ciphertext C_subtraction(context);
        operators.sub(C1, C2, C_subtraction);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C_addition);

        heongpu::Plaintext P4(context);
        decryptor.decrypt(P4, C_subtraction);

        std::vector<uint64_t> gpu_addition_result;
        encoder.decode(gpu_addition_result, P3);

        std::vector<uint64_t> gpu_subtraction_result;
        encoder.decode(gpu_subtraction_result, P4);

        cudaDeviceSynchronize();

        EXPECT_EQ(std::equal(message_addition_result.begin(),
                             message_addition_result.end(),
                             gpu_addition_result.begin()),
                  true);

        EXPECT_EQ(std::equal(message_subtraction_result.begin(),
                             message_subtraction_result.end(),
                             gpu_subtraction_result.begin()),
                  true);
    }

    {
        size_t poly_modulus_degree = 8192;
        int plain_modulus = 1032193;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({54, 54, 54}, {55});
        context.set_plain_modulus(plain_modulus);
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
        std::vector<uint64_t> message2(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<uint64_t> message_addition_result(poly_modulus_degree,
                                                      0ULL);
        std::vector<uint64_t> message_subtraction_result(poly_modulus_degree,
                                                         0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            uint64_t addition_result = message1[i] + message2[i];
            message_addition_result[i] = (addition_result > plain_modulus)
                                             ? (addition_result - plain_modulus)
                                             : addition_result;

            uint64_t subtraction_result = (message2[i] > message1[i])
                                              ? (message1[i] + plain_modulus)
                                              : message1[i];
            message_subtraction_result[i] = subtraction_result - message2[i];
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        heongpu::Ciphertext C_addition(context);
        operators.add(C1, C2, C_addition);

        heongpu::Ciphertext C_subtraction(context);
        operators.sub(C1, C2, C_subtraction);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C_addition);

        heongpu::Plaintext P4(context);
        decryptor.decrypt(P4, C_subtraction);

        std::vector<uint64_t> gpu_addition_result;
        encoder.decode(gpu_addition_result, P3);

        std::vector<uint64_t> gpu_subtraction_result;
        encoder.decode(gpu_subtraction_result, P4);

        cudaDeviceSynchronize();

        EXPECT_EQ(std::equal(message_addition_result.begin(),
                             message_addition_result.end(),
                             gpu_addition_result.begin()),
                  true);

        EXPECT_EQ(std::equal(message_subtraction_result.begin(),
                             message_subtraction_result.end(),
                             gpu_subtraction_result.begin()),
                  true);
    }

    {
        size_t poly_modulus_degree = 16384;
        int plain_modulus = 786433;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({54, 54, 54, 54, 55, 55, 55}, {55});
        context.set_plain_modulus(plain_modulus);
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
        std::vector<uint64_t> message2(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<uint64_t> message_addition_result(poly_modulus_degree,
                                                      0ULL);
        std::vector<uint64_t> message_subtraction_result(poly_modulus_degree,
                                                         0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            uint64_t addition_result = message1[i] + message2[i];
            message_addition_result[i] = (addition_result > plain_modulus)
                                             ? (addition_result - plain_modulus)
                                             : addition_result;

            uint64_t subtraction_result = (message2[i] > message1[i])
                                              ? (message1[i] + plain_modulus)
                                              : message1[i];
            message_subtraction_result[i] = subtraction_result - message2[i];
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        heongpu::Ciphertext C_addition(context);
        operators.add(C1, C2, C_addition);

        heongpu::Ciphertext C_subtraction(context);
        operators.sub(C1, C2, C_subtraction);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C_addition);

        heongpu::Plaintext P4(context);
        decryptor.decrypt(P4, C_subtraction);

        std::vector<uint64_t> gpu_addition_result;
        encoder.decode(gpu_addition_result, P3);

        std::vector<uint64_t> gpu_subtraction_result;
        encoder.decode(gpu_subtraction_result, P4);

        cudaDeviceSynchronize();

        EXPECT_EQ(std::equal(message_addition_result.begin(),
                             message_addition_result.end(),
                             gpu_addition_result.begin()),
                  true);

        EXPECT_EQ(std::equal(message_subtraction_result.begin(),
                             message_subtraction_result.end(),
                             gpu_subtraction_result.begin()),
                  true);
    }

    {
        size_t poly_modulus_degree = 32768;
        int plain_modulus = 786433;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus(
            {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59}, {59});
        context.set_plain_modulus(plain_modulus);
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
        std::vector<uint64_t> message2(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<uint64_t> message_addition_result(poly_modulus_degree,
                                                      0ULL);
        std::vector<uint64_t> message_subtraction_result(poly_modulus_degree,
                                                         0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            uint64_t addition_result = message1[i] + message2[i];
            message_addition_result[i] = (addition_result > plain_modulus)
                                             ? (addition_result - plain_modulus)
                                             : addition_result;

            uint64_t subtraction_result = (message2[i] > message1[i])
                                              ? (message1[i] + plain_modulus)
                                              : message1[i];
            message_subtraction_result[i] = subtraction_result - message2[i];
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        heongpu::Ciphertext C_addition(context);
        operators.add(C1, C2, C_addition);

        heongpu::Ciphertext C_subtraction(context);
        operators.sub(C1, C2, C_subtraction);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C_addition);

        heongpu::Plaintext P4(context);
        decryptor.decrypt(P4, C_subtraction);

        std::vector<uint64_t> gpu_addition_result;
        encoder.decode(gpu_addition_result, P3);

        std::vector<uint64_t> gpu_subtraction_result;
        encoder.decode(gpu_subtraction_result, P4);

        cudaDeviceSynchronize();

        EXPECT_EQ(std::equal(message_addition_result.begin(),
                             message_addition_result.end(),
                             gpu_addition_result.begin()),
                  true);

        EXPECT_EQ(std::equal(message_subtraction_result.begin(),
                             message_subtraction_result.end(),
                             gpu_subtraction_result.begin()),
                  true);
    }

    {
        size_t poly_modulus_degree = 65536;
        int plain_modulus = 786433;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({58, 58, 58, 58, 58, 58, 58, 58, 58, 59,
                                   59, 59, 59, 59, 59, 59, 59, 59, 59, 59,
                                   59, 59, 59, 59, 59, 59, 59, 59, 59},
                                  {59});
        context.set_plain_modulus(plain_modulus);
        context.generate();

        heongpu::HEKeyGenerator keygen(context);
        heongpu::Secretkey secret_key(context);
        keygen.generate_secret_key(secret_key);

        heongpu::Publickey public_key(context);
        keygen.generate_public_key(public_key, secret_key);

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
        std::vector<uint64_t> message2(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message1[i] = dis(gen);
            message2[i] = dis(gen);
        }

        std::vector<uint64_t> message_addition_result(poly_modulus_degree,
                                                      0ULL);
        std::vector<uint64_t> message_subtraction_result(poly_modulus_degree,
                                                         0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            uint64_t addition_result = message1[i] + message2[i];
            message_addition_result[i] = (addition_result > plain_modulus)
                                             ? (addition_result - plain_modulus)
                                             : addition_result;

            uint64_t subtraction_result = (message2[i] > message1[i])
                                              ? (message1[i] + plain_modulus)
                                              : message1[i];
            message_subtraction_result[i] = subtraction_result - message2[i];
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message1);

        heongpu::Plaintext P2(context);
        encoder.encode(P2, message2);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Ciphertext C2(context);
        encryptor.encrypt(C2, P2);

        heongpu::Ciphertext C_addition(context);
        operators.add(C1, C2, C_addition);

        heongpu::Ciphertext C_subtraction(context);
        operators.sub(C1, C2, C_subtraction);

        heongpu::Plaintext P3(context);
        decryptor.decrypt(P3, C_addition);

        heongpu::Plaintext P4(context);
        decryptor.decrypt(P4, C_subtraction);

        std::vector<uint64_t> gpu_addition_result;
        encoder.decode(gpu_addition_result, P3);

        std::vector<uint64_t> gpu_subtraction_result;
        encoder.decode(gpu_subtraction_result, P4);

        cudaDeviceSynchronize();

        EXPECT_EQ(std::equal(message_addition_result.begin(),
                             message_addition_result.end(),
                             gpu_addition_result.begin()),
                  true);

        EXPECT_EQ(std::equal(message_subtraction_result.begin(),
                             message_subtraction_result.end(),
                             gpu_subtraction_result.begin()),
                  true);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}