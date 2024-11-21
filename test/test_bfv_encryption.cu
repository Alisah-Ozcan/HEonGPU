// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include <gtest/gtest.h>

TEST(HEonGPU, BFV_Encryption_Decryption)
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

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<uint64_t> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            std::equal(message.begin(), message.end(), gpu_result.begin()),
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

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<uint64_t> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            std::equal(message.begin(), message.end(), gpu_result.begin()),
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

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<uint64_t> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            std::equal(message.begin(), message.end(), gpu_result.begin()),
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

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<uint64_t> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            std::equal(message.begin(), message.end(), gpu_result.begin()),
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

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
        std::vector<uint64_t> message(poly_modulus_degree, 0ULL);
        for (int i = 0; i < poly_modulus_degree; i++)
        {
            message[i] = dis(gen);
        }

        heongpu::Plaintext P1(context);
        encoder.encode(P1, message);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        heongpu::Plaintext P2(context);
        decryptor.decrypt(P2, C1);

        std::vector<uint64_t> gpu_result;
        encoder.decode(gpu_result, P2);

        cudaDeviceSynchronize();

        EXPECT_EQ(
            std::equal(message.begin(), message.end(), gpu_result.begin()),
            true);
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}