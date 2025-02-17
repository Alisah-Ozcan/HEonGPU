// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include <gtest/gtest.h>

TEST(HEonGPU, BFV_Ciphertext_Rotation_Keyswitching_Method_II)
{
    cudaSetDevice(0);

    {
        size_t poly_modulus_degree = 4096;
        int plain_modulus = 1032193;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({40, 40}, {40, 40});
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

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey galois_key(context, shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    {
        size_t poly_modulus_degree = 8192;
        int plain_modulus = 1032193;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({54, 54, 54}, {55, 55});
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

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey galois_key(context, shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    {
        size_t poly_modulus_degree = 16384;
        int plain_modulus = 786433;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({54, 54, 54, 54, 55, 55, 55}, {55, 55});
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

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey galois_key(context, shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    {
        size_t poly_modulus_degree = 32768;
        int plain_modulus = 786433;
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus(
            {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59}, {59, 59});
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

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey galois_key(context, shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }

    {
        size_t poly_modulus_degree = 65536;
        int plain_modulus = 786433;
        // TODO: find optimal way to store huge galois key, maybe store it in
        // CPU RAM.
        // heongpu::Parameters context(heongpu::scheme_type::bfv,
        // heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        // heongpu::sec_level_type::none);
        // context.set_poly_modulus_degree(poly_modulus_degree);
        // context.set_coeff_modulus({58, 58, 58, 58, 58, 58, 58, 58, 58, 59,
        // 59, 59, 59, 59, 59,
        //     59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59},
        //    {59});
        // context.set_plain_modulus(plain_modulus);
        // context.generate();
        heongpu::Parameters context(
            heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus(
            {58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59}, {59, 59});
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

        std::vector<int> shift_key_index = {-5, -3, 31};
        heongpu::Galoiskey galois_key(context, shift_key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (size_t j = 0; j < shift_key_index.size(); j++)
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint64_t> dis(0, plain_modulus - 1);
            std::vector<uint64_t> message1(poly_modulus_degree, 0ULL);
            for (int i = 0; i < poly_modulus_degree; i++)
            {
                message1[i] = dis(gen);
            }

            int shift_count = shift_key_index[j];
            size_t row_size = poly_modulus_degree / 2;
            std::vector<uint64_t> message_rotation_result(poly_modulus_degree,
                                                          0ULL);
            for (int i = 0; i < row_size; i++)
            {
                int index = ((i + shift_count) < 0)
                                ? ((i + shift_count) + row_size)
                                : ((i + shift_count) % row_size);
                message_rotation_result[i] = message1[index];
                message_rotation_result[i + row_size] =
                    message1[index + row_size];
            }

            heongpu::Plaintext P1(context);
            encoder.encode(P1, message1);

            heongpu::Ciphertext C1(context);
            encryptor.encrypt(C1, P1);

            operators.rotate_rows(C1, C1, galois_key, shift_count);

            heongpu::Plaintext P3(context);
            decryptor.decrypt(P3, C1);

            std::vector<uint64_t> gpu_rotation_result;
            encoder.decode(gpu_rotation_result, P3);

            cudaDeviceSynchronize();

            EXPECT_EQ(std::equal(message_rotation_result.begin(),
                                 message_rotation_result.end(),
                                 gpu_rotation_result.begin()),
                      true);
        }
    }
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}