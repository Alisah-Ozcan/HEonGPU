// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "../example_util.h"
#include <omp.h>

void default_stream_function(
    heongpu::Parameters& context, heongpu::HEArithmeticOperator& operators,
    heongpu::Relinkey& relinkeys, heongpu::Galoiskey& galoiskeys,
    std::vector<std::vector<Data64>>& ciphertext1_in_cpu,
    std::vector<std::vector<Data64>>& ciphertext2_in_cpu,
    std::vector<std::vector<Data64>>& ciphertext3_in_cpu,
    std::vector<std::vector<Data64>>& plaintext_in_cpu,
    std::vector<heongpu::Ciphertext>& ciphertext_result_in_gpu, int count)
{
    heongpu::Ciphertext temp1(context);
    heongpu::Ciphertext temp2(context);

    for (int i = 0; i < count; i++)
    {
        heongpu::Ciphertext ciphertext1_in_gpu(ciphertext1_in_cpu[i], context);
        heongpu::Plaintext plaintext_in_gpu(plaintext_in_cpu[i], context);

        operators.multiply_plain(ciphertext1_in_gpu, plaintext_in_gpu, temp1);

        heongpu::Ciphertext ciphertext2_in_gpu(ciphertext2_in_cpu[i], context);

        operators.sub(temp1, ciphertext2_in_gpu, temp2);

        heongpu::Ciphertext ciphertext3_in_gpu(ciphertext3_in_cpu[i], context);

        operators.multiply_inplace(temp2, ciphertext3_in_gpu);

        operators.relinearize_inplace(temp2, relinkeys);

        operators.add_inplace(temp1, temp2);

        heongpu::Ciphertext result(context);
        operators.rotate_rows(temp1, result, galoiskeys, 5);

        ciphertext_result_in_gpu[i] = std::move(result);
    }
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    // Initialize encryption parameters for the CKKS scheme.
    heongpu::Parameters context(
        heongpu::scheme_type::bfv,
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 8192;
    int plain_modulus = 786433;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_default_coeff_modulus(1);
    context.set_plain_modulus(plain_modulus);
    context.generate();
    context.print_parameters();

    // Generate keys: the public key for encryption, the secret key for
    // decryption, and evaluation keys(relinkey, galoiskey, switchkey).
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
    heongpu::HEArithmeticOperator operators(context, encoder);

    // Assume ciphertexts and plaintexts come from network(from CPU RAM)
    std::cout << "Random ciphertext generetes in CPU... " << std::endl;
    int total_size = 64;
    std::vector<std::vector<Data64>> ciphertext1_in_cpu;
    std::vector<std::vector<Data64>> ciphertext2_in_cpu;
    std::vector<std::vector<Data64>> ciphertext3_in_cpu;
    std::vector<std::vector<Data64>> plaintext_in_cpu;

    // we know ciphetext and plaintext size
    int cipher_size =
        2 * context.poly_modulus_degree() * context.ciphertext_modulus_count();
    int plain_size = context.poly_modulus_degree();

    for (int i = 0; i < total_size; i++)
    {
        ciphertext1_in_cpu.push_back(
            std::move(random_vector_generator(cipher_size)));
        ciphertext2_in_cpu.push_back(
            std::move(random_vector_generator(cipher_size)));
        ciphertext3_in_cpu.push_back(
            std::move(random_vector_generator(cipher_size)));
        plaintext_in_cpu.push_back(
            std::move(random_vector_generator(plain_size)));
    }
    std::cout << "Genereted." << std::endl;

    std::cout << "Operations starts..." << std::endl;
    std::vector<heongpu::Ciphertext> results(total_size);
    default_stream_function(context, operators, relin_key, galois_key,
                            ciphertext1_in_cpu, ciphertext2_in_cpu,
                            ciphertext3_in_cpu, plaintext_in_cpu, results,
                            total_size);
    std::cout << "Done." << std::endl;

    std::cout << "To see differece, check streams view with using NVIDIA "
                 "Nsight Systems."
              << std::endl;

    return EXIT_SUCCESS;
}
