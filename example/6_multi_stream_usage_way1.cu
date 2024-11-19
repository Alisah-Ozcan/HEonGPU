// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "example_util.h"
#include <omp.h>

void multi_stream_function_way1(
    heongpu::Parameters& context, heongpu::HEOperator& operators,
    heongpu::Relinkey& relinkeys, heongpu::Galoiskey& galoiskeys,
    std::vector<std::vector<Data>>& ciphertext1_in_cpu,
    std::vector<std::vector<Data>>& ciphertext2_in_cpu,
    std::vector<std::vector<Data>>& ciphertext3_in_cpu,
    std::vector<std::vector<Data>>& plaintext_in_cpu,
    std::vector<heongpu::Ciphertext>& ciphertext_result_in_gpu, int count,
    std::vector<heongpu::HEStream>& s)
{
    int num_threads = s.size();

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < count; i++)
    {
        int threadID = omp_get_thread_num();

        heongpu::Ciphertext ciphertext1_in_gpu(ciphertext1_in_cpu[i], context,
                                               s[threadID]);
        heongpu::Plaintext plaintext_in_gpu(plaintext_in_cpu[i], context,
                                            s[threadID]);

        heongpu::Ciphertext temp1(context, s[threadID]);
        operators.multiply_plain(ciphertext1_in_gpu, plaintext_in_gpu, temp1,
                                 s[threadID]);

        heongpu::Ciphertext ciphertext2_in_gpu(ciphertext2_in_cpu[i], context,
                                               s[threadID]);

        heongpu::Ciphertext temp2(context, s[threadID]);
        operators.sub(temp1, ciphertext2_in_gpu, temp2, s[threadID]);

        heongpu::Ciphertext ciphertext3_in_gpu(ciphertext3_in_cpu[i], context,
                                               s[threadID]);

        operators.multiply_inplace(temp2, ciphertext3_in_gpu, s[threadID]);

        operators.relinearize_inplace(temp2, relinkeys, s[threadID]);

        operators.add_inplace(temp1, temp2, s[threadID]);

        heongpu::Ciphertext result(
            context, s[threadID]); // DO not forget result stream is
                                   // s[threadID], not default or others!
        operators.rotate_rows(temp1, result, galoiskeys, 5, s[threadID]);

        // Ensure the ciphertext_result_in_gpu's streams, while using different
        // place!
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

    heongpu::HEOperator operators(context);

    // Assume ciphertexts and plaintexts come from network(from CPU RAM)
    std::cout << "Random ciphertext generetes in CPU... " << std::endl;
    int total_size = 64;
    std::vector<std::vector<Data>> ciphertext1_in_cpu;
    std::vector<std::vector<Data>> ciphertext2_in_cpu;
    std::vector<std::vector<Data>> ciphertext3_in_cpu;
    std::vector<std::vector<Data>> plaintext_in_cpu;

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

    int num_threads = 4; // it depends on your application and devices
    std::cout << "HEStream generetes." << std::endl;
    std::vector<heongpu::HEStream> s;
    for (int i = 0; i < num_threads; i++)
    {
        heongpu::HEStream inner(context);
        s.push_back(inner);
    }

    std::cout << "Operations starts..." << std::endl;
    std::vector<heongpu::Ciphertext> results(total_size);
    multi_stream_function_way1(context, operators, relin_key, galois_key,
                               ciphertext1_in_cpu, ciphertext2_in_cpu,
                               ciphertext3_in_cpu, plaintext_in_cpu, results,
                               total_size, s);
    std::cout << "Done." << std::endl;

    std::cout << "To see differece, check streams view with using NVIDIA "
                 "Nsight Systems."
              << std::endl;

    return EXIT_SUCCESS;
}
