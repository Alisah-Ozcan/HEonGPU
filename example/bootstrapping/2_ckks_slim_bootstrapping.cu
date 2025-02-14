// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "../example_util.h"

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    // Initialize encryption parameters for the CKKS scheme.
    heongpu::Parameters context(
        heongpu::scheme_type::ckks,
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
        heongpu::sec_level_type::none);
    size_t poly_modulus_degree = 4096;
    context.set_poly_modulus_degree(poly_modulus_degree);

    context.set_coeff_modulus({60, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                               50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                               50, 50, 50, 50, 50, 50, 50, 50, 50},
                              {60, 60, 60});
    context.generate();
    context.print_parameters();

    // The scale is set to 2^50, resulting in 50 bits of precision before the
    // decimal point.
    double scale = pow(2.0, 50);

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization.
    heongpu::HEKeyGenerator keygen(context);
    heongpu::Secretkey secret_key(context,
                                  16); // hamming weight is 16 in this example
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Initialize Encoder, Encryptor, Evaluator, and Decryptor. The Encoder will
    // encode the message for SIMD operations. The Encryptor will use the public
    // key to encrypt data, while the Decryptor will use the secret key to
    // decrypt it. The Evaluator will handle operations on the encrypted data.
    heongpu::HEEncoder encoder(context);
    heongpu::HEEncryptor encryptor(context, public_key);
    heongpu::HEDecryptor decryptor(context, secret_key);
    // heongpu::HEOperator operators(context);
    heongpu::HEArithmeticOperator operators(context, encoder);

    // Generate simple vector in CPU.
    const int slot_count = poly_modulus_degree / 2;
    std::vector<double>
        message; // Message has to be Real Number for Slim bootstrapping
    for (int i = 0; i < slot_count; i++)
    {
        message.push_back(0.2);
    }

    //  Transfer that vector from CPU to GPU and Encode that simple vector in
    //  GPU.
    heongpu::Plaintext P1(context);
    encoder.encode(P1, message, scale);

    heongpu::Ciphertext C1(context);
    encryptor.encrypt(C1, P1);

    // Check README.md for more detail information
    // CtoS_piece_ = [2,5]
    // StoC_piece_ = [2,5]
    // taylor_number_ = [6,15]
    // less_key_mode_ = true or false
    int StoC_piece = 3;
    heongpu::BootstrappingConfig boot_config(3, StoC_piece, 7, true);
    // Generates all bootstrapping parameters before bootstrapping
    operators.generate_bootstrapping_params(scale, boot_config);

    std::vector<int> key_index = operators.bootstrapping_key_indexs();
    std::cout << "Total galois key needed for CKKS bootstrapping: "
              << key_index.size() << std::endl;
    heongpu::Galoiskey galois_key(
        context, key_index); // all galois keys are stored in GPU
    // heongpu::Galoiskey galois_key(context,key_index, false); // all galois
    // keys are stored in CPU

    // Generates all galois key needed for bootstrapping
    keygen.generate_galois_key(galois_key, secret_key);

    // Drop all level until one level remain
    for (int i = 0; i < (28 - StoC_piece); i++)
    {
        operators.mod_drop_inplace(C1);
    }

    std::cout << "Depth before bootstrapping: " << C1.depth() << std::endl;

    // Bootstapping Operation
    heongpu::Ciphertext cipher_boot =
        operators.slim_bootstrapping(C1, galois_key, relin_key);

    std::cout << "Depth after bootstrapping: " << cipher_boot.depth()
              << std::endl;

    heongpu::Plaintext P_res1(context);
    decryptor.decrypt(P_res1, cipher_boot);
    std::vector<Complex64> decrypted_1;
    encoder.decode(decrypted_1, P_res1);

    // for(int j = 0; j < slot_count; j++){
    for (int j = 0; j < 16; j++)
    {
        std::cout << j << "-> EXPECTED:" << message[j]
                  << " - ACTUAL:" << decrypted_1[j] << std::endl;
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
