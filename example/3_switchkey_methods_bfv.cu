// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "example_util.h"
#include <omp.h>

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    heongpu::Parameters context(
        heongpu::scheme_type::bfv,
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    // heongpu::keyswitching_type::KEYSWITCHING_METHOD_III not supports rotation
    // because of key size, only supports relinearization!

    size_t poly_modulus_degree = 16384;
    context.set_poly_modulus_degree(poly_modulus_degree);

    context.set_default_coeff_modulus(
        1); // Pair size = 2, that is why P size should be 2.
    // Increasing the pair size improves performance(n >= 16384), but it also
    // reduces the noise budget.

    // Alternative -> first input is Q (ciphertext modulus), second one
    // is P and it determines to Q_tilda which is key modulus.(Q_tilda = QxP)
    // context.set_coeff_modulus({50, 50, 50, 50, 50, 50}, {50, 50});

    int plain_modulus = 786433;
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
    keygen.generate_galois_key(
        galois_key, secret_key); // This way will create 16(2x8) different power
                                 // of 2, if you need more change from define.h

    // Alternative way 1 -> calculate dedicated shift value:
    // std::vector<int> shifts = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, shifts);
    // keygen.generate_galois_key(galois_key, secret_key);

    // Alternative way 2 -> calculate dedicated galois value:
    // std::vector<uint32_t> galois = {0, -1, 128, -16, -32, -48, -64, -80, -96,
    // -112}; // example heongpu::Galoiskey galois_key(context, galois);
    // keygen.generate_galois_key(galois_key, secret_key);
    // use apply_galois instead of rotate_row!

    heongpu::Secretkey secret_key2(context);
    keygen.generate_secret_key(secret_key2);

    heongpu::Switchkey switch_key1_to_key2(context, false);
    keygen.generate_switch_key(switch_key1_to_key2, secret_key2, secret_key);

    heongpu::HEEncoder encoder(context);
    heongpu::HEEncryptor encryptor(context, public_key);
    // Initialize 2 different Decryptor with respect to different secretkeys.
    heongpu::HEDecryptor decryptor_key1(context, secret_key);
    heongpu::HEDecryptor decryptor_key2(context, secret_key2);
    heongpu::HEOperator operators(context);

    // Generate simple matrix in CPU.
    const int row_size = poly_modulus_degree / 2;
    std::cout << "Plaintext matrix row size: " << row_size << std::endl;
    std::vector<uint64_t> message(poly_modulus_degree, 8ULL); // In CPU
    message[0] = 1ULL;
    message[1] = 12ULL;
    message[2] = 23ULL;
    message[3] = 31ULL;
    message[row_size] = 7ULL;
    message[row_size + 1] = 54ULL;
    message[row_size + 2] = 6ULL;
    message[row_size + 3] = 100ULL;

    //  [1,  12,  23,  31,  8,  8, ...,  8]
    //  [7,  54,  6,  100,  8,  8, ...,  8]

    std::cout << "Message plaintext matrix:" << std::endl;
    display_matrix(message, row_size);

    std::cout << "Transfer matrix to GPU and encode matrix." << std::endl;
    heongpu::Plaintext P1(context);
    // Transfer that vector from CPU to GPU and Encode that simple vector in
    // GPU.
    encoder.encode(P1, message);

    std::cout << "Encrypt plaintext matrix." << std::endl;
    heongpu::Ciphertext C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "Initial noise budget in C1: "
              << decryptor_key1.remainder_noise_budget(C1) << " bits"
              << std::endl;

    std::cout << "Square message homomorphically." << std::endl;
    operators.multiply_inplace(C1, C1);
    std::cout << "Remove non-linear part of ciphertext." << std::endl;
    operators.relinearize_inplace(C1, relin_key);

    std::cout << "Noise budget in C1 after multiplication: "
              << decryptor_key1.remainder_noise_budget(C1) << " bits"
              << std::endl;

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext P2(context);
    decryptor_key1.decrypt(P2, C1);

    std::cout << "Decode plaintext and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<uint64_t> check1;
    encoder.decode(check1, P2);

    //  [1,  144,   529, 961,   64, 64, ...,  64]
    //  [49, 2916, 36,  10000, 64, 64, ...,  64]

    std::cout << "Check result:" << std::endl;
    display_matrix(check1, row_size);

    std::cout << "Shift row +3:" << std::endl;
    operators.rotate_rows_inplace(C1, galois_key, 3);

    std::cout << "Noise budget in C1 after multiplication: "
              << decryptor_key1.remainder_noise_budget(C1) << " bits"
              << std::endl;

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext P4(context);
    decryptor_key1.decrypt(P4, C1);

    std::vector<uint64_t> check2;
    encoder.decode(check2, P4);

    //  [961, 64, 64, 64, 64, ..., 64, 64,  1,144,529 ]
    //  [10000, 64, 64, 64, 64, ..., 64, 64, 49,2916, 36 ]

    std::cout << "Check result2:" << std::endl;
    display_matrix(check2, row_size);

    std::cout << "Switch columns:" << std::endl;
    operators.rotate_columns(C1, C1, galois_key);

    std::cout << "Noise budget in C1 after multiplication: "
              << decryptor_key1.remainder_noise_budget(C1) << " bits"
              << std::endl;

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext P5(context);
    decryptor_key1.decrypt(P5, C1);

    std::vector<uint64_t> check3;
    encoder.decode(check3, P5);

    //  [10000, 64, 64, 64, 64, ..., 64, 64, 49,2916, 36 ]
    //  [961, 64, 64, 64, 64, ..., 64, 64,  1,144,529 ]

    std::cout << "Check result3:" << std::endl;
    display_matrix(check3, row_size);

    std::cout << "Change ciphertext key, from secretkey1 to secretkey2:"
              << std::endl;
    operators.keyswitch(C1, C1, switch_key1_to_key2);

    std::cout << "Noise budget in C1 after multiplication: "
              << decryptor_key2.remainder_noise_budget(C1) << " bits"
              << std::endl; // do not forget this should be decryptor of key2

    std::cout << "Decrypt and decode result." << std::endl;
    heongpu::Plaintext P6(context);
    decryptor_key2.decrypt(
        P6, C1); // do not forget this should be decryptor of key2

    std::vector<uint64_t> check4;
    encoder.decode(check4, P6);

    //  [10000, 64, 64, 64, 64, ..., 64, 64, 49,2916, 36 ]
    //  [961, 64, 64, 64, 64, ..., 64, 64,  1,144,529 ]

    std::cout << "Check result4:" << std::endl;
    display_matrix(check4, row_size);

    return EXIT_SUCCESS;
}
