// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "heongpu.cuh"
#include "example_util.h"
#include <omp.h>

// These examples have been developed with reference to the Microsoft SEAL
// library.

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    // Initialize encryption parameters for the CKKS scheme.
    heongpu::Parameters context(
        heongpu::scheme_type::ckks,
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    // Set the polynomial modulus degree. This controls the complexity
    // of the computations and the size of the ciphertexts. Larger values
    // allow more complex operations but increase computational cost.
    // Here, we choose 8192, a moderate value.
    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);

    // Alternative -> select sec_level_type as sec128(default), sec192,
    // sec256, none
    // heongpu::Parameters context(
    //    heongpu::scheme_type::bfv,
    //    heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
    //    heongpu::sec_level_type::sec128);

    // A recommended approach is to align the initial scale and the primes in
    // the coeff_modulus closely. After multiplication and rescaling, this
    // alignment helps maintain a consistent scale throughout computations. For
    // a circuit with depth D, D rescalings are required, removing D primes from
    // the coeff_modulus. The remaining prime should be slightly larger than the
    // scale to preserve the plaintext's precision.

    // A suggested for parameter selection: Start with a 60-bit prime in the
    // Q(ciphertext modulus) for maximum decryption precision. Use 60-bit prime
    // for P, ensuring it matches the size of the others (Q_tilda = QxP). Select
    // intermediate primes that are similar in value.
    context.set_coeff_modulus({60, 30, 30, 30}, {60});

    // Generate a HEonGPU context with these parameters. The context checks the
    // validity of the parameters and provides various utilities needed for
    // encryption, decryption, and evaluation.
    context.generate();
    context.print_parameters();

    // The scale is set to 2^30, resulting in 30 bits of precision before the
    // decimal point and approximately 20-30 bits after at the final level.
    // Intermediate primes, close to 30 bits, ensure consistent scale
    // stabilization.
    double scale = pow(2.0, 30);

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization..
    heongpu::HEKeyGenerator keygen(context);
    heongpu::Secretkey secret_key(context);
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
    heongpu::HEOperator operators(context);

    // Generate simple matrix in CPU.
    const int slot_count = poly_modulus_degree / 2;
    std::cout << "Plaintext vector size: " << slot_count << std::endl;
    std::vector<double> message(slot_count, 3);
    message[0] = 10;
    message[1] = 20;
    message[2] = 30;
    message[3] = 40;
    message[4] = 0.5;

    // Alternative: HostVector use use pinned memory and memory pool, provide
    // faster data transfer between CPU and GPU.(and vice versa)
    // heongpu::HostVector<double> message(slot_count, 3);
    // message[0] = 10;
    // message[1] = 20;
    // message[2] = 30;
    // message[3] = 40;
    // message[4] = 0.5;

    //  [10,  20,  30,  40,  0.5,  3, ...,  3]

    std::cout << "Message plaintext vector:" << std::endl;
    display_vector(message);

    std::cout << "Transfer vector to GPU and encode vector." << std::endl;
    heongpu::Plaintext P1(context);
    //  Transfer that vector from CPU to GPU and Encode that simple vector in
    //  GPU.
    encoder.encode(P1, message, scale);

    // Alternative way!
    // heongpu::Plaintext P1(context);
    // encoder.encode(P1, message, scale);

    // This plaintext(in GPU) value will be converted into an encrypted form
    // (ciphertext in GPU), which can be used for secure computations.
    std::cout << "Encrypt plaintext vector." << std::endl;
    heongpu::Ciphertext C1(context);
    encryptor.encrypt(C1, P1);

    std::cout << "Square message homomorphically." << std::endl;
    operators.multiply_inplace(C1, C1);
    std::cout << "Remove non-linear part of ciphertext." << std::endl;
    operators.relinearize_inplace(C1, relin_key);
    std::cout << "Divede ciphertext to last modulus and reduce noise."
              << std::endl;
    operators.rescale_inplace(C1);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext P2(context);
    decryptor.decrypt(P2, C1);

    std::cout << "Decode Plaintext and Transfer data from GPU to CPU."
              << std::endl;
    std::vector<double> check1;
    encoder.decode(check1, P2);

    //  Approximately:
    //  [100,  400, 900, 1600, 0.25, 9, ...,  9]

    std::cout << "Check result:" << std::endl;
    display_vector(check1);

    std::vector<double> message2(slot_count, 0.25); // In CPU

    //  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ..., 0.5]

    std::cout << "Message2 plaintext vector." << std::endl;
    display_vector(message2);

    heongpu::Plaintext P3(context);
    encoder.encode(P3, message2, scale);

    std::cout << "Drop the P3 last modulus." << std::endl;
    operators.mod_drop_inplace(
        P3); // for now, do it manually for each levels you need to go.

    heongpu::Ciphertext C2(context);
    std::cout << "Mutiply ciphertext with plaintext." << std::endl;
    operators.multiply_plain(C1, P3, C2);
    std::cout << "Add ciphertext to itself." << std::endl;
    operators.add_inplace(C2, C2);

    operators.rescale_inplace(C2);

    std::cout << "Decrypt result." << std::endl;
    heongpu::Plaintext P4(context);
    decryptor.decrypt(P4, C2);

    std::vector<double> check2;
    encoder.decode(check2, P4);

    //  [50, 200, 450, 800, 4.5, 4.5, ..., 4.5]

    std::cout << "Check result2:" << std::endl;
    display_vector(check2);

    return EXIT_SUCCESS;
}
