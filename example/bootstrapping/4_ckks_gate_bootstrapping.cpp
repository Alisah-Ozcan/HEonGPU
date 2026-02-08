// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

int main(int argc, char* argv[])
{
    // Initialize encryption parameters for the CKKS scheme.
    heongpu::HEContext<heongpu::Scheme::CKKS> context =
        heongpu::GenHEContext<heongpu::Scheme::CKKS>(
            heongpu::sec_level_type::none);
    size_t poly_modulus_degree = 4096;
    context->set_poly_modulus_degree(poly_modulus_degree);

    // Last modulus has to be three times the value of scale, otherwise, it will
    // fail. 3298535538689ULL =~ 3 * pow(2.0, 40)
    context->set_coeff_modulus_values(
        {3298535538689ULL, 1099512938497ULL, 1099515691009ULL,
         1099516870657ULL, 1099521458177ULL, 1099522375681ULL,
         1099523555329ULL, 1099525128193ULL, 1099526176769ULL,
         1099529060353ULL, 1099535220737ULL, 1099536138241ULL,
         1099537580033ULL, 1099538104321ULL, 1099540725761ULL,
         1099540856833ULL, 1099543085057ULL, 1099544002561ULL,
         1099544395777ULL, 1099548327937ULL},
        {3298535669761ULL, 3298538684417ULL, 3298540650497ULL});
    context->generate();
    context->print_parameters();

    // The scale is set to 2^40, resulting in 40 bits of precision before the
    // decimal point.
    double scale = pow(2.0, 40);

    // Generate keys: the public key for encryption, the secret key for
    // decryption and evaluation key(relinkey) for relinearization.
    heongpu::HEKeyGenerator<heongpu::Scheme::CKKS> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::CKKS> secret_key(
        context,
        16); // hamming weight is 16 in this example
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::CKKS> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<heongpu::Scheme::CKKS> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    // Initialize Encoder, Encryptor, Evaluator, and Decryptor. The Encoder will
    // encode the message for SIMD operations. The Encryptor will use the public
    // key to encrypt data, while the Decryptor will use the secret key to
    // decrypt it. The Evaluator will handle operations on the encrypted data.
    heongpu::HEEncoder<heongpu::Scheme::CKKS> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::CKKS> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::CKKS> decryptor(context, secret_key);
    // heongpu::HEOperator operators(context);
    heongpu::HELogicOperator<heongpu::Scheme::CKKS> operators(context, encoder,
                                                              scale);

    // Generate simple vector in CPU.
    const int slot_count = poly_modulus_degree / 2;
    std::cout << "Plaintext vector size: " << slot_count << std::endl;
    std::vector<double> message1(slot_count, 1);
    message1[0] = 0;
    message1[2] = 0;

    std::vector<double> message2(slot_count, 1);

    //  Transfer that vector from CPU to GPU and Encode that simple vector in
    //  GPU.
    heongpu::Plaintext<heongpu::Scheme::CKKS> P1(context);
    encoder.encode(P1, message1, scale);

    heongpu::Plaintext<heongpu::Scheme::CKKS> P2(context);
    encoder.encode(P2, message2, scale);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C1(context);
    encryptor.encrypt(C1, P1);

    heongpu::Ciphertext<heongpu::Scheme::CKKS> C2(context);
    encryptor.encrypt(C2, P2);

    // Check README.md for more detail information
    // CtoS_piece_ = [2,5]
    // StoC_piece_ = [2,5]
    // taylor_number_ = [6,15]
    // less_key_mode_ = true or false
    int StoC_piece = 3;
    heongpu::BootstrappingConfig boot_config(3, StoC_piece, 6, true);
    // Generates all bootstrapping parameters before bootstrapping
    // operators.generate_bootstrapping_parameters(encoder, scale, boot_config);
    operators.generate_bootstrapping_params(
        scale, boot_config,
        heongpu::logic_bootstrapping_type::GATE_BOOTSTRAPPING);

    std::vector<int> key_index = operators.bootstrapping_key_indexs();
    std::cout << "Total galois key needed for CKKS bootstrapping: "
              << key_index.size() << std::endl;
    heongpu::Galoiskey<heongpu::Scheme::CKKS> galois_key(context, key_index);

    // Generates all galois key needed for bootstrapping
    keygen.generate_galois_key(galois_key,
                               secret_key); // all galois keys are stored in GPU
    // keygen.generate_galois_key(galois_key, secret_key,
    // heongpu::ExecutionOptions().set_storage_type(heongpu::storage_type::HOST));
    // // all galois keys are stored in CPU

    // Drop all level until one level remain
    for (int i = 0; i < (19 - StoC_piece); i++)
    {
        operators.mod_drop_inplace(C1);
        operators.mod_drop_inplace(C2);
    }

    std::cout << "Depth before bootstrapping: " << C1.depth() << std::endl;

    // Bootstapping Operation
    heongpu::Ciphertext<heongpu::Scheme::CKKS> cipher_boot =
        operators.AND_bootstrapping(C1, C2, galois_key, relin_key);

    std::cout << "Depth after bootstrapping: " << cipher_boot.depth()
              << std::endl;

    heongpu::Plaintext<heongpu::Scheme::CKKS> P_res1(context);
    decryptor.decrypt(P_res1, cipher_boot);
    std::vector<Complex64> decrypted_1;
    encoder.decode(decrypted_1, P_res1);

    std::vector<double> expected_message(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        expected_message[i] =
            static_cast<double>(int(message1[i]) & int(message2[i]));
    }

    // Compute and print precision statistics
    heongpu::PrecisionStats prec_stats =
        heongpu::get_precision_stats(expected_message, decrypted_1);

    std::cout << "\n=== Bootstrapping Precision Statistics ===" << std::endl;
    std::cout << prec_stats.to_string() << std::endl;

    // for(int j = 0; j < slot_count; j++){
    for (int j = 0; j < 16; j++)
    {
        std::cout << j << "-> EXPECTED:" << expected_message[j]
                  << " - ACTUAL:" << decrypted_1[j] << std::endl;
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
