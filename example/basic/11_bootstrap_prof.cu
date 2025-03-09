#include "heongpu.cuh"
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

int main(int argc, char* argv[]) {
    cudaSetDevice(0); // Use it for memory pool

    std::vector<double> bootstrap_times;

    for (int run = 0; run < 10; ++run) {
        // Initialize encryption parameters for the CKKS scheme.
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);
        size_t poly_modulus_degree = 4096;
        context.set_poly_modulus_degree(poly_modulus_degree);

        context.set_coeff_modulus({60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                    50, 50, 50, 50, 50, 50, 50, 50, 50},
                                   {60, 60, 60});
        context.generate();
        //context.print_parameters(); //optional, if you want to print parameters each run.

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

        heongpu::HEEncoder encoder(context);
        heongpu::HEEncryptor encryptor(context, public_key);
        heongpu::HEDecryptor decryptor(context, secret_key);
        heongpu::HEArithmeticOperator operators(context, encoder);

        // Generate simple vector in CPU.
        const int slot_count = poly_modulus_degree / 2;
        std::vector<Complex64> message;
        for (int i = 0; i < slot_count; i++) {
            message.push_back(Complex64(0.2, 0.4));
        }

        //  Transfer that vector from CPU to GPU and Encode that simple vector in
        //  GPU.
        heongpu::Plaintext P1(context);
        encoder.encode(P1, message, scale);

        heongpu::Ciphertext C1(context);
        encryptor.encrypt(C1, P1);

        
        //For Depth 30->29, use 5, 5, 11, true
        //For depth 30-> 26 use 4, 3, 11, true
        //For depth 30-> 23 use 2, 3, 10, true
        //For depth 30-> 20 use 2, 2, 8, true
        int StoC_piece = 2;
        heongpu::BootstrappingConfig boot_config(2, StoC_piece, 8, true);
        operators.generate_bootstrapping_params(scale, boot_config);

        std::vector<int> key_index = operators.bootstrapping_key_indexs();
        heongpu::Galoiskey galois_key(context, key_index);
        keygen.generate_galois_key(galois_key, secret_key);

        for (int i = 0; i < 31 - 1; i++) {
            operators.mod_drop_inplace(C1);
        }
        std::cout << "Depth before bootstrapping: " << C1.depth() << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        heongpu::Ciphertext cipher_boot =
            operators.regular_bootstrapping(C1, galois_key, relin_key);

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Depth after bootstrapping: " << cipher_boot.depth()
              << std::endl;
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        bootstrap_times.push_back(std::chrono::duration<double>(end - start).count());

        //Optional decryption and print of first 16 values.
        heongpu::Plaintext P_res1(context);
        decryptor.decrypt(P_res1, cipher_boot);
        std::vector<Complex64> decrypted_1;
        encoder.decode(decrypted_1, P_res1);

        // for(int j = 0; j < slot_count; j++){
        // if (run == 9){ // only print results of last run.
        //     for (int j = 0; j < 16; j++) {
        //         std::cout << j << "-> EXPECTED:" << message[j]
        //                   << " - ACTUAL:" << decrypted_1[j] << std::endl;
        //     }
        //     std::cout << std::endl;
        // }

    }

    double mean_time = std::accumulate(bootstrap_times.begin(), bootstrap_times.end(), 0.0) / bootstrap_times.size();
    std::cout << "Mean bootstrapping time: " << mean_time << " seconds" << std::endl;

    return EXIT_SUCCESS;
}