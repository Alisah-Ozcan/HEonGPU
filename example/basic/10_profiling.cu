//Will profile multiplication for levels 1, 5, 10, 15 over 10 runs each and take the mean
#include "heongpu.cuh"
#include "../example_util.h"
#include <omp.h>
#include <unistd.h>

int main(int argc, char* argv[])
{
    cudaSetDevice(0); 
    const int num_contexts = 10;

    std::vector<heongpu::Parameters> contexts;
    std::vector<heongpu::Secretkey> secret_keys;
    std::vector<heongpu::Publickey> public_keys;
    std::vector<heongpu::Relinkey> relin_keys;
    std::vector<heongpu::HEEncoder> encoders;
    std::vector<heongpu::HEEncryptor> encryptors;
    std::vector<heongpu::HEDecryptor> decryptors;
    std::vector<heongpu::HEArithmeticOperator> operators;

    contexts.reserve(num_contexts);
    secret_keys.reserve(num_contexts);
    public_keys.reserve(num_contexts);
    relin_keys.reserve(num_contexts);
    encoders.reserve(num_contexts);
    encryptors.reserve(num_contexts);
    decryptors.reserve(num_contexts);
    operators.reserve(num_contexts);

    size_t poly_modulus_degree = 4096;

    // Initialize multiple contexts and associated objects
    for (int i = 0; i < num_contexts; i++)
    {
        // Create and configure context
        heongpu::Parameters context(
            heongpu::scheme_type::ckks,
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
            heongpu::sec_level_type::none);

        context.set_poly_modulus_degree(poly_modulus_degree);
        context.set_coeff_modulus({60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
                                    50, 50, 50, 50, 50, 50, 50, 50, 50},
                                    {60, 60, 60});
        context.generate();

        // Move context into vector
        contexts.push_back(std::move(context));

        // Generate keys
        heongpu::HEKeyGenerator keygen(contexts[i]);
        heongpu::Secretkey secret_key(contexts[i], 16);
        keygen.generate_secret_key(secret_key);
        secret_keys.push_back(std::move(secret_key));

        heongpu::Publickey public_key(contexts[i]);
        keygen.generate_public_key(public_key, secret_keys[i]);
        public_keys.push_back(std::move(public_key));

        heongpu::Relinkey relin_key(contexts[i]);
        keygen.generate_relin_key(relin_key, secret_keys[i]);
        relin_keys.push_back(std::move(relin_key));

        // Initialize encoding, encryption, and arithmetic operators
        encoders.emplace_back(contexts[i]);
        encryptors.emplace_back(contexts[i], public_keys[i]);
        decryptors.emplace_back(contexts[i], secret_keys[i]);
        operators.emplace_back(contexts[i], encoders[i]);
    }
    double scale = pow(2.0, 50);
    // Print parameters for verification
    // for (int i = 0; i < num_contexts; i++)
    // {
    //     std::cout << "Context " << i << " initialized with parameters:" << std::endl;
    //     contexts[i].print_parameters();
    // }

    std::cout << "\t\t multiply \t rescale \t modswitch" << std::endl;
    
    std::vector<int> levels = {1, 5, 10, 15};
    for (int level : levels)
    {
        sleep(5);
        std::vector<double> multiplication_times;
        std::vector<double> rescale_times;
        std::vector<double> modswitch_times;
        for (int i = 0; i < num_contexts; i++)
        {
            // Encode and encrypt the number 4
            std::vector<double> plaintext = {4.0};
            heongpu::Plaintext P1(contexts[i]);
            encoders[i].encode(P1, plaintext, scale);

            heongpu::Ciphertext C1(contexts[i]);
            encryptors[i].encrypt(C1, P1);

            // Modulus switching to the required level
            
            while (31-1-1-C1.depth() > level){
                //std::cout << "Depth is " << C1.depth() <<std::endl;
                operators[i].mod_drop_inplace(C1);
            }

            // Time multiplication
            auto start = std::chrono::high_resolution_clock::now();
            operators[i].multiply_inplace(C1, C1);
            auto end = std::chrono::high_resolution_clock::now();
            // Print timing result
            std::chrono::duration<double> duration = end - start;
            double duration1 = std::chrono::duration<double>(end - start).count();
            multiplication_times.push_back(duration1);
            operators[i].relinearize_inplace(C1, relin_keys[i]);
            start = std::chrono::high_resolution_clock::now();
            operators[i].rescale_inplace(C1);
            end = std::chrono::high_resolution_clock::now();
            rescale_times.push_back(std::chrono::duration<double>(end - start).count());

            start = std::chrono::high_resolution_clock::now();
            operators[i].mod_drop_inplace(C1);
            end = std::chrono::high_resolution_clock::now();
            modswitch_times.push_back(std::chrono::duration<double>(end - start).count());
            //std::cout << "Context " << i << " Level " << level << " multiplication time: " << duration.count() << " seconds" << std::endl;
            
        }
        double mean_multiplication_time = std::accumulate(multiplication_times.begin(), multiplication_times.end(), 0.0) / num_contexts;
        double mean_rescale_time = std::accumulate(rescale_times.begin(), rescale_times.end(), 0.0) / num_contexts;
        double mean_modswitch_time = std::accumulate(modswitch_times.begin(), modswitch_times.end(), 0.0) / num_contexts;
        
        std::cout << "Level " << level << " \t " << mean_multiplication_time <<"\t" << mean_rescale_time<< "\t" << mean_modswitch_time << std::endl;

        //std::cout << "Mean multiplication time at level " << level << ": " << mean_multiplication_time << " seconds" << std::endl;
        //std::cout << "Mean rescale time at level " << level << ": " << mean_rescale_time << " seconds" << std::endl;
        //std::cout << "Mean modswitch time at level " << level << ": " << mean_modswitch_time << " seconds" << std::endl;
    }

    

    return 0;
}