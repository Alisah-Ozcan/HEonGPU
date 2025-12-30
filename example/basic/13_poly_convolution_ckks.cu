// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <random>
#include <cmath>

constexpr auto Scheme = heongpu::Scheme::CKKS;

static std::vector<double> cpu_negacyclic_convolution(
    const std::vector<double>& a, const std::vector<double>& b)
{
    const size_t n = a.size();
    std::vector<double> out(n, 0.0);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            size_t k = i + j;
            bool wrap = k >= n;
            k = wrap ? (k - n) : k;

            if (wrap)
            {
                out[k] -= a[i] * b[j]; // x^N = -1
            }
            else
            {
                out[k] += a[i] * b[j];
            }
        }
    }

    return out;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,heongpu::sec_level_type::none);

    const size_t poly_modulus_degree = 4096;
    context.set_poly_modulus_degree(poly_modulus_degree);

    // Keep it small: only needed to initialize NTT tables for poly mult.
    // No rescale is performed in this example.
    // Match scale (2^30) with 30-bit primes so that after a single
    // multiplication and rescale, the scale stays close to 2^30.
    context.set_coeff_modulus_bit_sizes({60, 30, 30}, {60});
    context.generate();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<Scheme> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::Relinkey<Scheme> relin_key(context);
    keygen.generate_relin_key(relin_key, secret_key);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, public_key);
    heongpu::HEDecryptor<Scheme> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<Scheme> operators(context, encoder);

    const int n = context.get_poly_modulus_degree();
    const int q_count = context.get_ciphertext_modulus_count();
    const auto primes = context.get_key_modulus();

    std::mt19937_64 rng(0xC0FFEE);

    // Coefficient inputs (as doubles) to be encoded into plaintext polynomials.
    // Use a CKKS-friendly scale aligned with the modulus chain.
    const double scale = std::pow(2.0, 30);
    // Use small real values, similar to typical NN inference weights/activations.
    std::uniform_real_distribution<double> dist(-0.1, 0.1);
    std::vector<double> a_coeff(static_cast<size_t>(n), 0.0);
    std::vector<double> b_coeff(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; i++)
    {
        a_coeff[i] = dist(rng);
        b_coeff[i] = dist(rng);
    }

    heongpu::Plaintext<Scheme> Pa(context);
    heongpu::Plaintext<Scheme> Pb(context);
    encoder.encode_coeffs(Pa, a_coeff, scale);
    encoder.encode_coeffs(Pb, b_coeff, scale);

    heongpu::Ciphertext<Scheme> Ca(context);
    heongpu::Ciphertext<Scheme> Cb(context);
    encryptor.encrypt(Ca, Pa);
    encryptor.encrypt(Cb, Pb);

    // Ciphertext multiplication corresponds to negacyclic polynomial
    // convolution modulo x^N + 1 on encoded coefficients (in each RNS prime).
    operators.multiply_inplace(Ca, Cb);
    operators.relinearize_inplace(Ca, relin_key);
    operators.rescale_inplace(Ca);

    heongpu::Plaintext<Scheme> Pout(context);
    decryptor.decrypt(Pout, Ca);

    std::vector<double> decoded;
    encoder.decode_coeffs(decoded, Pout);

    std::vector<double> ref = cpu_negacyclic_convolution(a_coeff, b_coeff);

    const double abs_eps = 2e-3;
    const double rel_eps = 1e-6;
    for (int i = 0; i < n; i++)
    {
        const double abs_err = std::abs(decoded[i] - ref[i]);
        const double tol = abs_eps + rel_eps * std::max(1.0, std::abs(ref[i]));
        if (abs_err > tol)
        {
            std::cout << "FAIL at i=" << i << " got=" << decoded[i]
                      << " ref=" << ref[i]
                      << " abs_err=" << abs_err << " tol=" << tol
                      << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::cout << "Ciphertext polynomial convolution (coeff encoding): PASS"
              << std::endl;
    return EXIT_SUCCESS;
}
