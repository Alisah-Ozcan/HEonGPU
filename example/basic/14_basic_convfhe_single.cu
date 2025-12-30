// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

static inline void add_coeff_with_neg_exp(std::vector<double>& poly, int exp,
                                          double value)
{
    const int N = static_cast<int>(poly.size());
    if (exp >= 0)
    {
        if (exp >= N)
        {
            throw std::invalid_argument("Exponent out of bounds");
        }
        poly[exp] += value;
        return;
    }

    // In R = Z[X]/(X^N + 1): X^t := -X^{N+t} for t<0.
    const int idx = N + exp;
    if (idx < 0 || idx >= N)
    {
        throw std::invalid_argument("Negative exponent out of bounds");
    }
    poly[idx] -= value;
}

static std::vector<double>
negacyclic_multiply(const std::vector<double>& a, const std::vector<double>& b)
{
    const int N = static_cast<int>(a.size());
    if (static_cast<int>(b.size()) != N)
    {
        throw std::invalid_argument("Polynomial size mismatch");
    }

    std::vector<double> out(static_cast<size_t>(N), 0.0);

    for (int i = 0; i < N; i++)
    {
        const double ai = a[i];
        if (ai == 0.0)
        {
            continue;
        }
        for (int j = 0; j < N; j++)
        {
            const double prod = ai * b[j];
            if (prod == 0.0)
            {
                continue;
            }

            const int s = i + j;
            if (s < N)
            {
                out[s] += prod;
            }
            else
            {
                out[s - N] -= prod; // X^N = -1
            }
        }
    }

    return out;
}

static std::vector<double> valid_conv2d(const std::vector<double>& input,
                                        const std::vector<double>& kernel,
                                        int w, int k)
{
    const int d = w - k + 1;
    std::vector<double> out(static_cast<size_t>(d * d), 0.0);

    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < d; j++)
        {
            double acc = 0.0;
            for (int u = 0; u < k; u++)
            {
                for (int v = 0; v < k; v++)
                {
                    acc += input[(i + u) * w + (j + v)] * kernel[u * k + v];
                }
            }
            out[i * d + j] = acc;
        }
    }

    return out;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    // Demo parameters.
    const int w = 8;
    const int k = 3;
    const int d = w - k + 1;

    // Choose N as the CKKS poly modulus degree.
    const size_t poly_modulus_degree = 1024;

    // CKKS parameters: keep scale ~ 2^30 stable after one mul+rescale.
    const double scale = std::pow(2.0, 30);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> I(static_cast<size_t>(w * w), 0.0);
    std::vector<double> K(static_cast<size_t>(k * k), 0.0);
    for (auto& x : I)
        x = dist(rng);
    for (auto& x : K)
        x = dist(rng);

    // Reference 1: direct VALID convolution.
    std::vector<double> ref_direct = valid_conv2d(I, K, w, k);

    // Proposition 1 polynomial construction.
    std::vector<double> I_poly(poly_modulus_degree, 0.0);
    std::vector<double> K_poly(poly_modulus_degree, 0.0);

    // I(X)=Σ I[i,j] * X^{(i-k)*w + j}
    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < w; j++)
        {
            const int exp = (i - k) * w + j;
            add_coeff_with_neg_exp(I_poly, exp, I[i * w + j]);
        }
    }

    // K(X)=Σ K[i,j] * X^{w*k - (i*w + j)}
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j++)
        {
            const int exp = w * k - (i * w + j);
            add_coeff_with_neg_exp(K_poly, exp, K[i * k + j]);
        }
    }

    // Reference 2: polynomial construction + naive negacyclic multiply in R.
    std::vector<double> ref_poly = negacyclic_multiply(I_poly, K_poly);

    double max_ref_mismatch = 0.0;
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < d; j++)
        {
            const int idx = i * w + j;
            const double want = ref_direct[i * d + j];
            const double got = ref_poly[idx];
            max_ref_mismatch = std::max(max_ref_mismatch, std::abs(got - want));
        }
    }

    std::cout << "Ref check (direct vs poly): max_abs_err=" << max_ref_mismatch
              << std::endl;

    // FHE flow: coefficient encoding -> encrypt -> ciphertext multiplication.
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
    context.set_poly_modulus_degree(poly_modulus_degree);
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

    heongpu::Plaintext<Scheme> P_I(context);
    heongpu::Plaintext<Scheme> P_K(context);
    encoder.encode_coeffs(P_I, I_poly, scale);
    encoder.encode_coeffs(P_K, K_poly, scale);

    heongpu::Ciphertext<Scheme> C_I(context);
    heongpu::Ciphertext<Scheme> C_K(context);
    encryptor.encrypt(C_I, P_I);
    encryptor.encrypt(C_K, P_K);

    operators.multiply_inplace(C_I, C_K);
    operators.relinearize_inplace(C_I, relin_key);
    operators.rescale_inplace(C_I);

    heongpu::Plaintext<Scheme> P_prod(context);
    decryptor.decrypt(P_prod, C_I);

    std::vector<double> decoded_poly;
    encoder.decode_coeffs(decoded_poly, P_prod);

    // Extract 2D VALID conv outputs: coefficient (i*w + j).
    double max_abs_err = 0.0;
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < d; j++)
        {
            const int idx = i * w + j;
            const double want = ref_direct[i * d + j];
            const double got = decoded_poly[idx];
            max_abs_err = std::max(max_abs_err, std::abs(got - want));
        }
    }

    std::cout << "FHE VALID conv demo: w=" << w << " k=" << k << " d=" << d
              << " max_abs_err=" << max_abs_err << std::endl;

    return EXIT_SUCCESS;
}

