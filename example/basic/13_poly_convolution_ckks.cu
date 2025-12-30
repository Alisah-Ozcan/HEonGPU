// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <random>

constexpr auto Scheme = heongpu::Scheme::CKKS;

static std::vector<Data64> cpu_negacyclic_convolution_mod(
    const std::vector<Data64>& a, const std::vector<Data64>& b, Data64 mod)
{
    const size_t n = a.size();
    std::vector<Data64> out(n, 0);

    for (size_t i = 0; i < n; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            size_t k = i + j;
            bool wrap = k >= n;
            k = wrap ? (k - n) : k;

            __uint128_t prod = static_cast<__uint128_t>(a[i]) *
                               static_cast<__uint128_t>(b[j]);
            Data64 term = static_cast<Data64>(prod % mod);

            if (wrap)
            {
                out[k] = (out[k] + (mod - term)) % mod; // x^N = -1
            }
            else
            {
                out[k] = (out[k] + term) % mod;
            }
        }
    }

    return out;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0); // Use it for memory pool

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    const size_t poly_modulus_degree = 1024;
    context.set_poly_modulus_degree(poly_modulus_degree);

    // Keep it small: only needed to initialize NTT tables for poly mult.
    // No rescale is performed in this example.
    context.set_coeff_modulus_bit_sizes({50, 50, 50}, {50});
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
    heongpu::HEConvolution<Scheme> conv(context);

    const int n = context.get_poly_modulus_degree();
    const int q_count = context.get_ciphertext_modulus_count();
    const auto primes = context.get_key_modulus();

    std::mt19937_64 rng(0xC0FFEE);

    // Coefficient inputs (as doubles) to be encoded into plaintext polynomials.
    // Use scale=1.0 so the polynomial coefficients match integer convolution.
    const double scale = 1.0;
    std::uniform_int_distribution<int64_t> signed_dist(-(1LL << 20),
                                                       (1LL << 20));
    std::vector<double> a_coeff(static_cast<size_t>(n), 0.0);
    std::vector<double> b_coeff(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; i++)
    {
        a_coeff[i] = static_cast<double>(signed_dist(rng));
        b_coeff[i] = static_cast<double>(signed_dist(rng));
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

    heongpu::Plaintext<Scheme> Pout(context);
    decryptor.decrypt(Pout, Ca);

    const int out_rns_count = Pout.size() / n;
    heongpu::DeviceVector<Data64> out_ntt(Pout.size());
    cudaMemcpy(out_ntt.data(), Pout.data(), Pout.size() * sizeof(Data64),
               cudaMemcpyDeviceToDevice);

    conv.to_coeff_domain_inplace(out_ntt, /*poly_count=*/1, out_rns_count);

    std::vector<Data64> c_host(static_cast<size_t>(n) * out_rns_count);
    cudaMemcpy(c_host.data(), out_ntt.data(),
               c_host.size() * sizeof(Data64),
               cudaMemcpyDeviceToHost);

    // Compare against CPU reference in each modulus used by the decrypted plaintext.
    for (int qi = 0; qi < out_rns_count; qi++)
    {
        const Data64 q = primes[qi].value;
        std::vector<Data64> a_mod(static_cast<size_t>(n));
        std::vector<Data64> b_mod(static_cast<size_t>(n));
        for (int i = 0; i < n; i++)
        {
            int64_t ai = static_cast<int64_t>(llround(a_coeff[i]));
            int64_t bi = static_cast<int64_t>(llround(b_coeff[i]));
            int64_t am = ai % static_cast<int64_t>(q);
            int64_t bm = bi % static_cast<int64_t>(q);
            if (am < 0)
                am += static_cast<int64_t>(q);
            if (bm < 0)
                bm += static_cast<int64_t>(q);
            a_mod[i] = static_cast<Data64>(am);
            b_mod[i] = static_cast<Data64>(bm);
        }

        std::vector<Data64> ref = cpu_negacyclic_convolution_mod(a_mod, b_mod, q);

        for (int i = 0; i < n; i++)
        {
            if (c_host[static_cast<size_t>(qi) * n + i] != ref[i])
            {
                std::cout << "FAIL at qi=" << qi << " i=" << i << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    std::cout << "Ciphertext polynomial convolution (coeff encoding): PASS"
              << std::endl;
    return EXIT_SUCCESS;
}
