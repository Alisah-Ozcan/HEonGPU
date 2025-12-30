// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include "../example_util.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

struct ConvStageBenchResult
{
    int B = 0;
    int w = 0;
    int k = 0;
    int d = 0;
    int N = 0;

    // Operation counts (homomorphic).
    int mul_plain = 0;
    int rescale = 0;
    int galois_apply = 0;
    int monomial_shift = 0;
    int add = 0;

    double ms_conv = 0.0;
    double ms_pack = 0.0;
};

static ConvStageBenchResult run_once(int B, int w, int k, int N)
{
    cudaSetDevice(0);

    ConvStageBenchResult r;
    r.B = B;
    r.w = w;
    r.k = k;
    r.N = N;
    r.d = w - k + 1;

    const double scale = std::pow(2.0, 30);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> inputs(static_cast<size_t>(B * w * w), 0.0);
    std::vector<double> kernels(static_cast<size_t>(B * B * k * k), 0.0);
    for (auto& x : inputs)
        x = dist(rng);
    for (auto& x : kernels)
        x = dist(rng);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(static_cast<size_t>(N));
    // Enough for multiply_plain+rescale (conv stage) + galois (projection).
    context.set_coeff_modulus_bit_sizes({60, 30, 30, 30}, {60});
    context.generate();

    heongpu::HEKeyGenerator<Scheme> keygen(context);
    heongpu::Secretkey<Scheme> sk(context);
    keygen.generate_secret_key(sk);

    heongpu::Publickey<Scheme> pk(context);
    keygen.generate_public_key(pk, sk);

    heongpu::Relinkey<Scheme> rlk(context);
    keygen.generate_relin_key(rlk, sk);

    heongpu::HEEncoder<Scheme> encoder(context);
    heongpu::HEEncryptor<Scheme> encryptor(context, pk);
    heongpu::HEArithmeticOperator<Scheme> ops(context, encoder);
    heongpu::HEBatchConvPack<Scheme> packer(context);

    // Galois key for PackCoeffs projection subgroup elements.
    const int twoN = 2 * N;
    const int step = twoN / B;
    std::vector<uint32_t> galois_elts;
    for (int t = 1; t < B; t++)
        galois_elts.push_back(static_cast<uint32_t>(1 + t * step));
    heongpu::Galoiskey<Scheme> gk_pack(context, galois_elts);
    keygen.generate_galois_key(gk_pack, sk);
    gk_pack.store_in_device();

    // Encrypt packed input once.
    std::vector<double> I_sparse = packer.build_input_sparse_coeffs(inputs, B, w, k);
    heongpu::Plaintext<Scheme> P_in(context);
    encoder.encode_coeffs(P_in, I_sparse, scale);
    heongpu::Ciphertext<Scheme> ct_in(context);
    encryptor.encrypt(ct_in, P_in);

    // ---- Conv stage timing: B plaintext multiplies + B rescales ----
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<heongpu::Ciphertext<Scheme>> ct_b(static_cast<size_t>(B));
    for (int out = 0; out < B; out++)
    {
        std::vector<double> Kb_sparse =
            packer.build_kernel_sparse_coeffs_for_out(kernels, out, B, w, k);
        heongpu::Plaintext<Scheme> Pk(context);
        encoder.encode_coeffs(Pk, Kb_sparse, scale);

        ct_b[out] = ct_in;
        ops.multiply_plain_inplace(ct_b[out], Pk);
        ops.rescale_inplace(ct_b[out]);
    }
    HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    // ---- Pack stage timing: stride-B projection + monomial shifts + adds ----
    auto t2 = std::chrono::high_resolution_clock::now();
    heongpu::Ciphertext<Scheme> ct_pack =
        packer.pack_coeffs_strideB(ops, ct_b, B, gk_pack);
    HEONGPU_CUDA_CHECK(cudaDeviceSynchronize());
    auto t3 = std::chrono::high_resolution_clock::now();

    (void)ct_pack;

    r.ms_conv = std::chrono::duration<double, std::milli>(t1 - t0).count();
    r.ms_pack = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // Operation counts for this scheme:
    r.mul_plain = B;
    r.rescale = B;
    r.galois_apply = B * (B - 1);
    r.monomial_shift = B - 1;
    r.add = (B * (B - 1)) + (B - 1);

    return r;
}

int main(int argc, char** argv)
{
    // Fixed parameters for micro-benchmark.
    const int B = 8;
    const int w = 16;
    const int N = 8192;

    for (int k : {3, 5, 7})
    {
        if (k > w)
            continue;
        ConvStageBenchResult r = run_once(B, w, k, N);
        std::cout << "bench_conv_stage: B=" << r.B << " w=" << r.w
                  << " k=" << r.k << " d=" << r.d << " N=" << r.N << "\n";
        std::cout << "Ops: mul_plain=" << r.mul_plain
                  << " rescale=" << r.rescale
                  << " galois_apply=" << r.galois_apply
                  << " monomial_shift=" << r.monomial_shift
                  << " add=" << r.add << "\n";
        std::cout << "Time(ms): conv=" << r.ms_conv << " pack=" << r.ms_pack
                  << "\n";
    }

    return 0;
}

