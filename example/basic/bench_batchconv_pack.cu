// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/heongpu.hpp>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return static_cast<double>(ms);
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    const int B = 8;
    const int w = 16;
    const int k = 3;
    const int d = w - k + 1;

    const size_t N = 8192;
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
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    context.set_coeff_modulus_bit_sizes({60, 30, 30}, {60});
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
    heongpu::HEDecryptor<Scheme> decryptor(context, sk);
    heongpu::HEArithmeticOperator<Scheme> ops(context, encoder);
    heongpu::HEBatchConvPack<Scheme> packer(context);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // 1) Build + encrypt packed input.
    std::vector<double> I_sparse = packer.build_input_sparse_coeffs(inputs, B, w, k);
    heongpu::Plaintext<Scheme> P_I(context);
    encoder.encode_coeffs(P_I, I_sparse, scale);
    heongpu::Ciphertext<Scheme> C_I(context);
    encryptor.encrypt(C_I, P_I);

    // Warm-up.
    {
        std::vector<double> K0 = packer.build_kernel_sparse_coeffs_for_out(kernels, 0, B, w, k);
        heongpu::Plaintext<Scheme> P_K0(context);
        encoder.encode_coeffs(P_K0, K0, scale);
        heongpu::Ciphertext<Scheme> tmp = C_I;
        ops.multiply_plain_inplace(tmp, P_K0);
        ops.rescale_inplace(tmp);
        packer.monomial_shift_inplace(tmp, 1);
    }
    cudaDeviceSynchronize();

    // 2) Benchmark: B plaintext multiplications + B rescale.
    std::vector<heongpu::Ciphertext<Scheme>> ct_b(static_cast<size_t>(B));

    cudaEventRecord(ev_start);
    for (int out = 0; out < B; out++)
    {
        std::vector<double> Kb = packer.build_kernel_sparse_coeffs_for_out(kernels, out, B, w, k);
        heongpu::Plaintext<Scheme> P_K(context);
        encoder.encode_coeffs(P_K, Kb, scale);

        ct_b[out] = C_I;
        ops.multiply_plain_inplace(ct_b[out], P_K);
        ops.rescale_inplace(ct_b[out]);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    const double ms_batchconv = elapsed_ms(ev_start, ev_stop);

    // 3) Benchmark: pack (B-1) monomial shifts + (B-1) adds.
    cudaEventRecord(ev_start);
    for (int b = 1; b < B; b++)
    {
        packer.monomial_shift_inplace(ct_b[b], b);
        ops.add_inplace(ct_b[0], ct_b[b]);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    const double ms_pack = elapsed_ms(ev_start, ev_stop);

    std::cout << "bench_batchconv_pack:"
              << " B=" << B << " w=" << w << " k=" << k << " d=" << d
              << " N=" << N << std::endl;
    std::cout << "Ops:"
              << " mul_plain=" << B << " rescale=" << B
              << " monomial_shift=" << (B - 1) << " add=" << (B - 1)
              << " rotate=0" << std::endl;
    std::cout << "Time(ms): batchconv=" << ms_batchconv
              << " pack=" << ms_pack << std::endl;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return EXIT_SUCCESS;
}

