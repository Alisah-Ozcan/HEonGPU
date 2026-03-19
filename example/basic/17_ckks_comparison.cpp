// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <heongpu/heongpu.hpp>
#include "../example_util.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

// GPU performance is exploited in two ways:
//   Layer 1 (SIMD): one polynomial evaluation compares ALL n/2 slots in parallel.
//   Layer 2 (streams): K independent comparison ciphertexts dispatched to K
//                      CUDA streams via OpenMP, running concurrently on the GPU.

constexpr auto Scheme = heongpu::Scheme::CKKS;

// CUDA event timer.
class GPUTimer
{
    cudaEvent_t start_, stop_;

  public:
    GPUTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GPUTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_); }
    float stop()
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
};

// =========================================================================
// Manual degree-3 sign approximation: sign(x) ≈ 1.5*x - 0.5*x^3
//
// This uses basic HE operations (multiply, rescale, add) which are the
// public API. The internal evaluate_poly is tied to the bootstrapping
// context and doesn't work correctly for standalone polynomial evaluation.
//
// Depth: 3 levels (x^2: 1, x^3: 1, final multiply_plain + rescale: 1)
// Accuracy: good for |x| > 0.3, rough in [-0.3, 0.3] transition band
// =========================================================================
heongpu::Ciphertext<Scheme>
sign_approx_deg3(heongpu::Ciphertext<Scheme>& ct_x,
                 heongpu::HEArithmeticOperator<Scheme>& eval,
                 heongpu::HEContext<Scheme>& context,
                 heongpu::Relinkey<Scheme>& relin_key,
                 double scale,
                 const heongpu::ExecutionOptions& opts = heongpu::ExecutionOptions())
{
    std::cout << "  [sign] x^2 = x*x\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_x2(context);
    eval.multiply(ct_x, ct_x, ct_x2, opts);
    std::cout << "  [sign] relin x^2\n" << std::flush;
    eval.relinearize_inplace(ct_x2, relin_key, opts);
    std::cout << "  [sign] rescale x^2\n" << std::flush;
    eval.rescale_inplace(ct_x2, opts);

    std::cout << "  [sign] x^3 = x*x^2\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_x_drop(context);
    ct_x_drop = ct_x;
    eval.mod_drop_inplace(ct_x_drop, opts);
    heongpu::Ciphertext<Scheme> ct_x3(context);
    eval.multiply(ct_x_drop, ct_x2, ct_x3, opts);
    std::cout << "  [sign] relin x^3\n" << std::flush;
    eval.relinearize_inplace(ct_x3, relin_key, opts);
    std::cout << "  [sign] rescale x^3\n" << std::flush;
    eval.rescale_inplace(ct_x3, opts);

    std::cout << "  [sign] mod_drop x to level of x^3\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_x_l1(context);
    ct_x_l1 = ct_x;
    eval.mod_drop_inplace(ct_x_l1, opts);
    eval.mod_drop_inplace(ct_x_l1, opts);

    std::cout << "  [sign] 1.5*x\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_term1(context);
    eval.multiply_plain(ct_x_l1, 1.5, ct_term1, scale, opts);
    std::cout << "  [sign] rescale term1\n" << std::flush;
    eval.rescale_inplace(ct_term1, opts);

    std::cout << "  [sign] -0.5*x^3\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_term2(context);
    eval.multiply_plain(ct_x3, -0.5, ct_term2, scale, opts);
    std::cout << "  [sign] rescale term2\n" << std::flush;
    eval.rescale_inplace(ct_term2, opts);

    std::cout << "  [sign] add term1 + term2\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_result(context);
    eval.add(ct_term1, ct_term2, ct_result, opts);
    std::cout << "  [sign] done\n" << std::flush;

    return ct_result;
}

int main()
{
    cudaSetDevice(0);

    // =========================================================================
    // Setup: 3 usable levels for degree-3 sign approximation.
    // {40, 30, 30, 30} + {40} = 170 bits ≤ 218 (N=8192, 128-bit security).
    // =========================================================================
    heongpu::HEContext<Scheme> context = heongpu::GenHEContext<Scheme>();
    const size_t poly_modulus_degree = 8192;
    context->set_poly_modulus_degree(poly_modulus_degree);
    context->set_coeff_modulus_bit_sizes({40, 30, 30, 30}, {40});
    const double scale = pow(2.0, 30);
    context->generate();
    context->print_parameters();

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
    heongpu::HEArithmeticOperator<Scheme> evaluator(context, encoder);

    const int slot_count = static_cast<int>(poly_modulus_degree / 2);

    // =========================================================================
    // Part 1: Single-pair SIMD comparison
    //
    // One polynomial evaluation compares all n/2 = 4096 slot pairs at once.
    // Input: diff = a - b must lie in [-1, 1].
    // Here a, b in [-0.45, 0.45] so diff in [-0.9, 0.9].
    // =========================================================================
    std::vector<double> vec_a(slot_count), vec_b(slot_count);
    for (int i = 0; i < slot_count; i++)
    {
        vec_a[i] = (static_cast<double>((i * 7 + 3) % 10) - 4.5) * 0.1;
        vec_b[i] = (static_cast<double>((i * 3 + 1) % 10) - 4.5) * 0.1;
    }

    heongpu::Plaintext<Scheme> pt_a(context), pt_b(context);
    encoder.encode(pt_a, vec_a, scale);
    encoder.encode(pt_b, vec_b, scale);
    heongpu::Ciphertext<Scheme> ct_a(context), ct_b(context);
    encryptor.encrypt(ct_a, pt_a);
    encryptor.encrypt(ct_b, pt_b);

    // diff = a - b
    std::cout << "[DBG] computing diff = a - b\n" << std::flush;
    heongpu::Ciphertext<Scheme> ct_diff(context);
    evaluator.sub(ct_a, ct_b, ct_diff);
    std::cout << "[DBG] diff done. Starting sign_approx_deg3...\n" << std::flush;

    // Evaluate sign polynomial: all 4096 comparisons in one GPU call.
    GPUTimer timer;
    timer.start();
    heongpu::Ciphertext<Scheme> ct_sign =
        sign_approx_deg3(ct_diff, evaluator, context, relin_key, scale);
    float single_ms = timer.stop();
    std::cout << "[DBG] sign_approx_deg3 done.\n" << std::flush;

    // Decrypt and verify.
    heongpu::Plaintext<Scheme> pt_sign(context);
    decryptor.decrypt(pt_sign, ct_sign);
    std::vector<double> result_sign;
    encoder.decode(result_sign, pt_sign);

    std::cout << "=== Part 1: Single-pair SIMD comparison ===\n";
    std::cout << slot_count << " slots compared in " << single_ms << " ms\n";
    std::cout << "sign(x) approx 1.5*x - 0.5*x^3 (degree 3)\n";
    std::cout << "First 8 results:\n";
    for (int i = 0; i < 8; i++)
    {
        double diff_val = vec_a[i] - vec_b[i];
        double expected = 1.5 * diff_val - 0.5 * diff_val * diff_val * diff_val;
        int sign_expected = (diff_val > 0) ? 1 : (diff_val < 0 ? -1 : 0);
        int sign_got      = (result_sign[i] > 0) ? 1 : -1;
        bool correct      = (sign_got == sign_expected || std::abs(diff_val) < 0.15);
        std::cout << "  [" << i << "] diff=" << diff_val
                  << "  sign_approx=" << result_sign[i]
                  << "  (exact_poly=" << expected << ")"
                  << "  sign=" << sign_got
                  << (correct ? "" : "  MISMATCH") << "\n";
    }
    int correct_count = 0, total_counted = 0;
    for (int i = 0; i < slot_count; i++)
    {
        double diff_val = vec_a[i] - vec_b[i];
        if (std::abs(diff_val) > 0.15)
        {
            total_counted++;
            int expected = (diff_val > 0) ? 1 : -1;
            int got      = (result_sign[i] > 0) ? 1 : -1;
            if (got == expected) correct_count++;
        }
    }
    std::cout << "Accuracy (excluding |diff|<0.15): "
              << correct_count << " / " << total_counted << " correct\n\n";

    // =========================================================================
    // Part 2: Multi-stream parallelism - K independent comparisons
    //
    // K difference ciphertexts dispatched to K CUDA streams via OpenMP.
    // Each stream's polynomial evaluation runs concurrently on the GPU.
    // =========================================================================
    const int K = 4;

    std::vector<cudaStream_t> streams(K);
    for (int i = 0; i < K; i++)
        cudaStreamCreate(&streams[i]);

    auto make_diffs = [&]() {
        std::vector<heongpu::Ciphertext<Scheme>> diffs;
        diffs.reserve(K);
        for (int i = 0; i < K; i++)
        {
            std::vector<double> va(slot_count), vb(slot_count);
            for (int j = 0; j < slot_count; j++)
            {
                va[j] = (static_cast<double>(((j + i * 3) % 10)) - 4.5) * 0.09;
                vb[j] = (static_cast<double>(((j + i * 7) % 10)) - 4.5) * 0.09;
            }
            heongpu::Plaintext<Scheme> pa(context), pb(context);
            encoder.encode(pa, va, scale);
            encoder.encode(pb, vb, scale);
            heongpu::Ciphertext<Scheme> ca(context), cb(context);
            encryptor.encrypt(ca, pa);
            encryptor.encrypt(cb, pb);
            heongpu::Ciphertext<Scheme> d(context);
            evaluator.sub(ca, cb, d);
            diffs.push_back(std::move(d));
        }
        return diffs;
    };

    // -- Sequential baseline --
    auto diffs_seq = make_diffs();
    cudaDeviceSynchronize();
    GPUTimer seq_timer;
    seq_timer.start();
    std::vector<heongpu::Ciphertext<Scheme>> results_seq;
    results_seq.reserve(K);
    for (int i = 0; i < K; i++)
    {
        results_seq.push_back(
            sign_approx_deg3(diffs_seq[i], evaluator, context,
                             relin_key, scale));
    }
    std::cout << "[DBG] seq loop done, syncing...\n" << std::flush;
    cudaDeviceSynchronize();
    float seq_ms = seq_timer.stop();
    std::cout << "[DBG] seq_ms=" << seq_ms << "\n" << std::flush;

    // -- Multi-stream parallel --
    auto diffs_par = make_diffs();
    cudaDeviceSynchronize();
    GPUTimer par_timer;
    par_timer.start();
    std::vector<heongpu::Ciphertext<Scheme>> results_par(K, heongpu::Ciphertext<Scheme>(context));
#pragma omp parallel for num_threads(K)
    for (int i = 0; i < K; i++)
    {
        results_par[i] =
            sign_approx_deg3(diffs_par[i], evaluator, context,
                             relin_key, scale,
                             heongpu::ExecutionOptions().set_stream(streams[i]));
    }
    std::cout << "[DBG] parallel loop done, syncing...\n" << std::flush;
    cudaDeviceSynchronize();
    std::cout << "[DBG] synced, stopping timer...\n" << std::flush;
    float par_ms = par_timer.stop();
    std::cout << "[DBG] timer stopped\n" << std::flush;

    std::cout << "=== Part 2: Multi-stream K=" << K << " comparisons ===\n";
    std::cout << "  Sequential (default stream): " << seq_ms << " ms\n";
    std::cout << "  Parallel  (" << K << " streams):    " << par_ms << " ms\n";
    if (par_ms > 0)
        std::cout << "  Speedup: " << seq_ms / par_ms << "x\n\n";
    std::cout << std::flush;

    std::cout << "[DBG] Part 2 output done\n" << std::flush;

    // =========================================================================
    // Part 3: Matrix comparison pattern (extends example 16)
    // =========================================================================
    const int vec_len = static_cast<int>(std::sqrt(slot_count));
    std::cout << "=== Part 3: Matrix comparison pattern ===\n";
    std::cout << "For a " << vec_len << "x" << vec_len << " matrix:\n";
    std::cout << "  Naive approach: " << vec_len * vec_len
              << " scalar comparisons\n";
    std::cout << "  CKKS SIMD approach: 1 x sign_approx on (vR - vC)\n";
    std::cout << "  -> full " << vec_len << "x" << vec_len
              << " comparison matrix in a single ciphertext\n";
    std::cout << "  See replicateRow/replicateColumn in 16_ckks_rotation_parallel.cpp\n";
    std::cout << std::flush;

    std::cout << "[DBG] Part 3 output done, destroying streams...\n" << std::flush;
    for (auto& s : streams)
        cudaStreamDestroy(s);
    std::cout << "[DBG] streams destroyed, returning\n" << std::flush;

    return EXIT_SUCCESS;
}
