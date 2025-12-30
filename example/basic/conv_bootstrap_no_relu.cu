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
#include <string>
#include <vector>

constexpr auto Scheme = heongpu::Scheme::CKKS;

static std::vector<double>
valid_conv2d(const std::vector<double>& input, const std::vector<double>& kernel,
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

static void print_levels(const char* tag, const heongpu::Ciphertext<Scheme>& ct)
{
    std::cout << tag << ": depth=" << ct.depth() << " level=" << ct.level()
              << " scale(log2)=" << std::log2(ct.scale()) << "\n";
}

int main(int argc, char** argv)
{
    // Default small demo parameters.
    int B = 4;
    int w = 8;
    int k = 3;
    int N = 8192;
    double scale = std::pow(2.0, 30);

    // Minimal argument parsing:
    // conv_bootstrap_no_relu [B] [w] [k] [N] [log2_scale]
    if (argc > 1)
        B = std::stoi(argv[1]);
    if (argc > 2)
        w = std::stoi(argv[2]);
    if (argc > 3)
        k = std::stoi(argv[3]);
    if (argc > 4)
        N = std::stoi(argv[4]);
    if (argc > 5)
        scale = std::pow(2.0, std::stod(argv[5]));

    if (k > w)
    {
        std::cerr << "k must be <= w\n";
        return 1;
    }
    const int d = w - k + 1;

    cudaSetDevice(0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> inputs(static_cast<size_t>(B * w * w), 0.0);
    std::vector<double> weights(static_cast<size_t>(B * B * k * k), 0.0);
    for (auto& x : inputs)
        x = dist(rng);
    for (auto& x : weights)
        x = dist(rng);

    // Plain reference R_b.
    std::vector<std::vector<double>> ref_R(static_cast<size_t>(B));
    for (int out = 0; out < B; out++)
    {
        std::vector<double> acc(static_cast<size_t>(d * d), 0.0);
        for (int in = 0; in < B; in++)
        {
            std::vector<double> Iin(inputs.begin() + (in * w * w),
                                    inputs.begin() + ((in + 1) * w * w));
            std::vector<double> Kin(
                weights.begin() + ((out * B + in) * k * k),
                weights.begin() + ((out * B + in + 1) * k * k));
            std::vector<double> tmp = valid_conv2d(Iin, Kin, w, k);
            for (int idx = 0; idx < d * d; idx++)
                acc[idx] += tmp[idx];
        }
        ref_R[out] = std::move(acc);
    }

    // Context: give enough primes for conv + drops + bootstrap.
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(static_cast<size_t>(N));
    context.set_coeff_modulus_bit_sizes(
        {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30}, {60});
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

    // Pack galois keys (projection).
    const int twoN = 2 * N;
    const int step = twoN / B;
    std::vector<uint32_t> pack_galois;
    for (int t = 1; t < B; t++)
        pack_galois.push_back(static_cast<uint32_t>(1 + t * step));
    heongpu::Galoiskey<Scheme> gk_pack(context, pack_galois);
    keygen.generate_galois_key(gk_pack, sk);
    gk_pack.store_in_device();

    // Bootstrapping params (v2) + keys.
    const int Q_size = context.get_ciphertext_modulus_count();
    heongpu::EncodingMatrixConfig cts_cfg(
        heongpu::LinearTransformType::COEFFS_TO_SLOTS, Q_size - 2, 2.0, 4);
    heongpu::EncodingMatrixConfig stc_cfg(
        heongpu::LinearTransformType::SLOTS_TO_COEFFS, 4, 2.0, 3);
    heongpu::EvalModConfig eval_mod_cfg(/*level_start=*/0);
    heongpu::BootstrappingContext bctx(stc_cfg, eval_mod_cfg, cts_cfg);
    ops.generate_bootstrapping_params_v2(scale, bctx.config_v2);

    std::vector<int> boot_shifts = ops.bootstrapping_key_indexs();
    heongpu::Galoiskey<Scheme> gk_boot(context, boot_shifts);
    keygen.generate_galois_key(gk_boot, sk);
    gk_boot.store_in_device();
    bctx.galois_key = &gk_boot;
    bctx.relin_key = &rlk;

    // Encrypt packed input.
    std::vector<double> I_sparse = packer.build_input_sparse_coeffs(inputs, B, w, k);
    heongpu::Plaintext<Scheme> P_in(context);
    encoder.encode_coeffs(P_in, I_sparse, scale);
    heongpu::Ciphertext<Scheme> ct_in(context);
    encryptor.encrypt(ct_in, P_in);
    print_levels("ct_in", ct_in);

    heongpu::ConvParams cp;
    cp.B = B;
    cp.w = w;
    cp.k = k;
    cp.scale = scale;

    heongpu::ConvThenBootstrapStats st;
    auto t0 = std::chrono::high_resolution_clock::now();
    heongpu::Ciphertext<Scheme> ct_out = heongpu::eval_conv_then_bootstrap(
        ct_in, weights, cp, encoder, ops, packer, gk_pack, bctx,
        heongpu::ExecutionOptions(), &st);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Synchronize (avoid depending on exception macros here).
    cudaError_t err_sync = cudaDeviceSynchronize();
    if (err_sync != cudaSuccess)
    {
        std::cerr << "CUDA sync failed: " << cudaGetErrorString(err_sync) << "\n";
        return 1;
    }

    print_levels("ct_out", ct_out);
    std::cout << "Levels: L_in=" << st.level_in
              << " L_after_conv=" << st.level_after_conv
              << " L_before_boot=" << st.level_before_bootstrap
              << " L_after_boot=" << st.level_after_bootstrap
              << " bctx.output_level=" << bctx.output_level << "\n";

    std::cout << "Time(ms): total="
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " conv=" << st.ms_conv << " pack=" << st.ms_pack
              << " drop=" << st.ms_drop << " boot=" << st.ms_bootstrap << "\n";

    // Decrypt + decode coeffs.
    heongpu::Plaintext<Scheme> P_out(context);
    decryptor.decrypt(P_out, ct_out);
    std::vector<double> decoded;
    encoder.decode_coeffs(decoded, P_out);

    // Check VALID outputs at packed positions:
    // coeff index pos = B*(i*w + j) + out  stores (projection-scaled) value.
    const double abs_eps = 2e-2;
    const double rel_eps = 1e-5;
    double max_abs_err = 0.0;

    for (int out = 0; out < B; out++)
    {
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                const int pos = B * (i * w + j) + out;
                const double want = static_cast<double>(B) * ref_R[out][i * d + j];
                const double got = decoded[pos];
                const double abs_err = std::abs(got - want);
                max_abs_err = std::max(max_abs_err, abs_err);

                const double tol = abs_eps + rel_eps * std::max(1.0, std::abs(want));
                if (abs_err > tol)
                {
                    std::cout << "FAIL out=" << out << " i=" << i << " j=" << j
                              << " pos=" << pos << " got=" << got
                              << " ref=" << want << " abs_err=" << abs_err
                              << " tol=" << tol << "\n";
                    return 1;
                }
            }
        }
    }

    std::cout << "PASS max_abs_err=" << max_abs_err << "\n";
    return 0;
}

