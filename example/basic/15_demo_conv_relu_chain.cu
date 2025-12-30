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
#include <unordered_map>
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

static inline double sign_poly_eval(double x)
{
    // Odd polynomial sign approximation on [-1,1] (degree 7):
    // p(x) = (35x - 35x^3 + 21x^5 - 5x^7) / 16
    // Coeffs are mirrored in ConvReluLayerParams::sign_poly below.
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double x6 = x4 * x2;
    return (35.0 * x - 35.0 * x * x2 + 21.0 * x * x4 - 5.0 * x * x6) / 16.0;
}

static inline double relu_approx(double x)
{
    // ReLU(x) ≈ 0.5*(x + x*sign(x))
    const double s = sign_poly_eval(x);
    return 0.5 * (x + x * s);
}

static std::vector<double> apply_relu_approx(const std::vector<double>& v)
{
    std::vector<double> out = v;
    for (auto& x : out)
        x = relu_approx(x);
    return out;
}

static void print_state(const char* tag, const heongpu::Ciphertext<Scheme>& ct)
{
    std::cout << tag << ": level=" << ct.level() << " depth=" << ct.depth()
              << " scale(log2)=" << std::log2(ct.scale()) << std::endl;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    const int B = 4;
    const int w = 8;
    const int k = 3;
    const int d = w - k + 1;

    const size_t N = 8192;
    const double scale = std::pow(2.0, 30);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    std::vector<double> inputs(static_cast<size_t>(B * w * w), 0.0);
    std::vector<double> weights(static_cast<size_t>(B * B * k * k), 0.0);
    for (auto& x : inputs)
        x = dist(rng);
    for (auto& x : weights)
        x = dist(rng);

    // Build plaintext conv reference R_b.
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

    // HE setup.
    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I,
        heongpu::sec_level_type::none);
    context.set_poly_modulus_degree(N);
    // Give enough levels for: coeff->slot (4 rescales) + sign-poly eval +
    // a few multiplies/rescales + slot->coeff (3 rescales).
    context.set_coeff_modulus_bit_sizes(
        {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30},
        {60});
    context.generate();
    const int Q_size = context.get_ciphertext_modulus_count();

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

    // Galois key for PackCoeffs projection (stride-B).
    const int twoN = static_cast<int>(2 * N);
    const int step = twoN / B;
    std::vector<uint32_t> pack_galois;
    for (int t = 1; t < B; t++)
        pack_galois.push_back(static_cast<uint32_t>(1 + t * step));
    heongpu::Galoiskey<Scheme> gk_pack(context, pack_galois);
    keygen.generate_galois_key(gk_pack, sk);
    gk_pack.store_in_device();

    // Bootstrapping matrix params (v2) for coeff<->slot transforms.
    // Choose start levels to keep enough room for activation depth.
    const int cts_level_start =
        Q_size - 2; // matches ct_conv right after conv+pack
    const int stc_level_start = 4;

    heongpu::EncodingMatrixConfig cts_cfg(
        heongpu::LinearTransformType::COEFFS_TO_SLOTS, cts_level_start, 2.0, 4);
    heongpu::EncodingMatrixConfig stc_cfg(
        heongpu::LinearTransformType::SLOTS_TO_COEFFS, stc_level_start, 2.0, 3);
    heongpu::EvalModConfig eval_mod_cfg(/*level_start=*/0);
    heongpu::ConvReluLayerParams layer_params(stc_cfg, eval_mod_cfg, cts_cfg);
    layer_params.B = B;
    layer_params.w = w;
    layer_params.k = k;
    layer_params.scale = scale;
    layer_params.sign_poly = {0.0, 35.0 / 16.0, 0.0, -35.0 / 16.0,
                              0.0, 21.0 / 16.0, 0.0, -5.0 / 16.0};

    ops.generate_bootstrapping_params_v2(scale, layer_params.boot_cfg_v2);

    // Galois keys required by coeff_to_slot/slot_to_coeff.
    std::vector<int> boot_shifts = ops.bootstrapping_key_indexs();
    heongpu::Galoiskey<Scheme> gk_boot(context, boot_shifts);
    keygen.generate_galois_key(gk_boot, sk);
    gk_boot.store_in_device();

    // Input: pack B inputs into coeff-domain I_sparse and encrypt.
    std::vector<double> I_sparse = packer.build_input_sparse_coeffs(inputs, B, w, k);
    heongpu::Plaintext<Scheme> P_in(context);
    encoder.encode_coeffs(P_in, I_sparse, scale);
    heongpu::Ciphertext<Scheme> ct_in(context);
    encryptor.encrypt(ct_in, P_in);

    // ---- Run layer API ----
    auto t0 = std::chrono::high_resolution_clock::now();
    heongpu::ConvReluLayerStats stats;
    heongpu::Ciphertext<Scheme> ct_out = heongpu::eval_conv_relu_layer(
        ct_in, weights, layer_params, encoder, ops, packer, gk_pack, gk_boot, rlk,
        heongpu::ExecutionOptions(), &stats);
    auto t1 = std::chrono::high_resolution_clock::now();

    print_state("ct_out", ct_out);
    std::cout << "Layer time(ms)="
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << std::endl;
    std::cout << "Stages(ms): conv=" << stats.ms_conv << " pack=" << stats.ms_pack
              << " cts=" << stats.ms_cts << " act=" << stats.ms_act
              << " stc=" << stats.ms_stc << std::endl;
    std::cout << "Scales(log2): in=" << stats.log2_scale_in
              << " conv=" << stats.log2_scale_after_conv
              << " cts=" << stats.log2_scale_after_cts
              << " act=" << stats.log2_scale_after_act
              << " stc=" << stats.log2_scale_after_stc << std::endl;

    // ---- Reference: use actual CtoS mapping, apply activation in clear, then StoC ----
    // Recompute conv only to get ct_conv, then CtoS.
    // (Reuse the same pieces to make a reference without relying on coeff<->slot semantics.)
    std::vector<heongpu::Ciphertext<Scheme>> ct_b(static_cast<size_t>(B));
    for (int out = 0; out < B; out++)
    {
        std::vector<double> Kb_sparse =
            packer.build_kernel_sparse_coeffs_for_out(weights, out, B, w, k);
        heongpu::Plaintext<Scheme> Pk(context);
        encoder.encode_coeffs(Pk, Kb_sparse, scale);
        ct_b[out] = ct_in;
        ops.multiply_plain_inplace(ct_b[out], Pk);
        ops.rescale_inplace(ct_b[out]);
    }
    heongpu::Ciphertext<Scheme> ct_conv =
        packer.pack_coeffs_strideB(ops, ct_b, B, gk_pack);
    while (ct_conv.level() > cts_level_start)
        ops.mod_drop_inplace(ct_conv);
    auto slots = ops.coeff_to_slot_v2(ct_conv, gk_boot);

    // Decrypt slots and apply activation in clear.
    heongpu::Plaintext<Scheme> P_s0(context), P_s1(context);
    decryptor.decrypt(P_s0, slots[0]);
    decryptor.decrypt(P_s1, slots[1]);
    std::vector<double> s0, s1;
    encoder.decode(s0, P_s0);
    encoder.decode(s1, P_s1);
    s0 = apply_relu_approx(s0);
    s1 = apply_relu_approx(s1);

    heongpu::Plaintext<Scheme> P_a0(context), P_a1(context);
    encoder.encode(P_a0, s0, slots[0].scale());
    encoder.encode(P_a1, s1, slots[1].scale());
    heongpu::Ciphertext<Scheme> ct_a0(context), ct_a1(context);
    encryptor.encrypt(ct_a0, P_a0);
    encryptor.encrypt(ct_a1, P_a1);
    while (ct_a0.level() > stc_level_start)
        ops.mod_drop_inplace(ct_a0);
    while (ct_a1.level() > stc_level_start)
        ops.mod_drop_inplace(ct_a1);
    heongpu::Ciphertext<Scheme> ct_ref =
        ops.slot_to_coeff_v2(ct_a0, ct_a1, gk_boot);

    heongpu::Plaintext<Scheme> P_out(context), P_ref(context);
    decryptor.decrypt(P_out, ct_out);
    decryptor.decrypt(P_ref, ct_ref);
    std::vector<double> out_coeffs, ref_coeffs;
    encoder.decode_coeffs(out_coeffs, P_out);
    encoder.decode_coeffs(ref_coeffs, P_ref);

    // Compare at packed valid positions: pos = B*(i*w+j) + out.
    const double abs_eps = 1e-2;
    const double rel_eps = 1e-5;
    double max_abs_err = 0.0;

    for (int out = 0; out < B; out++)
    {
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                const int pos = B * (i * w + j) + out;
                const double want = ref_coeffs[pos];
                const double got = out_coeffs[pos];
                const double abs_err = std::abs(got - want);
                max_abs_err = std::max(max_abs_err, abs_err);
                const double tol = abs_eps + rel_eps * std::max(1.0, std::abs(want));
                if (abs_err > tol)
                {
                    std::cout << "FAIL pos=" << pos << " got=" << got
                              << " ref=" << want << " abs_err=" << abs_err
                              << " tol=" << tol << std::endl;
                    return EXIT_FAILURE;
                }
            }
        }
    }

    std::cout << "PASS: max_abs_err=" << max_abs_err << std::endl;
    return EXIT_SUCCESS;
}
