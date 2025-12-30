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
#include <unordered_map>

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

static std::vector<double> negacyclic_multiply(const std::vector<double>& a,
                                               const std::vector<double>& b)
{
    const int N = static_cast<int>(a.size());
    std::vector<double> out(static_cast<size_t>(N), 0.0);
    for (int i = 0; i < N; i++)
    {
        const double ai = a[i];
        if (ai == 0.0)
            continue;
        for (int j = 0; j < N; j++)
        {
            const double prod = ai * b[j];
            if (prod == 0.0)
                continue;
            const int s = i + j;
            if (s < N)
            {
                out[s] += prod;
            }
            else
            {
                out[s - N] -= prod;
            }
        }
    }
    return out;
}

static std::vector<double> pack_reference_strideB(const std::vector<std::vector<double>>& rb,
                                                  int B)
{
    const int N = static_cast<int>(rb[0].size());
    std::vector<double> out(static_cast<size_t>(N), 0.0);
    for (int b = 0; b < B; b++)
    {
        for (int i = 0; i < N - b; i++)
        {
            out[i + b] += rb[b][i];
        }
        for (int i = N - b; i < N; i++)
        {
            // wrap and negate due to X^N = -1
            out[i + b - N] -= rb[b][i];
        }
    }
    return out;
}

static std::vector<double>
project_strideB_reference(const std::vector<double>& poly, int B)
{
    const int N = static_cast<int>(poly.size());
    std::vector<double> out(static_cast<size_t>(N), 0.0);
    for (int i = 0; i < N; i++)
    {
        if ((i % B) == 0)
        {
            out[i] = static_cast<double>(B) * poly[i]; // projection scales by B
        }
    }
    return out;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    // Demo sweep.
    const std::vector<int> Bs = {4, 8, 16};
    const std::vector<int> ws = {8, 16};
    const std::vector<int> ks = {3, 5};

    const size_t N = 8192;
    const double scale = std::pow(2.0, 30);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    heongpu::HEContext<Scheme> context(
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);
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

    // Galois keys for projection subgroup elements.
    // g_t = 1 + t*(2N/B) mod 2N for each B used in the sweep.
    std::unordered_map<int, heongpu::Galoiskey<Scheme>> gks;
    for (int B : Bs)
    {
        if ((static_cast<int>(N) % B) != 0)
            continue;

        const int twoN = static_cast<int>(2 * N);
        const int step = twoN / B;
        std::vector<uint32_t> galois_elts;
        galois_elts.reserve(static_cast<size_t>(B - 1));
        for (int t = 1; t < B; t++)
        {
            galois_elts.push_back(static_cast<uint32_t>(1 + t * step));
        }
        heongpu::Galoiskey<Scheme> gk(context, galois_elts);
        keygen.generate_galois_key(gk, sk);
        gk.store_in_device();
        gks.emplace(B, std::move(gk));
    }

    const double abs_eps = 2e-3;
    const double rel_eps = 1e-6;

    for (int B : Bs)
    {
        for (int w : ws)
        {
            for (int k : ks)
            {
                if (k > w)
                    continue;
                const int d = w - k + 1;

                // Quick sanity: indices must fit N.
                const int max_needed_exp = B * ((d - 1) * w + (d - 1)) + (B - 1);
                if (max_needed_exp >= static_cast<int>(N))
                {
                    continue;
                }

                std::vector<double> inputs(static_cast<size_t>(B * w * w), 0.0);
                std::vector<double> kernels(static_cast<size_t>(B * B * k * k), 0.0);
                for (auto& x : inputs)
                    x = dist(rng);
                for (auto& x : kernels)
                    x = dist(rng);

                // Plain reference outputs R_b.
                std::vector<std::vector<double>> ref_R(static_cast<size_t>(B));
                for (int out = 0; out < B; out++)
                {
                    std::vector<double> acc(static_cast<size_t>(d * d), 0.0);
                    for (int in = 0; in < B; in++)
                    {
                        std::vector<double> Iin(inputs.begin() + (in * w * w),
                                                inputs.begin() + ((in + 1) * w * w));
                        std::vector<double> Kin(
                            kernels.begin() + ((out * B + in) * k * k),
                            kernels.begin() + ((out * B + in + 1) * k * k));
                        std::vector<double> tmp = valid_conv2d(Iin, Kin, w, k);
                        for (int idx = 0; idx < d * d; idx++)
                            acc[idx] += tmp[idx];
                    }
                    ref_R[out] = std::move(acc);
                }

                // Build sparse packed input polynomial and encrypt once.
                std::vector<double> I_sparse =
                    packer.build_input_sparse_coeffs(inputs, B, w, k);
                heongpu::Plaintext<Scheme> P_I(context);
                encoder.encode_coeffs(P_I, I_sparse, scale);
                heongpu::Ciphertext<Scheme> C_I(context);
                encryptor.encrypt(C_I, P_I);

                // For each out b: compute ct_b = C_I * Kb_sparse (plaintext).
                std::vector<heongpu::Ciphertext<Scheme>> ct_b(static_cast<size_t>(B));
                std::vector<std::vector<double>> rb_cpu(static_cast<size_t>(B));

                for (int out = 0; out < B; out++)
                {
                    std::vector<double> Kb_sparse =
                        packer.build_kernel_sparse_coeffs_for_out(kernels, out, B, w, k);

                    // CPU sim: r_b = I_sparse * Kb_sparse
                    rb_cpu[out] = negacyclic_multiply(I_sparse, Kb_sparse);

                    heongpu::Plaintext<Scheme> P_K(context);
                    encoder.encode_coeffs(P_K, Kb_sparse, scale);

                    ct_b[out] = C_I;
                    ops.multiply_plain_inplace(ct_b[out], P_K);
                    ops.rescale_inplace(ct_b[out]);
                }

                // PackCoeffs: first project stride-B coefficients (scaled by B),
                // then shift by b and sum.
                auto it = gks.find(B);
                if (it == gks.end())
                {
                    throw std::invalid_argument("Missing galois keys for B");
                }
                heongpu::Ciphertext<Scheme> ct_pack =
                    packer.pack_coeffs_strideB(ops, ct_b, B, it->second);

                heongpu::Plaintext<Scheme> P_out(context);
                decryptor.decrypt(P_out, ct_pack);
                std::vector<double> decoded;
                encoder.decode_coeffs(decoded, P_out);

                // CPU pack simulation: project stride-B (scaled by B), shift by b, sum.
                std::vector<std::vector<double>> rb_proj(static_cast<size_t>(B));
                for (int b = 0; b < B; b++)
                {
                    rb_proj[b] = project_strideB_reference(rb_cpu[b], B);
                }
                std::vector<double> pack_cpu = pack_reference_strideB(rb_proj, B);

                // Verify:
                // packed coeff at (B*(i*w+j) + out) equals R_out[i,j].
                double max_abs_err = 0.0;
                for (int out = 0; out < B; out++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            const int pos = B * (i * w + j) + out;
                            const double want =
                                static_cast<double>(B) * ref_R[out][i * d + j];
                            const double got = decoded[pos];
                            const double abs_err = std::abs(got - want);
                            max_abs_err = std::max(max_abs_err, abs_err);

                            const double tol =
                                abs_eps + rel_eps * std::max(1.0, std::abs(want));
                            if (abs_err > tol)
                            {
                                std::cout << "FAIL B=" << B << " w=" << w
                                          << " k=" << k << " out=" << out
                                          << " i=" << i << " j=" << j
                                          << " pos=" << pos << " got=" << got
                                          << " ref=" << want
                                          << " cpu_pack=" << pack_cpu[pos]
                                          << " abs_err=" << abs_err
                                          << " tol=" << tol << std::endl;
                                return EXIT_FAILURE;
                            }
                        }
                    }
                }

                // Also check decoded polynomial matches CPU packed poly on the probed indices.
                double max_abs_err_poly = 0.0;
                for (int out = 0; out < B; out++)
                {
                    for (int i = 0; i < d; i++)
                    {
                        for (int j = 0; j < d; j++)
                        {
                            const int pos = B * (i * w + j) + out;
                            max_abs_err_poly = std::max(
                                max_abs_err_poly, std::abs(decoded[pos] - pack_cpu[pos]));
                        }
                    }
                }

                std::cout << "OK B=" << B << " w=" << w << " k=" << k
                          << " max_abs_err=" << max_abs_err
                          << " max_abs_err_poly=" << max_abs_err_poly
                          << std::endl;
            }
        }
    }

    return EXIT_SUCCESS;
}
