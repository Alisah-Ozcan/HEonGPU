// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/conv_bootstrap.cuh>

#include <chrono>
#include <cmath>

namespace heongpu
{
    __host__ Ciphertext<Scheme::CKKS> eval_conv_then_bootstrap(
        Ciphertext<Scheme::CKKS>& ct_in_coeff, const std::vector<double>& weights,
        const ConvParams& p, HEEncoder<Scheme::CKKS>& encoder,
        HEArithmeticOperator<Scheme::CKKS>& operators,
        HEBatchConvPack<Scheme::CKKS>& packer, Galoiskey<Scheme::CKKS>& gk_pack,
        BootstrappingContext& bctx, const ExecutionOptions& options,
        ConvThenBootstrapStats* stats)
    {
        const int B = p.B;
        const int w = p.w;
        const int k = p.k;

        if (B <= 0 || (B & (B - 1)) != 0)
        {
            throw std::invalid_argument("B must be a power-of-two positive!");
        }
        if (w <= 0 || k <= 0 || k > w)
        {
            throw std::invalid_argument("Invalid w/k!");
        }
        if (p.scale <= 0.0)
        {
            throw std::invalid_argument("Invalid scale!");
        }
        if (static_cast<int>(weights.size()) != (B * B * k * k))
        {
            throw std::invalid_argument("Invalid weights size!");
        }
        if (bctx.galois_key == nullptr || bctx.relin_key == nullptr)
        {
            throw std::invalid_argument("BootstrappingContext keys are null");
        }

        auto sync_and_ms = [&](std::chrono::high_resolution_clock::time_point t0)
        {
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        if (stats != nullptr)
        {
            stats->level_in = ct_in_coeff.level();
        }

        // Step 1: Coeff-domain convolution (plaintext kernels), then pack.
        auto t_conv0 = std::chrono::high_resolution_clock::now();
        std::vector<Ciphertext<Scheme::CKKS>> ct_b(static_cast<size_t>(B));
        for (int out = 0; out < B; out++)
        {
            std::vector<double> Kb_sparse =
                packer.build_kernel_sparse_coeffs_for_out(weights, out, B, w, k);
            Plaintext<Scheme::CKKS> Pk;
            encoder.encode_coeffs(Pk, Kb_sparse, p.scale, options);

            ct_b[out] = ct_in_coeff;
            operators.multiply_plain_inplace(ct_b[out], Pk, options);
            operators.rescale_inplace(ct_b[out], options);
        }
        if (stats != nullptr)
        {
            stats->ms_conv = sync_and_ms(t_conv0);
        }

        auto t_pack0 = std::chrono::high_resolution_clock::now();
        Ciphertext<Scheme::CKKS> ct_conv_coeff =
            packer.pack_coeffs_strideB(operators, ct_b, B, gk_pack, options);
        if (stats != nullptr)
        {
            stats->ms_pack = sync_and_ms(t_pack0);
            stats->level_after_conv = ct_conv_coeff.level();
        }

        if (stats != nullptr && stats->level_in != -1)
        {
            if (!(stats->level_after_conv < stats->level_in))
            {
                throw std::logic_error(
                    "Expected level to decrease after convolution");
            }
        }

        // Step 2: Prepare for bootstrapping.
        // HEonGPU regular bootstrapping (v2) requires input at maximum depth:
        // current_decomp_count == 1  <=>  Q_size - depth == 1.
        auto t_drop0 = std::chrono::high_resolution_clock::now();
        const int Q_size = ct_conv_coeff.coeff_modulus_count();
        while ((Q_size - ct_conv_coeff.depth()) > 1)
        {
            operators.mod_drop_inplace(ct_conv_coeff, options);
        }
        if (stats != nullptr)
        {
            stats->ms_drop = sync_and_ms(t_drop0);
            stats->level_before_bootstrap = ct_conv_coeff.level();
        }

        // Step 2A-2D (logical decomposition):
        // A) CtoS: inside regular_bootstrapping_v2 -> coeff_to_slot_v2
        // B) (optional) slot mask: skipped; we preserve a well-defined sparse
        //    layout and bootstrap refreshes the whole ciphertext.
        // C) Bootstrapping core (EvalMod): inside regular_bootstrapping_v2
        // D) StoC: inside regular_bootstrapping_v2 -> slot_to_coeff_v2
        auto t_boot0 = std::chrono::high_resolution_clock::now();
        Ciphertext<Scheme::CKKS> ct_out_coeff =
            operators.regular_bootstrapping_v2(ct_conv_coeff, *bctx.galois_key,
                                               *bctx.relin_key, nullptr, nullptr,
                                               options);
        if (stats != nullptr)
        {
            stats->ms_bootstrap = sync_and_ms(t_boot0);
            stats->level_after_bootstrap = ct_out_coeff.level();
        }

        bctx.output_level = ct_out_coeff.level();

        if (stats != nullptr)
        {
            if (!(stats->level_after_bootstrap > stats->level_before_bootstrap))
            {
                throw std::logic_error(
                    "Expected level to increase after bootstrapping");
            }
        }

        return ct_out_coeff;
    }

} // namespace heongpu
