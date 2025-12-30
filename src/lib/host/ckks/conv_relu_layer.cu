// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/conv_relu_layer.cuh>

#include <chrono>
#include <cmath>

namespace heongpu
{
    __host__ Ciphertext<Scheme::CKKS> eval_conv_relu_layer(
        Ciphertext<Scheme::CKKS>& ct_in, const std::vector<double>& weights,
        const ConvReluLayerParams& params, HEEncoder<Scheme::CKKS>& encoder,
        HEArithmeticOperator<Scheme::CKKS>& operators,
        HEBatchConvPack<Scheme::CKKS>& packer, Galoiskey<Scheme::CKKS>& gk_pack,
        Galoiskey<Scheme::CKKS>& gk_boot, Relinkey<Scheme::CKKS>& relin_key,
        const ExecutionOptions& options, ConvReluLayerStats* stats)
    {
        const int B = params.B;
        const int w = params.w;
        const int k = params.k;

        if (B <= 0 || (B & (B - 1)) != 0)
        {
            throw std::invalid_argument("B must be a power-of-two positive!");
        }
        if (w <= 0 || k <= 0 || k > w)
        {
            throw std::invalid_argument("Invalid w/k!");
        }
        if (static_cast<int>(weights.size()) != (B * B * k * k))
        {
            throw std::invalid_argument("Invalid weights size!");
        }
        if (params.scale <= 0.0)
        {
            throw std::invalid_argument("Invalid scale!");
        }

        auto sync_and_ms = [&](std::chrono::high_resolution_clock::time_point t0)
        {
            HEONGPU_CUDA_CHECK(cudaStreamSynchronize(options.stream_));
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        if (stats != nullptr)
        {
            stats->level_in = ct_in.level();
            stats->log2_scale_in = std::log2(ct_in.scale());
        }

        // 1) BatchConv: for each out, multiply by plaintext Kb_sparse and rescale.
        auto t_conv0 = std::chrono::high_resolution_clock::now();
        std::vector<Ciphertext<Scheme::CKKS>> ct_b(static_cast<size_t>(B));
        for (int out = 0; out < B; out++)
        {
            std::vector<double> Kb_sparse =
                packer.build_kernel_sparse_coeffs_for_out(weights, out, B, w, k);

            Plaintext<Scheme::CKKS> P_K;
            encoder.encode_coeffs(P_K, Kb_sparse, params.scale, options);

            ct_b[out] = ct_in;
            operators.multiply_plain_inplace(ct_b[out], P_K, options);
            operators.rescale_inplace(ct_b[out], options);

            // Make sure projection sees a "clean" ciphertext.
            if (ct_b[out].relinearization_required())
            {
                throw std::logic_error("Unexpected non-linear part in ct_b");
            }
            if (ct_b[out].rescale_required())
            {
                throw std::logic_error("Unexpected rescale_required in ct_b");
            }
        }
        if (stats != nullptr)
        {
            stats->ms_conv = sync_and_ms(t_conv0);
        }

        // Pack stride-B coefficients (scaled by B due to projection).
        auto t_pack0 = std::chrono::high_resolution_clock::now();
        Ciphertext<Scheme::CKKS> ct_conv =
            packer.pack_coeffs_strideB(operators, ct_b, B, gk_pack, options);
        if (stats != nullptr)
        {
            stats->ms_pack = sync_and_ms(t_pack0);
            stats->level_after_conv = ct_conv.level();
            stats->log2_scale_after_conv = std::log2(ct_conv.scale());
        }

        // 2) Generate bootstrapping matrices if needed (once) and transform coeff->slot.
        // Caller is responsible for calling generate_bootstrapping_params_v2 on
        // the same operators instance before this function, or set params such
        // that it's generated here.
        // For simplicity, we generate here if not generated yet.
        try
        {
            (void)operators.bootstrapping_key_indexs();
        }
        catch (...)
        {
            operators.generate_bootstrapping_params_v2(params.scale,
                                                       params.boot_cfg_v2);
        }

        // Drop to cts_config_.level_start_ if needed. Note: coeff_to_slot_v2
        // consumes 4 rescale steps internally (piece=4), so you must start with
        // enough remaining moduli for your activation depth.
        const int cts_level_start = params.boot_cfg_v2.cts_config_.level_start_;
        if (ct_conv.level() < cts_level_start)
        {
            throw std::invalid_argument(
                "ct_conv level is lower than cts_level_start; increase modulus "
                "chain or choose a lower cts_level_start.");
        }
        while (ct_conv.level() > cts_level_start)
        {
            operators.mod_drop_inplace(ct_conv, options);
        }

        auto t_cts0 = std::chrono::high_resolution_clock::now();
        std::vector<Ciphertext<Scheme::CKKS>> slots =
            operators.coeff_to_slot_v2(ct_conv, gk_boot, options);
        if (slots.size() != 2)
        {
            throw std::logic_error("coeff_to_slot_v2 must return 2 ciphertexts");
        }
        if (stats != nullptr)
        {
            stats->ms_cts = sync_and_ms(t_cts0);
            stats->level_after_cts = slots[0].level();
            stats->log2_scale_after_cts = std::log2(slots[0].scale());
        }

        // 3) Activation in slot domain: ReLU(x) ≈ 0.5*(x + x*sign(x)).
        auto relu_inplace = [&](Ciphertext<Scheme::CKKS>& x)
        {
            Ciphertext<Scheme::CKKS> sign = operators.evaluate_poly_monomial(
                x, x.scale(), params.sign_poly, relin_key, options);

            // abs ≈ x * sign
            // Align levels before multiplication (evaluate_poly may drop levels).
            while (x.level() > sign.level())
            {
                operators.mod_drop_inplace(x, options);
            }

            Ciphertext<Scheme::CKKS> abs_ct = x;
            operators.multiply_inplace(abs_ct, sign, options);
            operators.relinearize_inplace(abs_ct, relin_key, options);
            operators.rescale_inplace(abs_ct, options);

            // Align levels for add.
            while (x.level() > abs_ct.level())
            {
                operators.mod_drop_inplace(x, options);
            }
            while (abs_ct.level() > x.level())
            {
                operators.mod_drop_inplace(abs_ct, options);
            }

            operators.add_inplace(x, abs_ct, options);

            // Multiply by 0.5:
            // Use constant-plaintext multiplication with a large encoding scale
            // so that 0.5 is not rounded away, then rescale back.
            Ciphertext<Scheme::CKKS> tmp;
            operators.multiply_plain(x, 0.5, tmp, x.scale(), options);
            operators.rescale_inplace(tmp, options);
            x = std::move(tmp);
        };

        auto t_act0 = std::chrono::high_resolution_clock::now();
        relu_inplace(slots[0]);
        relu_inplace(slots[1]);
        if (stats != nullptr)
        {
            stats->ms_act = sync_and_ms(t_act0);
            stats->level_after_act = slots[0].level();
            stats->log2_scale_after_act = std::log2(slots[0].scale());
        }

        // Drop to stc_config_.level_start_ if needed.
        const int stc_level_start = params.boot_cfg_v2.stc_config_.level_start_;
        if (slots[0].level() < stc_level_start || slots[1].level() < stc_level_start)
        {
            throw std::invalid_argument(
                "slot ciphertext level is lower than stc_level_start; increase "
                "modulus chain or choose a lower stc_level_start.");
        }
        while (slots[0].level() > stc_level_start)
        {
            operators.mod_drop_inplace(slots[0], options);
        }
        while (slots[1].level() > stc_level_start)
        {
            operators.mod_drop_inplace(slots[1], options);
        }

        // 4) slot->coeff
        auto t_stc0 = std::chrono::high_resolution_clock::now();
        Ciphertext<Scheme::CKKS> ct_out =
            operators.slot_to_coeff_v2(slots[0], slots[1], gk_boot, options);
        if (stats != nullptr)
        {
            stats->ms_stc = sync_and_ms(t_stc0);
            stats->level_after_stc = ct_out.level();
            stats->log2_scale_after_stc = std::log2(ct_out.scale());
        }

        return ct_out;
    }

} // namespace heongpu
