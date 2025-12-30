// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_CONV_RELU_LAYER_H
#define HEONGPU_CKKS_CONV_RELU_LAYER_H

#include <heongpu/host/ckks/batchconv_pack.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/operator.cuh>

namespace heongpu
{
    struct ConvReluLayerStats
    {
        int level_in = -1;
        int level_after_conv = -1;
        int level_after_cts = -1;
        int level_after_act = -1;
        int level_after_stc = -1;

        double log2_scale_in = 0.0;
        double log2_scale_after_conv = 0.0;
        double log2_scale_after_cts = 0.0;
        double log2_scale_after_act = 0.0;
        double log2_scale_after_stc = 0.0;

        double ms_conv = 0.0;
        double ms_pack = 0.0;
        double ms_cts = 0.0;
        double ms_act = 0.0;
        double ms_stc = 0.0;
    };

    struct ConvReluLayerParams
    {
        int B = 0;
        int w = 0;
        int k = 0;

        // Encoding scale for coefficient encoding inputs/weights.
        double scale = 0.0;

        // Bootstrapping encoding matrix configuration for coeff<->slot.
        BootstrappingConfigV2 boot_cfg_v2;

        // Polynomial coefficients for sign approximation (monomial basis).
        // Example default is a low-degree placeholder; replace as needed.
        std::vector<double> sign_poly = {0.0, 1.0};

        ConvReluLayerParams(EncodingMatrixConfig stc, EvalModConfig eval_mod,
                            EncodingMatrixConfig cts)
            : boot_cfg_v2(stc, eval_mod, cts)
        {
        }
    };

    /**
     * @brief Executes a conv->CtoS->activation->StoC layer.
     *
     * Assumptions:
     * - ct_in encodes I_sparse(X) in coefficient encoding (as in BatchConv).
     * - weights is B*B*k*k, out-major: weights[(out*B+in)*k*k + u*k + v].
     * - Activation is ReLU(x) ≈ 0.5*(x + x*sign(x)), where sign(x) is a
     *   polynomial approximation evaluated in slot domain.
     */
    __host__ Ciphertext<Scheme::CKKS> eval_conv_relu_layer(
        Ciphertext<Scheme::CKKS>& ct_in, const std::vector<double>& weights,
        const ConvReluLayerParams& params, HEEncoder<Scheme::CKKS>& encoder,
        HEArithmeticOperator<Scheme::CKKS>& operators,
        HEBatchConvPack<Scheme::CKKS>& packer, Galoiskey<Scheme::CKKS>& gk_pack,
        Galoiskey<Scheme::CKKS>& gk_boot, Relinkey<Scheme::CKKS>& relin_key,
        const ExecutionOptions& options = ExecutionOptions(),
        ConvReluLayerStats* stats = nullptr);

} // namespace heongpu

#endif // HEONGPU_CKKS_CONV_RELU_LAYER_H
