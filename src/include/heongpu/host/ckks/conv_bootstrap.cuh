// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_CONV_BOOTSTRAP_H
#define HEONGPU_CKKS_CONV_BOOTSTRAP_H

#include <heongpu/host/ckks/batchconv_pack.cuh>
#include <heongpu/host/ckks/encoder.cuh>
#include <heongpu/host/ckks/operator.cuh>

#include <vector>

namespace heongpu
{
    struct ConvParams
    {
        int B = 0;
        int w = 0;
        int k = 0;
        double scale = 0.0;
    };

    struct BootstrappingContext
    {
        // Bootstrapping config (v2). Caller should set these and call
        // operators.generate_bootstrapping_params_v2(scale, config) once.
        BootstrappingConfigV2 config_v2;

        // Keys used for bootstrapping rotations and EvalMod.
        // (Galois key must include ops.bootstrapping_key_indexs().)
        Galoiskey<Scheme::CKKS>* galois_key = nullptr;
        Relinkey<Scheme::CKKS>* relin_key = nullptr;

        // Recorded after execution.
        int output_level = -1;

        BootstrappingContext(EncodingMatrixConfig stc, EvalModConfig eval_mod,
                             EncodingMatrixConfig cts)
            : config_v2(stc, eval_mod, cts)
        {
        }
    };

    struct ConvThenBootstrapStats
    {
        int level_in = -1;
        int level_after_conv = -1;          // after conv+pack (before drops)
        int level_before_bootstrap = -1;    // after dropping to max depth
        int level_after_bootstrap = -1;     // output coeff-domain ciphertext

        double ms_conv = 0.0;
        double ms_pack = 0.0;
        double ms_drop = 0.0;
        double ms_bootstrap = 0.0;
    };

    /**
     * @brief Conv (coeff-domain) then regular bootstrapping (refresh noise and
     * restore modulus chain), no activation.
     *
     * Input/Output domains:
     * - ct_in_coeff: coefficient-encoding ciphertext in NTT domain (CKKS),
     *   representing a polynomial in R = Z[X]/(X^N + 1).
     * - The conv stage computes valid outputs in coefficient positions (as in
     *   HEBatchConvPack mapping) by multiplying ct_in_coeff with plaintext
     *   kernel polynomials and packing.
     * - Bootstrapping is performed via HEArithmeticOperator::regular_bootstrapping_v2
     *   which internally performs: modulus raise -> CtoS -> EvalMod -> StoC.
     * - ct_out_coeff: coefficient-domain ciphertext (NTT domain) with refreshed
     *   noise and higher level than the max-depth input to bootstrapping.
     *
     * Requirements enforced:
     * - Level decreases after conv (one rescale).
     * - Bootstrapping input is dropped to maximum depth (only 1 modulus left).
     * - Output level after bootstrapping is greater than level_before_bootstrap,
     *   and is recorded in bctx.output_level.
     */
    __host__ Ciphertext<Scheme::CKKS> eval_conv_then_bootstrap(
        Ciphertext<Scheme::CKKS>& ct_in_coeff, const std::vector<double>& weights,
        const ConvParams& p, HEEncoder<Scheme::CKKS>& encoder,
        HEArithmeticOperator<Scheme::CKKS>& operators,
        HEBatchConvPack<Scheme::CKKS>& packer, Galoiskey<Scheme::CKKS>& gk_pack,
        BootstrappingContext& bctx, const ExecutionOptions& options = ExecutionOptions(),
        ConvThenBootstrapStats* stats = nullptr);

} // namespace heongpu

#endif // HEONGPU_CKKS_CONV_BOOTSTRAP_H

