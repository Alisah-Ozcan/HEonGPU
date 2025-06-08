// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_TFHE_KEYGENERATOR_H
#define HEONGPU_TFHE_KEYGENERATOR_H

#include "ntt.cuh"
#include "keygeneration.cuh"
#include "switchkey.cuh"
#include "tfhe/context.cuh"
#include "tfhe/secretkey.cuh"
#include "tfhe/evaluationkey.cuh"

namespace heongpu
{
    template <> class HEKeyGenerator<Scheme::TFHE>
    {
      public:
        /**
         * @brief Initializes a key generator for the TFHE scheme.
         *
         * Sets up internal state using the given encryption context.
         *
         * @param context TFHE encryption context.
         */
        __host__ HEKeyGenerator(HEContext<Scheme::TFHE>& context);

        /**
         * @brief Generates a TFHE secret key.
         *
         * Writes the result to the provided `sk` object.
         *
         * @param sk Output secret key.
         * @param options Optional CUDA execution settings.
         */
        __host__ void generate_secret_key(
            Secretkey<Scheme::TFHE>& sk,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Generates a TFHE bootstrapping key and key switching key under
         * the Bootstrappingkey.
         *
         * Uses the given secret key and writes the result to `bk`.
         *
         * @param bk Output bootstrapping key.
         * @param sk Input secret key.
         * @param options Optional CUDA execution settings.
         */
        __host__ void generate_bootstrapping_key(
            Bootstrappingkey<Scheme::TFHE>& bk, Secretkey<Scheme::TFHE>& sk,
            const ExecutionOptions& options = ExecutionOptions());

      private:
        int rng_seed_;
        int rng_offset_;

        const scheme_type scheme_ = scheme_type::tfhe;

        Modulus64 prime_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        Ninverse64 n_inverse_;

        int ks_base_bit_;
        int ks_length_;

        double ks_stdev_;
        double bk_stdev_;
        double max_stdev_;

        // LWE Context
        int n_;
        // alpha_min = ks_stdev_
        // alpha_max = max_stdev_

        // TLWE Context
        int N_;
        int k_;
        // alpha_min = bk_stdev_
        // alpha_max = max_stdev_
        // extracted_lwe_params -> LWE {n = N*k, alpha_min, alpha_max}

        // TGSW Context
        int bk_l_;
        int bk_bg_bit_;
        int bg_;
        int half_bg_;
        int mask_mod_;
        // tlwe_params = TLWE Context
        int kpl_;
        std::vector<int> h_;
        int offset_;
    };

} // namespace heongpu
#endif // HEONGPU_TFHE_KEYGENERATOR_H