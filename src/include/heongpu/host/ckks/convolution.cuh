// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_CONVOLUTION_H
#define HEONGPU_CKKS_CONVOLUTION_H

#include "gpuntt/ntt_merge/ntt.cuh"
#include <heongpu/host/ckks/context.cuh>
#include <heongpu/kernel/convolution.cuh>
#include <heongpu/util/devicevector.cuh>
#include <heongpu/util/storagemanager.cuh>
#include <heongpu/util/util.cuh>

namespace heongpu
{
    /**
     * @brief HEConvolution provides coefficient-domain polynomial convolution
     * (negacyclic, modulo x^N + 1) using GPU NTT.
     *
     * This is a low-level primitive: inputs/outputs are RNS polynomials in the
     * coefficient domain, laid out as `[poly][rns][n]`.
     */
    template <> class HEConvolution<Scheme::CKKS>
    {
      public:
        __host__ explicit HEConvolution(HEContext<Scheme::CKKS>& context);

        __host__ void to_ntt_domain_inplace(DeviceVector<Data64>& poly_coeff_rns,
                                            int poly_count, int rns_count,
                                            const ExecutionOptions& options =
                                                ExecutionOptions());

        __host__ void to_coeff_domain_inplace(DeviceVector<Data64>& poly_ntt_rns,
                                              int poly_count, int rns_count,
                                              const ExecutionOptions& options =
                                                  ExecutionOptions());

        /**
         * @brief Computes out = a (*) b, where (*) is negacyclic convolution
         * modulo x^N + 1 in each RNS modulus.
         *
         * @param a_coeff_rns Input polynomial(s) in coefficient domain.
         * @param b_coeff_rns Input polynomial(s) in coefficient domain.
         * @param out_coeff_rns Output polynomial(s) in coefficient domain.
         * @param poly_count Number of independent polynomials (batch size).
         */
        __host__ void negacyclic_convolution_rns(
            const DeviceVector<Data64>& a_coeff_rns,
            const DeviceVector<Data64>& b_coeff_rns,
            DeviceVector<Data64>& out_coeff_rns, int poly_count,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ DeviceVector<Data64> negacyclic_convolution_rns(
            const DeviceVector<Data64>& a_coeff_rns,
            const DeviceVector<Data64>& b_coeff_rns, int poly_count,
            const ExecutionOptions& options = ExecutionOptions());

      private:
        scheme_type scheme_;

        int n_;
        int n_power_;
        int Q_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
    };

} // namespace heongpu

#endif // HEONGPU_CKKS_CONVOLUTION_H
