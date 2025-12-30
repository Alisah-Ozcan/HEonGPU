// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/convolution.cuh>

namespace heongpu
{
    __host__ HEConvolution<Scheme::CKKS>::HEConvolution(
        HEContext<Scheme::CKKS>& context)
        : scheme_(scheme_type::ckks), n_(context.n), n_power_(context.n_power),
          Q_size_(context.Q_size), modulus_(context.modulus_),
          ntt_table_(context.ntt_table_), intt_table_(context.intt_table_),
          n_inverse_(context.n_inverse_)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument(
                "HEConvolution requires a generated context!");
        }
    }

    __host__ void HEConvolution<Scheme::CKKS>::to_ntt_domain_inplace(
        DeviceVector<Data64>& poly_coeff_rns, int poly_count, int rns_count,
        const ExecutionOptions& options)
    {
        if (poly_count <= 0 || rns_count <= 0)
        {
            throw std::invalid_argument("Invalid poly_count or rns_count!");
        }

        const size_t expected_size =
            static_cast<size_t>(poly_count) * static_cast<size_t>(rns_count) *
            static_cast<size_t>(n_);
        if (poly_coeff_rns.size() != expected_size)
        {
            throw std::invalid_argument(
                "Invalid size for to_ntt_domain_inplace!");
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        const int ntt_batch = poly_count * rns_count;
        gpuntt::GPU_NTT_Inplace(poly_coeff_rns.data(), ntt_table_->data(),
                                modulus_->data(), cfg_ntt, ntt_batch,
                                rns_count);
    }

    __host__ void HEConvolution<Scheme::CKKS>::to_coeff_domain_inplace(
        DeviceVector<Data64>& poly_ntt_rns, int poly_count, int rns_count,
        const ExecutionOptions& options)
    {
        if (poly_count <= 0 || rns_count <= 0)
        {
            throw std::invalid_argument("Invalid poly_count or rns_count!");
        }

        const size_t expected_size =
            static_cast<size_t>(poly_count) * static_cast<size_t>(rns_count) *
            static_cast<size_t>(n_);
        if (poly_ntt_rns.size() != expected_size)
        {
            throw std::invalid_argument(
                "Invalid size for to_coeff_domain_inplace!");
        }

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        const int ntt_batch = poly_count * rns_count;
        gpuntt::GPU_INTT_Inplace(poly_ntt_rns.data(), intt_table_->data(),
                                 modulus_->data(), cfg_intt, ntt_batch,
                                 rns_count);
    }

    __host__ void HEConvolution<Scheme::CKKS>::negacyclic_convolution_rns(
        const DeviceVector<Data64>& a_coeff_rns,
        const DeviceVector<Data64>& b_coeff_rns, DeviceVector<Data64>& out,
        int poly_count, const ExecutionOptions& options)
    {
        if (poly_count <= 0)
        {
            throw std::invalid_argument("poly_count must be positive!");
        }

        const int rns_count = Q_size_;
        const size_t expected_size =
            static_cast<size_t>(poly_count) * static_cast<size_t>(rns_count) *
            static_cast<size_t>(n_);

        if (a_coeff_rns.size() != expected_size ||
            b_coeff_rns.size() != expected_size)
        {
            throw std::invalid_argument(
                "Invalid input size for negacyclic_convolution_rns!");
        }

        if (out.size() != expected_size)
        {
            out = DeviceVector<Data64>(expected_size, options.stream_);
        }

        DeviceVector<Data64> a_ntt(a_coeff_rns, options.stream_);
        DeviceVector<Data64> b_ntt(b_coeff_rns, options.stream_);

        to_ntt_domain_inplace(a_ntt, poly_count, rns_count, options);
        to_ntt_domain_inplace(b_ntt, poly_count, rns_count, options);

        const dim3 blocks(static_cast<unsigned>((n_ + 255) >> 8),
                          static_cast<unsigned>(rns_count),
                          static_cast<unsigned>(poly_count));
        rns_pointwise_multiply_kernel<<<blocks, 256, 0, options.stream_>>>(
            a_ntt.data(), b_ntt.data(), out.data(), modulus_->data(), n_power_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        to_coeff_domain_inplace(out, poly_count, rns_count, options);
    }

    __host__ DeviceVector<Data64> HEConvolution<Scheme::CKKS>::negacyclic_convolution_rns(
        const DeviceVector<Data64>& a_coeff_rns,
        const DeviceVector<Data64>& b_coeff_rns, int poly_count,
        const ExecutionOptions& options)
    {
        DeviceVector<Data64> out;
        negacyclic_convolution_rns(a_coeff_rns, b_coeff_rns, out, poly_count,
                                   options);
        return out;
    }

} // namespace heongpu
