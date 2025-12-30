// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/batchconv_pack.cuh>

namespace heongpu
{
    __host__ HEBatchConvPack<Scheme::CKKS>::HEBatchConvPack(
        HEContext<Scheme::CKKS>& context)
        : scheme_(scheme_type::ckks), n_(context.n), n_power_(context.n_power),
          Q_size_(context.Q_size), modulus_(context.modulus_),
          ntt_table_(context.ntt_table_), intt_table_(context.intt_table_),
          n_inverse_(context.n_inverse_)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument(
                "HEBatchConvPack requires a generated context!");
        }
    }

    __host__ void HEBatchConvPack<Scheme::CKKS>::add_coeff_with_neg_exp(
        std::vector<double>& poly, int exp, double value) const
    {
        const int N = static_cast<int>(poly.size());
        if (exp >= 0)
        {
            if (exp >= N)
            {
                throw std::invalid_argument("Exponent out of bounds");
            }
            poly[exp] += value;
            return;
        }

        const int idx = N + exp;
        if (idx < 0 || idx >= N)
        {
            throw std::invalid_argument("Negative exponent out of bounds");
        }
        poly[idx] -= value;
    }

    __host__ std::vector<double>
    HEBatchConvPack<Scheme::CKKS>::build_input_sparse_coeffs(
        const std::vector<double>& inputs, int B, int w, int k) const
    {
        if (B <= 0 || (B & (B - 1)) != 0)
        {
            throw std::invalid_argument("B must be a power-of-two positive!");
        }
        if (w <= 0 || k <= 0 || k > w)
        {
            throw std::invalid_argument("Invalid w/k!");
        }
        if (static_cast<int>(inputs.size()) != (B * w * w))
        {
            throw std::invalid_argument("Invalid inputs size!");
        }

        std::vector<double> poly(static_cast<size_t>(n_), 0.0);

        // I_sparse(X) = Σ_in I_in_poly(X^B) * X^{in}
        // I_in_poly(X) uses Proposition 1:
        // I(X)=Σ I[i,j] * X^{(i-k)*w + j}
        for (int in = 0; in < B; in++)
        {
            for (int i = 0; i < w; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    const int base_exp = (i - k) * w + j;
                    const int exp = base_exp * B + in;
                    add_coeff_with_neg_exp(poly, exp,
                                           inputs[(in * w + i) * w + j]);
                }
            }
        }

        return poly;
    }

    __host__ std::vector<double>
    HEBatchConvPack<Scheme::CKKS>::build_kernel_sparse_coeffs_for_out(
        const std::vector<double>& kernels, int out, int B, int w, int k) const
    {
        if (B <= 0 || (B & (B - 1)) != 0)
        {
            throw std::invalid_argument("B must be a power-of-two positive!");
        }
        if (w <= 0 || k <= 0 || k > w)
        {
            throw std::invalid_argument("Invalid w/k!");
        }
        if (out < 0 || out >= B)
        {
            throw std::invalid_argument("Invalid out index!");
        }
        if (static_cast<int>(kernels.size()) != (B * B * k * k))
        {
            throw std::invalid_argument("Invalid kernels size!");
        }

        std::vector<double> poly(static_cast<size_t>(n_), 0.0);

        // Kb_sparse(X) = Σ_in K_{b,in}_poly(X^B) * X^{-in}
        // K_poly(X)=Σ K[i,j] * X^{w*k - (i*w + j)}
        for (int in = 0; in < B; in++)
        {
            const int base_kernel =
                ((out * B + in) * k * k); // out-major
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < k; j++)
                {
                    const int base_exp = w * k - (i * w + j);
                    const int exp = base_exp * B - in;
                    add_coeff_with_neg_exp(poly, exp,
                                           kernels[base_kernel + i * k + j]);
                }
            }
        }

        return poly;
    }

    __host__ void HEBatchConvPack<Scheme::CKKS>::monomial_shift_inplace(
        Ciphertext<Scheme::CKKS>& ct, int shift,
        const ExecutionOptions& options)
    {
        // This is a depth-free linear transform: INTT -> negacyclic shift -> NTT.
        if (ct.rescale_required() || ct.relinearization_required())
        {
            throw std::invalid_argument(
                "Ciphertext can not be shifted (rescale/relin required)!");
        }

        const int current_decomp_count = Q_size_ - ct.depth();
        const int cipher_size = ct.size();
        const int n = n_;

        int shift_norm = shift % n;
        if (shift_norm >= n / 2)
        {
            // keep within [-n/2, n/2] to minimize range, though kernel supports
            // full wrap.
            shift_norm -= n;
        }
        if (shift_norm <= -n / 2)
        {
            shift_norm += n;
        }
        if (shift_norm == 0)
        {
            return;
        }

        DeviceVector<Data64> temp_ntt((cipher_size * n * current_decomp_count),
                                      options.stream_);
        DeviceVector<Data64> temp_coeff(
            (cipher_size * n * current_decomp_count), options.stream_);
        DeviceVector<Data64> temp_coeff_shifted(
            (cipher_size * n * current_decomp_count), options.stream_);

        cudaMemcpyAsync(temp_ntt.data(), ct.data(),
                        temp_ntt.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice, options.stream_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_intt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::INVERSE,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .mod_inverse = n_inverse_->data(),
            .stream = options.stream_};

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = options.stream_};

        const dim3 blocks_shift(static_cast<unsigned>((n + 255) >> 8),
                                static_cast<unsigned>(current_decomp_count), 1);

        for (int c = 0; c < cipher_size; c++)
        {
            Data64* ntt_in = temp_ntt.data() +
                             static_cast<size_t>(c) * current_decomp_count * n;
            Data64* coeff_out =
                temp_coeff.data() +
                static_cast<size_t>(c) * current_decomp_count * n;
            Data64* coeff_shifted =
                temp_coeff_shifted.data() +
                static_cast<size_t>(c) * current_decomp_count * n;

            gpuntt::GPU_INTT(ntt_in, coeff_out, intt_table_->data(),
                             modulus_->data(), cfg_intt, current_decomp_count,
                             current_decomp_count);

            negacyclic_shift_rns_kernel<<<blocks_shift, 256, 0,
                                          options.stream_>>>(
                coeff_out, coeff_shifted, modulus_->data(), shift_norm,
                n_power_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            gpuntt::GPU_NTT(coeff_shifted, ntt_in, ntt_table_->data(),
                            modulus_->data(), cfg_ntt, current_decomp_count,
                            current_decomp_count);
        }

        cudaMemcpyAsync(ct.data(), temp_ntt.data(),
                        temp_ntt.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice, options.stream_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());
    }

    __host__ Ciphertext<Scheme::CKKS> HEBatchConvPack<Scheme::CKKS>::pack_coeffs(
        HEArithmeticOperator<Scheme::CKKS>& operators,
        std::vector<Ciphertext<Scheme::CKKS>>& ct_b, int B,
        const ExecutionOptions& options)
    {
        if (static_cast<int>(ct_b.size()) != B)
        {
            throw std::invalid_argument("ct_b size must be B!");
        }
        if (B <= 0)
        {
            throw std::invalid_argument("B must be positive!");
        }

        // Ensure all ciphertexts are at same level/scale.
        for (int b = 1; b < B; b++)
        {
            if (ct_b[b].depth() != ct_b[0].depth())
            {
                throw std::invalid_argument("Ciphertext levels must match!");
            }
            if (ct_b[b].scale() != ct_b[0].scale())
            {
                throw std::invalid_argument("Ciphertext scales must match!");
            }
            if (ct_b[b].size() != ct_b[0].size())
            {
                throw std::invalid_argument("Ciphertext sizes must match!");
            }
        }

        Ciphertext<Scheme::CKKS> acc = ct_b[0];

        for (int b = 1; b < B; b++)
        {
            Ciphertext<Scheme::CKKS> tmp = ct_b[b];
            monomial_shift_inplace(tmp, b, options);
            operators.add_inplace(acc, tmp, options);
        }

        return acc;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEBatchConvPack<Scheme::CKKS>::project_strideB(
        HEArithmeticOperator<Scheme::CKKS>& operators,
        Ciphertext<Scheme::CKKS>& ct, int B, Galoiskey<Scheme::CKKS>& gk,
        const ExecutionOptions& options)
    {
        if (B <= 0 || (B & (B - 1)) != 0)
        {
            throw std::invalid_argument("B must be a power-of-two positive!");
        }
        if (n_ % B != 0)
        {
            throw std::invalid_argument("N must be divisible by B!");
        }
        if (ct.rescale_required() || ct.relinearization_required())
        {
            throw std::invalid_argument(
                "Ciphertext can not be projected (rescale/relin required)!");
        }

        Ciphertext<Scheme::CKKS> acc = ct;

        // Subgroup elements: g_t = 1 + t*(2N/B) mod 2N, t=1..B-1.
        const int twoN = 2 * n_;
        const int step = twoN / B;
        for (int t = 1; t < B; t++)
        {
            const int g = 1 + t * step;
            Ciphertext<Scheme::CKKS> tmp;
            operators.apply_galois(ct, tmp, gk, g, options);
            operators.add_inplace(acc, tmp, options);
        }

        // acc is scaled by B; caller may compensate if desired.
        return acc;
    }

    __host__ Ciphertext<Scheme::CKKS>
    HEBatchConvPack<Scheme::CKKS>::pack_coeffs_strideB(
        HEArithmeticOperator<Scheme::CKKS>& operators,
        std::vector<Ciphertext<Scheme::CKKS>>& ct_b, int B,
        Galoiskey<Scheme::CKKS>& gk, const ExecutionOptions& options)
    {
        if (static_cast<int>(ct_b.size()) != B)
        {
            throw std::invalid_argument("ct_b size must be B!");
        }
        if (B <= 0)
        {
            throw std::invalid_argument("B must be positive!");
        }

        // Project and pack.
        Ciphertext<Scheme::CKKS> out =
            project_strideB(operators, ct_b[0], B, gk, options);

        for (int b = 1; b < B; b++)
        {
            Ciphertext<Scheme::CKKS> proj =
                project_strideB(operators, ct_b[b], B, gk, options);
            monomial_shift_inplace(proj, b, options);
            operators.add_inplace(out, proj, options);
        }

        return out;
    }

} // namespace heongpu
