// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_ENCODER_H
#define HEONGPU_CKKS_ENCODER_H

#include "ntt.cuh"
#include "fft.cuh"
#include "encoding.cuh"
#include "ckks/context.cuh"
#include "ckks/plaintext.cuh"

namespace heongpu
{

    /**
     * @brief HEEncoder is responsible for encoding messages into plaintexts
     * suitable for homomorphic encryption.
     *
     * The HEEncoder class is initialized with encryption parameters and
     * provides methods to encode different types of messages (e.g., integers,
     * floating-point numbers, complex numbers) into plaintexts for BFV and CKKS
     * schemes. The class supports both synchronous and asynchronous encoding
     * operations, making it suitable for different homomorphic encryption
     * workflows.
     */
    template <> class HEEncoder<Scheme::CKKS>
    {
        template <Scheme S> friend class HEOperator;
        template <Scheme S> friend class HEArithmeticOperator;
        template <Scheme S> friend class HELogicOperator;

      public:
        /**
         * @brief Constructs a new HEEncoder object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encode parameters.
         */
        __host__ HEEncoder(HEContext<Scheme::CKKS>& context);

        /**
         * @brief Encodes a message of double values into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of double values representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<double>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of double values into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of double values representing the message
         * to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<double>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of complex numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of Complex64 representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const std::vector<Complex64>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of complex numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of Complex64 representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain,
               const HostVector<Complex64>& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if (message.size() > slot_count_)
                throw std::invalid_argument(
                    "Vector size can not be higher than slot count!");

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of single double numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Double representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain, const double& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if ((static_cast<int>(log2(fabs(message))) + 2) >=
                total_coeff_bit_count_)
            {
                throw std::invalid_argument("Encoded value is too large");
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        //

        /**
         * @brief Encodes a message of single int64_t numbers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message int64_t representing the message to be
         * encoded.
         */
        __host__ void
        encode(Plaintext<Scheme::CKKS>& plain, const std::int64_t& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            if ((scale <= 0) ||
                (static_cast<int>(log2(scale)) >= total_coeff_bit_count_))
            {
                throw std::invalid_argument("Scale out of bounds");
            }

            if ((static_cast<int>(log2(fabs(message))) + 2) >=
                total_coeff_bit_count_)
            {
                throw std::invalid_argument("Encoded value is too large");
            }

            output_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS>& plain_)
                {
                    encode_ckks(plain_, message, scale, options.stream_);

                    plain.plain_size_ = n * Q_size_;
                    plain.scheme_ = scheme_;
                    plain.depth_ = 0;
                    plain.scale_ = scale;
                    plain.in_ntt_domain_ = true;
                    plain.plaintext_generated_ = true;
                },
                options);
        }

        /**
         * @brief Decodes a plaintext into a vector of double values.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<double>& message, Plaintext<Scheme::CKKS> plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS> plain_)
                { decode_ckks(message, plain_, options.stream_); },
                options, false);
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of double values.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<double>& message, Plaintext<Scheme::CKKS> plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS> plain_)
                { decode_ckks(message, plain_, options.stream_); },
                options, false);
        }

        /**
         * @brief Decodes a plaintext into a vector of complex numbers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<Complex64>& message, Plaintext<Scheme::CKKS> plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS> plain_)
                { decode_ckks(message, plain_, options.stream_); },
                options, false);
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of complex numbers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<Complex64>& message, Plaintext<Scheme::CKKS> plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            input_storage_manager(
                plain,
                [&](Plaintext<Scheme::CKKS> plain_)
                { decode_ckks(message, plain_, options.stream_); },
                options, false);
        }

        /**
         * @brief Returns the number of slots.
         *
         * @return int Number of slots.
         */
        inline int slot_count() const noexcept { return slot_count_; }

        HEEncoder() = default;
        HEEncoder(const HEEncoder& copy) = default;
        HEEncoder(HEEncoder&& source) = default;
        HEEncoder& operator=(const HEEncoder& assign) = default;
        HEEncoder& operator=(HEEncoder&& assign) = default;

      private:
        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::vector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const HostVector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::vector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const HostVector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const double& message, const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext<Scheme::CKKS>& plain,
                                  const std::int64_t& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<double>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<double>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<Complex64>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<Complex64>& message,
                                  Plaintext<Scheme::CKKS>& plain,
                                  const cudaStream_t stream);

      private:
        scheme_type scheme_;

        int n;
        int n_power;
        int slot_count_;

        double two_pow_64;
        int log_slot_count_;
        int fft_length;
        Complex64 special_root;

        std::shared_ptr<DeviceVector<Complex64>> special_fft_roots_table_;
        std::shared_ptr<DeviceVector<Complex64>> special_ifft_roots_table_;
        std::shared_ptr<DeviceVector<int>> reverse_order;

        int Q_size_;
        int total_coeff_bit_count_;
        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> Mi_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data64>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data64>> decryption_modulus_;
    };

} // namespace heongpu
#endif // HEONGPU_CKKS_ENCODER_H
