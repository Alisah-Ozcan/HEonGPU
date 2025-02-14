// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef ENCODER_H
#define ENCODER_H

#include "common.cuh"
#include "cuda_runtime.h"
#include "encoding.cuh"
#include "ntt.cuh"
#include "fft.cuh"
#include "context.cuh"
#include "plaintext.cuh"

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
    class HEEncoder
    {
        friend class HEOperator;
        friend class HEArithmeticOperator;
        friend class HELogicOperator;

      public:
        /**
         * @brief Constructs a new HEEncoder object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encode parameters.
         */
        __host__ HEEncoder(Parameters& context);

        /**
         * @brief Encodes a message into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of unsigned 64-bit integers representing the
         * message to be encoded.
         * @param scale Parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext& plain, const std::vector<uint64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_bfv(plain_, message, options.stream_);

                            plain.plain_size_ = n;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = 0;
                            plain.in_ntt_domain_ = false;
                        },
                        options);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be uint64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of signed 64-bit integers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of signed 64-bit integers representing the
         * message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext& plain, const std::vector<int64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_bfv(plain_, message, options.stream_);

                            plain.plain_size_ = n;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = 0;
                            plain.in_ntt_domain_ = false;
                        },
                        options);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be int64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of unsigned 64-bit integers representing
         * the message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext& plain, const HostVector<uint64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_bfv(plain_, message, options.stream_);

                            plain.plain_size_ = n;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = 0;
                            plain.in_ntt_domain_ = false;
                        },
                        options);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be uint64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of signed 64-bit integers into a plaintext.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of signed 64-bit integers representing the
         * message to be encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void
        encode(Plaintext& plain, const HostVector<int64_t>& message,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_bfv(plain_, message, options.stream_);

                            plain.plain_size_ = n;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = 0;
                            plain.in_ntt_domain_ = false;
                        },
                        options);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be int64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

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
        encode(Plaintext& plain, const std::vector<double>& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        encode(Plaintext& plain, const HostVector<double>& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        encode(Plaintext& plain, const std::vector<Complex64>& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        encode(Plaintext& plain, const HostVector<Complex64>& message,
               double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if (message.size() > slot_count_)
                        throw std::invalid_argument(
                            "Vector size can not be higher than slot count!");

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        encode(Plaintext& plain, const double& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV does not support this feature");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if ((static_cast<int>(log2(fabs(message))) + 2) >=
                        total_coeff_bit_count_)
                    {
                        throw std::invalid_argument(
                            "Encoded value is too large");
                    }

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        encode(Plaintext& plain, const std::int64_t& message, double scale,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV does not support this feature");
                    break;
                case 2: // CKKS
                    if ((scale <= 0) || (static_cast<int>(log2(scale)) >=
                                         total_coeff_bit_count_))
                    {
                        throw std::invalid_argument("Scale out of bounds");
                    }

                    if ((static_cast<int>(log2(fabs(message))) + 2) >=
                        total_coeff_bit_count_)
                    {
                        throw std::invalid_argument(
                            "Encoded value is too large");
                    }

                    output_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        {
                            encode_ckks(plain_, message, scale,
                                        options.stream_);

                            plain.plain_size_ = n * Q_size_;
                            plain.scheme_ = scheme_;
                            plain.depth_ = 0;
                            plain.scale_ = scale;
                            plain.in_ntt_domain_ = true;
                        },
                        options);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

        /**
         * @brief Decodes a plaintext into a vector of unsigned 64-bit integers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<uint64_t>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_bfv(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be uint64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a vector of signed 64-bit integers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<int64_t>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_bfv(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be int64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of unsigned 64-bit
         * integers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<uint64_t>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_bfv(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be uint64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a HostVector of signed 64-bit
         * integers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<int64_t>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_bfv(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS message can not be int64_t");
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

        /**
         * @brief Decodes a plaintext into a vector of double values.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<double>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_ckks(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of double values.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<double>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_ckks(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a vector of complex numbers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(std::vector<Complex64>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_ckks(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        //

        /**
         * @brief Decodes a plaintext into a HostVector of complex numbers.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void
        decode(HostVector<Complex64>& message, Plaintext& plain,
               const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        plain,
                        [&](Plaintext& plain_)
                        { decode_ckks(message, plain_, options.stream_); },
                        options, false);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
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
        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<uint64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<int64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<uint64_t>& message,
                                 const cudaStream_t stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<int64_t>& message,
                                 const cudaStream_t stream);

        //

        __host__ void decode_bfv(std::vector<uint64_t>& message,
                                 Plaintext& plain, const cudaStream_t stream);

        __host__ void decode_bfv(std::vector<int64_t>& message,
                                 Plaintext& plain, const cudaStream_t stream);

        __host__ void decode_bfv(HostVector<uint64_t>& message,
                                 Plaintext& plain, const cudaStream_t stream);

        __host__ void decode_bfv(HostVector<int64_t>& message, Plaintext& plain,
                                 const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<double>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<Complex64>& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void encode_ckks(Plaintext& plain, const double& message,
                                  const double scale,
                                  const cudaStream_t stream);

        __host__ void encode_ckks(Plaintext& plain, const std::int64_t& message,
                                  const double scale,
                                  const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<double>& message,
                                  Plaintext& plain, const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<double>& message, Plaintext& plain,
                                  const cudaStream_t stream);

        //

        __host__ void decode_ckks(std::vector<Complex64>& message,
                                  Plaintext& plain, const cudaStream_t stream);

        __host__ void decode_ckks(HostVector<Complex64>& message,
                                  Plaintext& plain, const cudaStream_t stream);

        //////////////////////////////
        //////////////////////////////

      private:
        scheme_type scheme_;

        int n;
        int n_power;
        int slot_count_;

        // BFV
        std::shared_ptr<DeviceVector<Modulus64>> plain_modulus_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root64>> plain_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> plain_intt_tables_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;

        // CKKS
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
#endif // ENCODER_H
