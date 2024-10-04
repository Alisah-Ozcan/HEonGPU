﻿// Copyright 2024 Alişah Özcan
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
      public:
        /**
         * @brief Constructs a new HEEncoder object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encode parameters.
         */
        __host__ HEEncoder(Parameters& context);

        // for bfv

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
        __host__ void encode(Plaintext& plain,
                             const std::vector<uint64_t>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message);
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
         * @brief Encodes a message into a plaintext asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of unsigned 64-bit integers representing the
         * message to be encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const std::vector<uint64_t>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message, stream);
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
        __host__ void encode(Plaintext& plain,
                             const std::vector<int64_t>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message);
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
         * @brief Encodes a message of signed 64-bit integers into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of signed 64-bit integers representing the
         * message to be encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const std::vector<int64_t>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message, stream);
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
        __host__ void encode(Plaintext& plain,
                             const HostVector<uint64_t>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message);
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
         * @brief Encodes a message into a plaintext asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of unsigned 64-bit integers representing
         * the message to be encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const HostVector<uint64_t>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message, stream);
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
        __host__ void encode(Plaintext& plain,
                             const HostVector<int64_t>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message);
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
         * @brief Encodes a message of signed 64-bit integers into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of signed 64-bit integers representing the
         * message to be encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const HostVector<int64_t>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    encode_bfv(plain, message, stream);
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
        __host__ void encode(Plaintext& plain,
                             const std::vector<double>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of double values into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of double values representing the message to be
         * encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const std::vector<double>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale, stream);
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
        __host__ void encode(Plaintext& plain,
                             const HostVector<double>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of double values into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of double values representing the message
         * to be encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const HostVector<double>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale, stream);
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
         * @param message Vector of COMPLEX_C representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const std::vector<COMPLEX_C>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of complex numbers into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message Vector of COMPLEX_C representing the message to be
         * encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const std::vector<COMPLEX_C>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale, stream);
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
         * @param message HostVector of COMPLEX_C representing the message to be
         * encoded.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const HostVector<COMPLEX_C>& message,
                             double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Encodes a message of complex numbers into a plaintext
         * asynchronously.
         *
         * @param plain Plaintext object where the result of the encoding will
         * be stored.
         * @param message HostVector of COMPLEX_C representing the message to be
         * encoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         * @param scale parameter defining encoding precision(for CKKS), default
         * is 0.
         */
        __host__ void encode(Plaintext& plain,
                             const HostVector<COMPLEX_C>& message,
                             HEStream& stream, double scale = 0)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    encode_ckks(plain, message, scale, stream);
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
        __host__ void decode(std::vector<uint64_t>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain);
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
         * @brief Decodes a plaintext into a vector of unsigned 64-bit integers
         * asynchronously.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(std::vector<uint64_t>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain, stream);
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
        __host__ void decode(std::vector<int64_t>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain);
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
         * @brief Decodes a plaintext into a vector of signed 64-bit integers
         * asynchronously.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(std::vector<int64_t>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain, stream);
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
        __host__ void decode(HostVector<uint64_t>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain);
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
         * @brief Decodes a plaintext into a HostVector of unsigned 64-bit
         * integers asynchronously.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(HostVector<uint64_t>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain, stream);
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
        __host__ void decode(HostVector<int64_t>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain);
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
         * @brief Decodes a plaintext into a HostVector of signed 64-bit
         * integers asynchronously.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(HostVector<int64_t>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    decode_bfv(message, plain, stream);
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
        __host__ void decode(std::vector<double>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a vector of double values
         * asynchronously.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(std::vector<double>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain, stream);
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
        __host__ void decode(HostVector<double>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a HostVector of double values
         * asynchronously.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(HostVector<double>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain, stream);
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
         * @brief Decodes a plaintext into a vector of complex numbers.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         */
        __host__ void decode(std::vector<COMPLEX_C>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a vector of complex numbers
         * asynchronously.
         *
         * @param message Vector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(std::vector<COMPLEX_C>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain, stream);
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
        __host__ void decode(HostVector<COMPLEX_C>& message, Plaintext& plain)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain);
                    break;
                case 3: // BGV

                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Decodes a plaintext into a HostVector of complex numbers
         * asynchronously.
         *
         * @param message HostVector where the decoded message will be stored.
         * @param plain Plaintext object to be decoded.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous operation.
         */
        __host__ void decode(HostVector<COMPLEX_C>& message, Plaintext& plain,
                             HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    throw std::invalid_argument(
                        "BFV message can not be double");
                    break;
                case 2: // CKKS
                    decode_ckks(message, plain, stream);
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
        // for bfv

        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<uint64_t>& message);

        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<uint64_t>& message,
                                 HEStream& stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<int64_t>& message);

        __host__ void encode_bfv(Plaintext& plain,
                                 const std::vector<int64_t>& message,
                                 HEStream& stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<uint64_t>& message);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<uint64_t>& message,
                                 HEStream& stream);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<int64_t>& message);

        __host__ void encode_bfv(Plaintext& plain,
                                 const HostVector<int64_t>& message,
                                 HEStream& stream);

        //

        __host__ void decode_bfv(std::vector<uint64_t>& message,
                                 Plaintext& plain);

        __host__ void decode_bfv(std::vector<uint64_t>& message,
                                 Plaintext& plain, HEStream& stream);

        __host__ void decode_bfv(std::vector<int64_t>& message,
                                 Plaintext& plain);

        __host__ void decode_bfv(std::vector<int64_t>& message,
                                 Plaintext& plain, HEStream& stream);

        __host__ void decode_bfv(HostVector<uint64_t>& message,
                                 Plaintext& plain);

        __host__ void decode_bfv(HostVector<uint64_t>& message,
                                 Plaintext& plain, HEStream& stream);

        __host__ void decode_bfv(HostVector<int64_t>& message,
                                 Plaintext& plain);

        __host__ void decode_bfv(HostVector<int64_t>& message, Plaintext& plain,
                                 HEStream& stream);

        // for ckks

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<double>& message,
                                  const double scale);

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<double>& message,
                                  const double scale, HEStream& stream);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<double>& message,
                                  const double scale);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<double>& message,
                                  const double scale, HEStream& stream);

        //

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<COMPLEX_C>& message,
                                  const double scale);

        __host__ void encode_ckks(Plaintext& plain,
                                  const std::vector<COMPLEX_C>& message,
                                  const double scale, HEStream& stream);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<COMPLEX_C>& message,
                                  const double scale);

        __host__ void encode_ckks(Plaintext& plain,
                                  const HostVector<COMPLEX_C>& message,
                                  const double scale, HEStream& stream);

        //

        __host__ void decode_ckks(std::vector<double>& message,
                                  Plaintext& plain);

        __host__ void decode_ckks(std::vector<double>& message,
                                  Plaintext& plain, HEStream& stream);

        __host__ void decode_ckks(HostVector<double>& message,
                                  Plaintext& plain);

        __host__ void decode_ckks(HostVector<double>& message, Plaintext& plain,
                                  HEStream& stream);

        //

        __host__ void decode_ckks(std::vector<COMPLEX_C>& message,
                                  Plaintext& plain);

        __host__ void decode_ckks(std::vector<COMPLEX_C>& message,
                                  Plaintext& plain, HEStream& stream);

        __host__ void decode_ckks(HostVector<COMPLEX_C>& message,
                                  Plaintext& plain);

        __host__ void decode_ckks(HostVector<COMPLEX_C>& message,
                                  Plaintext& plain, HEStream& stream);

      private:
        scheme_type scheme;

        int n;

        int n_power;

        int slot_count_;

        DeviceVector<Data> encoding_location_;

        // BFV
        std::shared_ptr<DeviceVector<Modulus>> plain_modulus_;
        std::shared_ptr<DeviceVector<Ninverse>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root>> plain_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> plain_intt_tables_;

        // CKKS
        double two_pow_64;

        double fft_root;
        DeviceVector<COMPLEX> fft_roots_table_;
        DeviceVector<COMPLEX> ifft_roots_table_;
        DeviceVector<COMPLEX> temp_complex;

        int Q_size_;
        std::shared_ptr<DeviceVector<Modulus>> modulus_;
        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;

        std::shared_ptr<DeviceVector<Data>> Mi_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_;
        std::shared_ptr<DeviceVector<Data>> upper_half_threshold_;
        std::shared_ptr<DeviceVector<Data>> decryption_modulus_;

      private:
        void GenerateRootTables();
    };

} // namespace heongpu
#endif // ENCODER_H