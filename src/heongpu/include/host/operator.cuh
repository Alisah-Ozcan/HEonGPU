// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef OPERATOR_H
#define OPERATOR_H

#include "addition.cuh"
#include "multiplication.cuh"
#include "switchkey.cuh"
#include "keygeneration.cuh"

#include "keyswitch.cuh"
#include "ciphertext.cuh"
#include "plaintext.cuh"

namespace heongpu
{
    /**
     * @brief HEOperator is responsible for performing homomorphic operations on
     * encrypted data, such as addition, subtraction, multiplication, and other
     * functions.
     *
     * The HEOperator class is initialized with encryption parameters and
     * provides various functions for performing operations on ciphertexts,
     * including BFV and CKKS schemes. It supports both in-place and
     * out-of-place operations, as well as asynchronous processing using CUDA
     * streams.
     */
    class HEOperator
    {
      public:
        /**
         * @brief Construct a new HEOperator object with the given parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters for the operator.
         */
        __host__ HEOperator(Parameters& context);

        /**
         * @brief Adds two ciphertexts and stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output);

        /**
         * @brief Adds two ciphertexts asynchronously with a given stream and
         * stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void add(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output, HEStream& stream);

        /**
         * @brief Adds the second ciphertext to the first ciphertext, modifying
         * the first ciphertext with the result.
         *
         * @param input1 The ciphertext to which the value of input2 will be
         * added.
         * @param input2 The ciphertext to be added to input1.
         */
        __host__ void add_inplace(Ciphertext& input1, Ciphertext& input2)
        {
            add(input1, input2, input1);
        }

        /**
         * @brief Adds the second ciphertext to the first asynchronously,
         * modifying the first ciphertext with the result.
         *
         * @param input1 The ciphertext to which the value of input2 will be
         * added.
         * @param input2 The ciphertext to be added to input1.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void add_inplace(Ciphertext& input1, Ciphertext& input2,
                                  HEStream& stream)
        {
            add(input1, input2, input1, stream);
        }

        /**
         * @brief Subtracts the second ciphertext from the first and stores the
         * result in the output.
         *
         * @param input1 First input ciphertext (minuend).
         * @param input2 Second input ciphertext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         */
        __host__ void sub(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output);

        /**
         * @brief Subtracts the second ciphertext from the first asynchronously
         * and stores the result in the output.
         *
         * @param input1 First input ciphertext (minuend).
         * @param input2 Second input ciphertext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void sub(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output, HEStream& stream);

        /**
         * @brief Subtracts the second ciphertext from the first, modifying the
         * first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         */
        __host__ void sub_inplace(Ciphertext& input1, Ciphertext& input2)
        {
            sub(input1, input2, input1);
        }

        /**
         * @brief Subtracts the second ciphertext from the first asynchronously,
         * modifying the first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void sub_inplace(Ciphertext& input1, Ciphertext& input2,
                                  HEStream& stream)
        {
            sub(input1, input2, input1, stream);
        }

        /**
         * @brief Negates a ciphertext and stores the result in the output.
         *
         * @param input1 Input ciphertext to be negated.
         * @param output Ciphertext where the result of the negation is stored.
         */
        __host__ void negate(Ciphertext& input1, Ciphertext& output);

        /**
         * @brief Negates a ciphertext asynchronously and stores the result in
         * the output.
         *
         * @param input1 Input ciphertext to be negated.
         * @param output Ciphertext where the result of the negation is stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void negate(Ciphertext& input1, Ciphertext& output,
                             HEStream& stream);

        /**
         * @brief Negates a ciphertext in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         */
        __host__ void negate_inplace(Ciphertext& input1)
        {
            negate(input1, input1);
        }

        /**
         * @brief Negates a ciphertext asynchronously in-place, modifying the
         * input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void negate_inplace(Ciphertext& input1, HEStream& stream)
        {
            negate(input1, input1, stream);
        }

        /**
         * @brief Adds a ciphertext and a plaintext and stores the result in the
         * output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input plaintext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add_plain(Ciphertext& input1, Plaintext& input2,
                                Ciphertext& output)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    add_plain_bfv(input1, input2, output);
                    break;
                }
                case 2: // CKKS
                {
                    add_plain_ckks(input1, input2, output);
                    output.scale_ = input1.scale_;
                    break;
                }
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Adds a ciphertext and a plaintext asynchronously and stores
         * the result in the output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input plaintext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void add_plain(Ciphertext& input1, Plaintext& input2,
                                Ciphertext& output, HEStream& stream)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    add_plain_bfv(input1, input2, output, stream);
                    break;
                }
                case 2: // CKKS
                {
                    add_plain_ckks(input1, input2, output, stream);
                    output.scale_ = input1.scale_;
                    break;
                }
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Adds a plaintext to a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Plaintext to be added to the ciphertext.
         */
        __host__ void add_plain_inplace(Ciphertext& input1, Plaintext& input2)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    add_plain_bfv_inplace(input1, input2);
                    break;
                }
                case 2: // CKKS
                    add_plain_ckks_inplace(input1, input2);
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Adds a plaintext to a ciphertext asynchronously in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Plaintext to be added to the ciphertext.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void add_plain_inplace(Ciphertext& input1, Plaintext& input2,
                                        HEStream& stream)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    add_plain_bfv_inplace(input1, input2, stream);
                    break;
                }
                case 2: // CKKS
                    add_plain_ckks_inplace(input1, input2, stream);
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext (minuend).
         * @param input2 Input plaintext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         */
        __host__ void sub_plain(Ciphertext& input1, Plaintext& input2,
                                Ciphertext& output)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    sub_plain_bfv(input1, input2, output);
                    break;
                }
                case 2: // CKKS
                {
                    sub_plain_ckks(input1, input2, output);
                    output.scale_ = input1.scale_;
                    break;
                }
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext (minuend).
         * @param input2 Input plaintext (subtrahend).
         * @param output Ciphertext where the result of the subtraction is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void sub_plain(Ciphertext& input1, Plaintext& input2,
                                Ciphertext& output, HEStream& stream)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
            {
                throw std::logic_error(
                    "Ciphertext and Plaintext should be in same domain");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    sub_plain_bfv(input1, input2, output, stream);
                    break;
                }
                case 2: // CKKS
                {
                    sub_plain_ckks(input1, input2, output, stream);
                    output.scale_ = input1.scale_;
                    break;
                }
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext from which the plaintext will be subtracted.
         * @param input2 Plaintext to be subtracted from the ciphertext.
         */
        __host__ void sub_plain_inplace(Ciphertext& input1, Plaintext& input2)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    sub_plain_bfv_inplace(input1, input2);
                    break;
                }
                case 2: // CKKS
                    sub_plain_ckks_inplace(input1, input2);
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext asynchronously
         * in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext from which the plaintext will be subtracted.
         * @param input2 Plaintext to be subtracted from the ciphertext.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void sub_plain_inplace(Ciphertext& input1, Plaintext& input2,
                                        HEStream& stream)
        {
            if (input1.depth_ != input2.depth_)
            {
                throw std::logic_error("Ciphertexts leveled are not equal");
            }

            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ || input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in NTT domain");
                    }
                    sub_plain_bfv_inplace(input1, input2, stream);
                    break;
                }
                case 2: // CKKS
                    sub_plain_ckks_inplace(input1, input2, stream);
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Multiplies two ciphertexts and stores the result in the
         * output.
         *
         * @param input1 First input ciphertext to be multiplied.
         * @param input2 Second input ciphertext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void multiply(Ciphertext& input1, Ciphertext& input2,
                               Ciphertext& output)
        {
            if (input1.relinearization_required_ ||
                input2.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the "
                    "non-linear part! Please use relinearization operation!");
            }

            if (input1.rescale_required_ || input2.rescale_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the noise! "
                    "Please use rescale operation to get rid of additional "
                    "noise!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    multiply_bfv(input1, input2, output);
                    break;
                case 2: // CKKS
                    multiply_ckks(input1, input2, output);
                    output.rescale_required_ = true;
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = true;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Multiplies two ciphertexts asynchronously and stores the
         * result in the output.
         *
         * @param input1 First input ciphertext to be multiplied.
         * @param input2 Second input ciphertext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void multiply(Ciphertext& input1, Ciphertext& input2,
                               Ciphertext& output, HEStream& stream)
        {
            if (input1.relinearization_required_ ||
                input2.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the "
                    "non-linear part! Please use relinearization operation!");
            }

            if (input1.rescale_required_ || input2.rescale_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be multiplied because of the noise! "
                    "Please use rescale operation to get rid of additional "
                    "noise!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    multiply_bfv(input1, input2, output, stream);
                    break;
                case 2: // CKKS
                    multiply_ckks(input1, input2, output, stream);
                    output.rescale_required_ = true;
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = true;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Multiplies two ciphertexts in-place, modifying the first
         * ciphertext.
         *
         * @param input1 Ciphertext to be multiplied, and where the result will
         * be stored.
         * @param input2 Second input ciphertext to be multiplied.
         */
        __host__ void multiply_inplace(Ciphertext& input1, Ciphertext& input2)
        {
            multiply(input1, input2, input1);
        }

        /**
         * @brief Multiplies two ciphertexts asynchronously in-place, modifying
         * the first ciphertext.
         *
         * @param input1 Ciphertext to be multiplied, and where the result will
         * be stored.
         * @param input2 Second input ciphertext to be multiplied.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void multiply_inplace(Ciphertext& input1, Ciphertext& input2,
                                       HEStream& stream)
        {
            multiply(input1, input2, input1, stream);
        }

        /**
         * @brief Multiplies a ciphertext and a plaintext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param input2 Input plaintext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void multiply_plain(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in same domain");
                    }
                    multiply_plain_bfv(input1, input2, output);
                    output.rescale_required_ = input1.rescale_required_;
                    break;
                }
                case 2: // CKKS
                    multiply_plain_ckks(input1, input2, output);
                    output.rescale_required_ = true;
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Multiplies a ciphertext and a plaintext asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param input2 Input plaintext to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void multiply_plain(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output, HEStream& stream)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                {
                    if (input1.in_ntt_domain_ != input2.in_ntt_domain_)
                    {
                        throw std::logic_error("BFV ciphertext or plaintext "
                                               "should be not in same domain");
                    }
                    multiply_plain_bfv(input1, input2, output, stream);
                    output.rescale_required_ = input1.rescale_required_;
                    break;
                }
                case 2: // CKKS
                    multiply_plain_ckks(input1, input2, output, stream);
                    output.rescale_required_ = true;
                    break;
                case 3: // BGV
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Plaintext to be multiplied with the ciphertext.
         */
        __host__ void multiply_plain_inplace(Ciphertext& input1,
                                             Plaintext& input2)
        {
            multiply_plain(input1, input2, input1);
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext asynchronously
         * in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Plaintext to be multiplied with the ciphertext.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void multiply_plain_inplace(Ciphertext& input1,
                                             Plaintext& input2,
                                             HEStream& stream)
        {
            multiply_plain(input1, input2, input1, stream);
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         */
        __host__ void relinearize_inplace(Ciphertext& input1,
                                          Relinkey& relin_key)
        {
            if ((!input1.relinearization_required_))
            {
                throw std::invalid_argument(
                    "Ciphertexts can not use relinearization, since no "
                    "non-linear part!");
            }

            switch (static_cast<int>(relin_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_seal_method_inplace(input1, relin_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_seal_method_inplace_ckks(input1, relin_key);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II

                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_external_product_method2_inplace(input1,
                                                                     relin_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_external_product_method2_inplace_ckks(
                            input1, relin_key);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

                    break;
                case 3: // KEYSWITHING_METHOD_III

                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_external_product_method_inplace(input1,
                                                                    relin_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_external_product_method_inplace_ckks(
                            input1, relin_key);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            input1.relinearization_required_ = false;
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * asynchronously using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void relinearize_inplace(Ciphertext& input1,
                                          Relinkey& relin_key, HEStream& stream)
        {
            if ((!input1.relinearization_required_))
            {
                throw std::invalid_argument(
                    "Ciphertexts can not use relinearization, since no "
                    "non-linear part!");
            }

            switch (static_cast<int>(relin_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_seal_method_inplace(input1, relin_key,
                                                        stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_seal_method_inplace_ckks(input1, relin_key,
                                                             stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II

                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_external_product_method2_inplace(
                            input1, relin_key, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_external_product_method2_inplace_ckks(
                            input1, relin_key, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

                    break;
                case 3: // KEYSWITHING_METHOD_III

                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        relinearize_external_product_method_inplace(
                            input1, relin_key, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        relinearize_external_product_method_inplace_ckks(
                            input1, relin_key, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            input1.relinearization_required_ = false;
        }

        /**
         * @brief Rotates the rows of a ciphertext by a given shift value and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         */
        __host__ void rotate_rows(Ciphertext& input1, Ciphertext& output,
                                  Galoiskey& galois_key, int shift)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_method_I(input1, output, galois_key, shift);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        rotate_ckks_method_I(input1, output, galois_key, shift);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_method_II(input1, output, galois_key, shift);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        rotate_ckks_method_II(input1, output, galois_key,
                                              shift);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Rotates the rows of a ciphertext asynchronously by a given
         * shift value and stores the result in the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void rotate_rows(Ciphertext& input1, Ciphertext& output,
                                  Galoiskey& galois_key, int shift,
                                  HEStream& stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_method_I(input1, output, galois_key, shift,
                                        stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        rotate_ckks_method_I(input1, output, galois_key, shift,
                                             stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_method_II(input1, output, galois_key, shift,
                                         stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        rotate_ckks_method_II(input1, output, galois_key, shift,
                                              stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Rotates the rows of a ciphertext in-place by a given shift
         * value, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be rotated.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         */
        __host__ void rotate_rows_inplace(Ciphertext& input1,
                                          Galoiskey& galois_key, int shift)
        {
            rotate_rows(input1, input1, galois_key, shift);
        }

        /**
         * @brief Rotates the rows of a ciphertext asynchronously in-place by a
         * given shift value, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be rotated.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void rotate_rows_inplace(Ciphertext& input1,
                                          Galoiskey& galois_key, int shift,
                                          HEStream& stream)
        {
            rotate_rows(input1, input1, galois_key, shift, stream);
        }

        /**
         * @brief Rotates the columns of a ciphertext and stores the result in
         * the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         */
        __host__ void rotate_columns(Ciphertext& input1, Ciphertext& output,
                                     Galoiskey& galois_key)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_columns_method_I(input1, output, galois_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        throw std::invalid_argument("Unsupported scheme");
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        rotate_columns_method_II(input1, output, galois_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        throw std::invalid_argument("Unsupported scheme");
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Rotates the columns of a ciphertext asynchronously and stores
         * the result in the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void rotate_columns(Ciphertext& input1, Ciphertext& output,
                                     Galoiskey& galois_key, HEStream& stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        rotate_columns_method_I(input1, output, galois_key,
                                                stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        throw std::invalid_argument("Unsupported scheme");
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        rotate_columns_method_II(input1, output, galois_key,
                                                 stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        throw std::invalid_argument("Unsupported scheme");
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to which the Galois operation will be
         * applied.
         * @param output Ciphertext where the result of the Galois operation is
         * stored.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         */
        __host__ void apply_galois(Ciphertext& input1, Ciphertext& output,
                                   Galoiskey& galois_key, int galois_elt)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        apply_galois_method_I(input1, output, galois_key,
                                              galois_elt);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        apply_galois_ckks_method_I(input1, output, galois_key,
                                                   galois_elt);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        apply_galois_method_II(input1, output, galois_key,
                                               galois_elt);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        apply_galois_ckks_method_II(input1, output, galois_key,
                                                    galois_elt);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext asynchronously
         * and stores the result in the output.
         *
         * @param input1 Input ciphertext to which the Galois operation will be
         * applied.
         * @param output Ciphertext where the result of the Galois operation is
         * stored.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void apply_galois(Ciphertext& input1, Ciphertext& output,
                                   Galoiskey& galois_key, int galois_elt,
                                   HEStream& stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        apply_galois_method_I(input1, output, galois_key,
                                              galois_elt, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        apply_galois_ckks_method_I(input1, output, galois_key,
                                                   galois_elt, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        apply_galois_method_II(input1, output, galois_key,
                                               galois_elt, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        apply_galois_ckks_method_II(input1, output, galois_key,
                                                    galois_elt, stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for rotation operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to which the Galois operation will be
         * applied.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         */
        __host__ void apply_galois_inplace(Ciphertext& input1,
                                           Galoiskey& galois_key,
                                           int galois_elt)
        {
            apply_galois(input1, input1, galois_key, galois_elt);
        }

        /**
         * @brief Applies a Galois automorphism to the ciphertext asynchronously
         * in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to which the Galois operation will be
         * applied.
         * @param galois_key Galois key used for the operation.
         * @param galois_elt The Galois element to apply.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void apply_galois_inplace(Ciphertext& input1,
                                           Galoiskey& galois_key,
                                           int galois_elt, HEStream& stream)
        {
            apply_galois(input1, input1, galois_key, galois_elt, stream);
        }

        /**
         * @brief Performs key switching on the ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be key-switched.
         * @param output Ciphertext where the result of the key switching is
         * stored.
         * @param switch_key Switch key used for the key switching operation.
         */
        __host__ void keyswitch(Ciphertext& input1, Ciphertext& output,
                                Switchkey& switch_key)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(switch_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        switchkey_method_I(input1, output, switch_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        switchkey_ckks_method_I(input1, output, switch_key);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        switchkey_method_II(input1, output, switch_key);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        switchkey_ckks_method_II(input1, output, switch_key);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for keyswitch operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Performs key switching on the ciphertext asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be key-switched.
         * @param output Ciphertext where the result of the key switching is
         * stored.
         * @param switch_key Switch key used for the key switching operation.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void keyswitch(Ciphertext& input1, Ciphertext& output,
                                Switchkey& switch_key, HEStream& stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            switch (static_cast<int>(switch_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        switchkey_method_I(input1, output, switch_key, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        switchkey_ckks_method_I(input1, output, switch_key,
                                                stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme == scheme_type::bfv)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        switchkey_method_II(input1, output, switch_key, stream);
                    }
                    else if (scheme == scheme_type::ckks)
                    {
                        switchkey_ckks_method_II(input1, output, switch_key,
                                                 stream);
                    }
                    else
                    {
                        throw std::invalid_argument(
                            "Invalid Key Switching Type");
                    }
                    break;
                case 3: // KEYSWITHING_METHOD_III

                    throw std::invalid_argument(
                        "KEYSWITHING_METHOD_III are not supported because of "
                        "high memory consumption for keyswitch operation!");

                    break;
                default:
                    throw std::invalid_argument("Invalid Key Switching Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Rescales a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to be rescaled.
         */
        __host__ void rescale_inplace(Ciphertext& input1)
        {
            if ((!input1.rescale_required_) || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertexts can not be rescaled!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument("BFV Does Not Support!");
                    break;
                case 2: // CKKS
                    rescale_inplace_ckks_leveled(input1);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            input1.rescale_required_ = false;
        }

        /**
         * @brief Rescales a ciphertext asynchronously in-place, modifying the
         * input ciphertext.
         *
         * @param input1 Ciphertext to be rescaled.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void rescale_inplace(Ciphertext& input1, HEStream& stream)
        {
            if ((!input1.rescale_required_) || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertexts can not be rescaled!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument("BFV Does Not Support!");
                    break;
                case 2: // CKKS
                    rescale_inplace_ckks_leveled(input1, stream);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            input1.rescale_required_ = false;
        }

        /**
         * @brief Drop the last modulus of ciphertext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input ciphertext from which last modulus will be
         * dropped.
         * @param output Ciphertext where the result of the modulus drop is
         * stored.
         */
        __host__ void mod_drop(Ciphertext& input1, Ciphertext& output)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext's modulus can not be dropped!!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_leveled(input1, output);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Drop the last modulus of ciphertext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input ciphertext from which last modulus will be
         * dropped.
         * @param output Ciphertext where the result of the modulus drop is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void mod_drop(Ciphertext& input1, Ciphertext& output,
                               HEStream& stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext's modulus can not be dropped!!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_leveled(input1, output, stream);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Drop the last modulus of plaintext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input plaintext from which modulus will be dropped.
         * @param output Plaintext where the result of the modulus drop is
         * stored.
         */
        __host__ void mod_drop(Plaintext& input1, Plaintext& output)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_plaintext(input1, output);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
        }

        /**
         * @brief Drop the last modulus of plaintext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input plaintext from which modulus will be dropped.
         * @param output Plaintext where the result of the modulus drop is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void mod_drop(Plaintext& input1, Plaintext& output,
                               HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_plaintext(input1, output, stream);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
        }

        /**
         * @brief Drop the last modulus of plaintext in-place on a plaintext,
         * modifying the input plaintext.
         *
         * @param input1 Plaintext to perform modulus dropping on.
         */
        __host__ void mod_drop_inplace(Plaintext& input1)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_plaintext_inplace(input1);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Drop the last modulus of plaintext in-place on a plaintext,
         * modifying the input plaintext.
         *
         * @param input1 Plaintext to perform modulus dropping on.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void mod_drop_inplace(Plaintext& input1, HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_plaintext_inplace(input1);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Drop the last modulus of ciphertext in-place on a ciphertext,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to perform modulus dropping on.
         */
        __host__ void mod_drop_inplace(Ciphertext& input1)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_leveled_inplace(input1);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Drop the last modulus of ciphertext in-place on a ciphertext,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to perform modulus dropping on.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void mod_drop_inplace(Ciphertext& input1, HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    mod_drop_ckks_leveled_inplace(input1, stream);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        __host__ void multiply_power_of_X(Ciphertext& input1,
                                          Ciphertext& output, int index)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (index != 0)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        negacyclic_shift_poly_coeffmod(input1, output, index);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support multiply_power_of_X!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        __host__ void multiply_power_of_X(Ciphertext& input1,
                                          Ciphertext& output, int index,
                                          HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (index != 0)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        negacyclic_shift_poly_coeffmod(input1, output, index,
                                                       stream);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support multiply_power_of_X!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = input1.in_ntt_domain_;
        }

        /**
         * @brief Transforms a plaintext to the NTT domain and stores the result
         * in the output.
         *
         * @param input1 Input plaintext to be transformed.
         * @param output Plaintext where the result of the transformation is
         * stored.
         */
        __host__ void transform_to_ntt(Plaintext& input1, Plaintext& output)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        transform_to_ntt_bfv_plain(input1, output);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_to_ntt_inplace!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Transforms a plaintext to the NTT domain asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input plaintext to be transformed.
         * @param output Plaintext where the result of the transformation is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_to_ntt(Plaintext& input1, Plaintext& output,
                                       HEStream& stream)
        {
            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        transform_to_ntt_bfv_plain(input1, output, stream);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_to_ntt_inplace!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Transforms a plaintext to the NTT domain in-place, modifying
         * the input plaintext.
         *
         * @param input1 Plaintext to be transformed.
         */
        __host__ void transform_to_ntt_inplace(Plaintext& input1)
        {
            transform_to_ntt(input1, input1);
        }

        /**
         * @brief Transforms a plaintext to the NTT domain asynchronously
         * in-place, modifying the input plaintext.
         *
         * @param input1 Plaintext to be transformed.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_to_ntt_inplace(Plaintext& input1,
                                               HEStream& stream)
        {
            transform_to_ntt(input1, input1, stream);
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void transform_to_ntt(Ciphertext& input1, Ciphertext& output)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed to NTT!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        transform_to_ntt_bfv_cipher(input1, output);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_to_ntt_inplace!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = true;
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be transformed.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_to_ntt(Ciphertext& input1, Ciphertext& output,
                                       HEStream& stream)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed to NTT!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        transform_to_ntt_bfv_cipher(input1, output, stream);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_to_ntt_inplace!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = true;
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed.
         */
        __host__ void transform_to_ntt_inplace(Ciphertext& input1)
        {
            transform_to_ntt(input1, input1);
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain asynchronously
         * in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_to_ntt_inplace(Ciphertext& input1,
                                               HEStream& stream)
        {
            transform_to_ntt(input1, input1, stream);
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed from the NTT domain.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void transform_from_ntt(Ciphertext& input1, Ciphertext& output)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed from NTT!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (input1.in_ntt_domain_)
                    {
                        transform_from_ntt_bfv_cipher(input1, output);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_from_ntt!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = false;
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain asynchronously and
         * stores the result in the output.
         *
         * @param input1 Input ciphertext to be transformed from the NTT domain.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_from_ntt(Ciphertext& input1, Ciphertext& output,
                                         HEStream& stream)
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed from NTT!");
            }

            switch (static_cast<int>(scheme))
            {
                case 1: // BFV
                    if (input1.in_ntt_domain_)
                    {
                        transform_from_ntt_bfv_cipher(input1, output, stream);
                    }
                    break;
                case 2: // CKKS
                    throw std::invalid_argument(
                        "CKKS does dot support transform_from_ntt!");
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }

            output.depth_ = input1.depth_;
            output.rescale_required_ = input1.rescale_required_;
            output.relinearization_required_ = input1.relinearization_required_;
            output.in_ntt_domain_ = false;
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed from the NTT domain.
         */
        __host__ void transform_from_ntt_inplace(Ciphertext& input1)
        {
            transform_from_ntt(input1, input1);
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain asynchronously
         * in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed from the NTT domain.
         * @param stream The HEStream object representing the CUDA stream to be
         * used for asynchronous operation.
         */
        __host__ void transform_from_ntt_inplace(Ciphertext& input1,
                                                 HEStream& stream)
        {
            transform_from_ntt(input1, input1, stream);
        }

        ////////////////////////////////////////
        // BGV (Coming Soon)

        ////////////////////////////////////////

        HEOperator() = default;
        HEOperator(const HEOperator& copy) = default;
        HEOperator(HEOperator&& source) = default;
        HEOperator& operator=(const HEOperator& assign) = default;
        HEOperator& operator=(HEOperator&& assign) = default;

      private:
        __host__ void add_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output);
        __host__ void add_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output, HEStream& stream);

        __host__ void add_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2);
        __host__ void add_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2,
                                            HEStream& stream);

        __host__ void add_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output);
        __host__ void add_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output, HEStream& stream);

        __host__ void add_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2);
        __host__ void add_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2,
                                             HEStream& stream);

        __host__ void sub_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output);
        __host__ void sub_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output, HEStream& stream);

        __host__ void sub_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2);
        __host__ void sub_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2,
                                            HEStream& stream);

        __host__ void sub_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output);
        __host__ void sub_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output, HEStream& stream);

        __host__ void sub_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2);
        __host__ void sub_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2,
                                             HEStream& stream);

        __host__ void multiply_bfv(Ciphertext& input1, Ciphertext& input2,
                                   Ciphertext& output);
        __host__ void multiply_bfv(Ciphertext& input1, Ciphertext& input2,
                                   Ciphertext& output, HEStream& stream);

        __host__ void multiply_ckks(Ciphertext& input1, Ciphertext& input2,
                                    Ciphertext& output);
        __host__ void multiply_ckks(Ciphertext& input1, Ciphertext& input2,
                                    Ciphertext& output, HEStream& stream);

        __host__ void multiply_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                         Ciphertext& output);
        __host__ void multiply_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                         Ciphertext& output, HEStream& stream);

        __host__ void multiply_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                          Ciphertext& output);
        __host__ void multiply_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                          Ciphertext& output, HEStream& stream);

        ///////////////////////////////////////////////////

        __host__ void relinearize_seal_method_inplace(Ciphertext& input1,
                                                      Relinkey& relin_key);

        __host__ void relinearize_seal_method_inplace(Ciphertext& input1,
                                                      Relinkey& relin_key,
                                                      HEStream& stream);

        __host__ void
        relinearize_external_product_method_inplace(Ciphertext& input1,
                                                    Relinkey& relin_key);

        __host__ void relinearize_external_product_method_inplace(
            Ciphertext& input1, Relinkey& relin_key, HEStream& stream);

        __host__ void
        relinearize_external_product_method2_inplace(Ciphertext& input1,
                                                     Relinkey& relin_key);

        __host__ void relinearize_external_product_method2_inplace(
            Ciphertext& input1, Relinkey& relin_key, HEStream& stream);

        __host__ void relinearize_seal_method_inplace_ckks(Ciphertext& input1,
                                                           Relinkey& relin_key);

        __host__ void relinearize_seal_method_inplace_ckks(Ciphertext& input1,
                                                           Relinkey& relin_key,
                                                           HEStream& stream);

        __host__ void
        relinearize_external_product_method_inplace_ckks(Ciphertext& input1,
                                                         Relinkey& relin_key);

        __host__ void relinearize_external_product_method_inplace_ckks(
            Ciphertext& input1, Relinkey& relin_key, HEStream& stream);

        __host__ void
        relinearize_external_product_method2_inplace_ckks(Ciphertext& input1,
                                                          Relinkey& relin_key);

        __host__ void relinearize_external_product_method2_inplace_ckks(
            Ciphertext& input1, Relinkey& relin_key, HEStream& stream);

        ///////////////////////////////////////////////////

        __host__ void rotate_method_I(Ciphertext& input1, Ciphertext& output,
                                      Galoiskey& galois_key, int shift);

        __host__ void rotate_method_I(Ciphertext& input1, Ciphertext& output,
                                      Galoiskey& galois_key, int shift,
                                      HEStream& stream);

        __host__ void rotate_method_II(Ciphertext& input1, Ciphertext& output,
                                       Galoiskey& galois_key, int shift);

        __host__ void rotate_method_II(Ciphertext& input1, Ciphertext& output,
                                       Galoiskey& galois_key, int shift,
                                       HEStream& stream);

        __host__ void rotate_ckks_method_I(Ciphertext& input1,
                                           Ciphertext& output,
                                           Galoiskey& galois_key, int shift);

        __host__ void rotate_ckks_method_I(Ciphertext& input1,
                                           Ciphertext& output,
                                           Galoiskey& galois_key, int shift,
                                           HEStream& stream);

        __host__ void rotate_ckks_method_II(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key, int shift);

        __host__ void rotate_ckks_method_II(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key, int shift,
                                            HEStream& stream);

        ///////////////////////////////////////////////////

        // TODO: Merge with rotation, provide code integrity
        __host__ void apply_galois_method_I(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key,
                                            int galois_elt);

        __host__ void apply_galois_method_I(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key,
                                            int galois_elt, HEStream& stream);

        __host__ void apply_galois_method_II(Ciphertext& input1,
                                             Ciphertext& output,
                                             Galoiskey& galois_key,
                                             int galois_elt);

        __host__ void apply_galois_method_II(Ciphertext& input1,
                                             Ciphertext& output,
                                             Galoiskey& galois_key,
                                             int galois_elt, HEStream& stream);

        __host__ void apply_galois_ckks_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Galoiskey& galois_key,
                                                 int galois_elt);

        __host__ void apply_galois_ckks_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Galoiskey& galois_key,
                                                 int galois_elt,
                                                 HEStream& stream);

        __host__ void apply_galois_ckks_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Galoiskey& galois_key,
                                                  int galois_elt);

        __host__ void apply_galois_ckks_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Galoiskey& galois_key,
                                                  int galois_elt,
                                                  HEStream& stream);

        ///////////////////////////////////////////////////

        __host__ void rotate_columns_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key);

        __host__ void rotate_columns_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key,
                                              HEStream& stream);

        __host__ void rotate_columns_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key);

        __host__ void rotate_columns_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key,
                                               HEStream& stream);

        ///////////////////////////////////////////////////

        __host__ void switchkey_method_I(Ciphertext& input1, Ciphertext& output,
                                         Switchkey& switch_key);

        __host__ void switchkey_method_I(Ciphertext& input1, Ciphertext& output,
                                         Switchkey& switch_key,
                                         HEStream& stream);

        __host__ void switchkey_method_II(Ciphertext& input1,
                                          Ciphertext& output,
                                          Switchkey& switch_key);

        __host__ void switchkey_method_II(Ciphertext& input1,
                                          Ciphertext& output,
                                          Switchkey& switch_key,
                                          HEStream& stream);

        __host__ void switchkey_ckks_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Switchkey& switch_key);

        __host__ void switchkey_ckks_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Switchkey& switch_key,
                                              HEStream& stream);

        __host__ void switchkey_ckks_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Switchkey& switch_key);

        __host__ void switchkey_ckks_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Switchkey& switch_key,
                                               HEStream& stream);

        ///////////////////////////////

        __host__ void rescale_inplace_ckks_leveled(Ciphertext& input1);

        __host__ void rescale_inplace_ckks_leveled(Ciphertext& input1,
                                                   HEStream& stream);

        ///////////////////////////////

        __host__ void mod_drop_ckks_leveled(Ciphertext& input1,
                                            Ciphertext& input2);

        __host__ void mod_drop_ckks_leveled(Ciphertext& input1,
                                            Ciphertext& input2,
                                            HEStream& stream);

        __host__ void mod_drop_ckks_plaintext(Plaintext& input1,
                                              Plaintext& input2);

        __host__ void mod_drop_ckks_plaintext(Plaintext& input1,
                                              Plaintext& input2,
                                              HEStream& stream);

        __host__ void mod_drop_ckks_plaintext_inplace(Plaintext& input1);

        __host__ void mod_drop_ckks_plaintext_inplace(Plaintext& input1,
                                                      HEStream& stream);

        __host__ void mod_drop_ckks_leveled_inplace(Ciphertext& input1);

        __host__ void mod_drop_ckks_leveled_inplace(Ciphertext& input1,
                                                    HEStream& stream);

        ///////////////////////////////

        __host__ void negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     int index);

        __host__ void negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     int index,
                                                     HEStream& stream);

        ///////////////////////////////

        __host__ void transform_to_ntt_bfv_plain(Plaintext& input1,
                                                 Plaintext& output);
        __host__ void transform_to_ntt_bfv_plain(Plaintext& input1,
                                                 Plaintext& output,
                                                 HEStream& stream);

        __host__ void transform_to_ntt_bfv_cipher(Ciphertext& input1,
                                                  Ciphertext& output);
        __host__ void transform_to_ntt_bfv_cipher(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  HEStream& stream);

        __host__ void transform_from_ntt_bfv_cipher(Ciphertext& input1,
                                                    Ciphertext& output);
        __host__ void transform_from_ntt_bfv_cipher(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    HEStream& stream);

      private:
        scheme_type scheme;

        int n;

        int n_power;

        int bsk_mod_count_;

        // New
        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus>> modulus_;
        std::shared_ptr<DeviceVector<Root>> ntt_table_;
        std::shared_ptr<DeviceVector<Root>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse>> n_inverse_;
        std::shared_ptr<DeviceVector<Data>> last_q_modinv_;

        std::shared_ptr<DeviceVector<Modulus>> base_Bsk_;
        std::shared_ptr<DeviceVector<Root>> bsk_ntt_tables_; // check
        std::shared_ptr<DeviceVector<Root>> bsk_intt_tables_; // check
        std::shared_ptr<DeviceVector<Ninverse>> bsk_n_inverse_; // check

        Modulus m_tilde_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_Bsk_;
        std::shared_ptr<DeviceVector<Data>> inv_punctured_prod_mod_base_array_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_m_tilde_;

        Data inv_prod_q_mod_m_tilde_;
        std::shared_ptr<DeviceVector<Data>> inv_m_tilde_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data>> prod_q_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data>> inv_prod_q_mod_Bsk_;

        Modulus plain_modulus_;

        std::shared_ptr<DeviceVector<Data>> base_change_matrix_q_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_msk_;

        std::shared_ptr<DeviceVector<Data>> inv_punctured_prod_mod_B_array_;
        Data inv_prod_B_mod_m_sk_;
        std::shared_ptr<DeviceVector<Data>> prod_B_mod_q_;

        std::shared_ptr<DeviceVector<Modulus>> q_Bsk_merge_modulus_;
        std::shared_ptr<DeviceVector<Root>> q_Bsk_merge_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> q_Bsk_merge_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse>> q_Bsk_n_inverse_;

        std::shared_ptr<DeviceVector<Data>> half_p_;
        std::shared_ptr<DeviceVector<Data>> half_mod_;

        Data upper_threshold_;
        std::shared_ptr<DeviceVector<Data>> upper_halfincrement_;

        Data Q_mod_t_;
        std::shared_ptr<DeviceVector<Data>> coeeff_div_plainmod_;

        /////////

        int d;
        int d_tilda;
        int r_prime;

        std::shared_ptr<DeviceVector<Modulus>> B_prime_;
        std::shared_ptr<DeviceVector<Root>> B_prime_ntt_tables_;
        std::shared_ptr<DeviceVector<Root>> B_prime_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse>> B_prime_n_inverse_;

        std::shared_ptr<DeviceVector<Data>> base_change_matrix_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_B_to_D_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_B_to_D_;
        std::shared_ptr<DeviceVector<Data>> prod_D_to_B_;
        std::shared_ptr<DeviceVector<Data>> prod_B_to_D_;

        // Method2
        std::shared_ptr<DeviceVector<Data>> base_change_matrix_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data>> prod_D_to_Q_tilda_;

        std::shared_ptr<DeviceVector<int>> I_j_;
        std::shared_ptr<DeviceVector<int>> I_location_;
        std::shared_ptr<DeviceVector<int>> Sk_pair_;

        /////////

        /////////
        // !!! LEVELED !!!

        std::shared_ptr<std::vector<int>> l_leveled_;
        std::shared_ptr<std::vector<int>> l_tilda_leveled_;
        std::shared_ptr<std::vector<int>> d_leveled_;
        std::shared_ptr<std::vector<int>> d_tilda_leveled_;
        int r_prime_leveled_;

        std::shared_ptr<DeviceVector<Modulus>> B_prime_leveled_;
        std::shared_ptr<DeviceVector<Root>> B_prime_ntt_tables_leveled_;
        std::shared_ptr<DeviceVector<Root>> B_prime_intt_tables_leveled_;
        std::shared_ptr<DeviceVector<Ninverse>> B_prime_n_inverse_leveled_;

        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> Mi_inv_D_to_B_leveled_;
        std::shared_ptr<DeviceVector<Data>> Mi_inv_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>> prod_B_to_D_leveled_;

        // Method2
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            base_change_matrix_D_to_Qtilda_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            Mi_inv_D_to_Qtilda_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data>>>
            prod_D_to_Qtilda_leveled_;

        std::shared_ptr<std::vector<DeviceVector<int>>> I_j_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> I_location_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> Sk_pair_leveled_;

        std::shared_ptr<DeviceVector<int>> prime_location_leveled_;

        /////////

        // Leveled Rescale
        std::shared_ptr<DeviceVector<Data>> rescaled_last_q_modinv_;
        std::shared_ptr<DeviceVector<Data>> rescaled_half_;
        std::shared_ptr<DeviceVector<Data>> rescaled_half_mod_;

        std::vector<Modulus> prime_vector_; // in CPU

        // Temp(to avoid allocation time)
        DeviceVector<Data> temp_mul;
        Data* temp1_mul;
        Data* temp2_mul;

        DeviceVector<Data> temp_relin;
        Data* temp1_relin;
        Data* temp2_relin;

        DeviceVector<Data> temp_relin_new;
        Data* temp1_relin_new;
        Data* temp2_relin_new;
        Data* temp3_relin_new;

        DeviceVector<Data> temp_rescale;
        Data* temp1_rescale;
        Data* temp2_rescale;

        DeviceVector<Data> temp_rotation;
        Data* temp0_rotation;
        Data* temp1_rotation;
        Data* temp2_rotation;
        Data* temp3_rotation;
        Data* temp4_rotation;

        DeviceVector<Data> temp_plain_mul;
        Data* temp1_plain_mul;

        DeviceVector<Data> temp_mod_drop_;
        Data* temp_mod_drop;

        // new method
        DeviceVector<int> new_prime_locations_;
        DeviceVector<int> new_input_locations_;
        int* new_prime_locations;
        int* new_input_locations;
    };

} // namespace heongpu

#endif // OPERATOR_H
