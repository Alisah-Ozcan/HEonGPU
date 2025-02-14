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
#include "encoder.cuh"
#include "bootstrapping.cuh"

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
      protected:
        /**
         * @brief Construct a new HEOperator object with the given parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters for the operator.
         */
        __host__ HEOperator(Parameters& context, HEEncoder& encoder);

      public:
        /**
         * @brief Adds two ciphertexts and stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Adds the second ciphertext to the first ciphertext, modifying
         * the first ciphertext with the result.
         *
         * @param input1 The ciphertext to which the value of input2 will be
         * added.
         * @param input2 The ciphertext to be added to input1.
         */
        __host__ void
        add_inplace(Ciphertext& input1, Ciphertext& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            add(input1, input2, input1, options);
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
                          Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Subtracts the second ciphertext from the first, modifying the
         * first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         */
        __host__ void
        sub_inplace(Ciphertext& input1, Ciphertext& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            sub(input1, input2, input1, options);
        }

        /**
         * @brief Negates a ciphertext and stores the result in the output.
         *
         * @param input1 Input ciphertext to be negated.
         * @param output Ciphertext where the result of the negation is stored.
         */
        __host__ void
        negate(Ciphertext& input1, Ciphertext& output,
               const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Negates a ciphertext in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         */
        __host__ void
        negate_inplace(Ciphertext& input1,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            negate(input1, input1, options);
        }

        /**
         * @brief Adds a ciphertext and a plaintext and stores the result in the
         * output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input plaintext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void
        add_plain(Ciphertext& input1, Plaintext& input2, Ciphertext& output,
                  const ExecutionOptions& options = ExecutionOptions())
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

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                        {
                                            if (input1_.in_ntt_domain_ ||
                                                input2_.in_ntt_domain_)
                                            {
                                                throw std::logic_error(
                                                    "BFV ciphertext or "
                                                    "plaintext "
                                                    "should be not in NTT "
                                                    "domain");
                                            }
                                            add_plain_bfv(input1_, input2_,
                                                          output_,
                                                          options.stream_);
                                            break;
                                        }
                                        case 2: // CKKS
                                        {
                                            add_plain_ckks(input1_, input2_,
                                                           output_,
                                                           options.stream_);
                                            break;
                                        }
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.scale_ = input1_.scale_;
                                    output_.rescale_required_ =
                                        input1_.rescale_required_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Adds a plaintext to a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Plaintext to be added to the ciphertext.
         */
        __host__ void
        add_plain_inplace(Ciphertext& input1, Plaintext& input2,
                          const ExecutionOptions& options = ExecutionOptions())
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

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            switch (static_cast<int>(scheme_))
                            {
                                case 1: // BFV
                                {
                                    if (input1.in_ntt_domain_ ||
                                        input2.in_ntt_domain_)
                                    {
                                        throw std::logic_error(
                                            "BFV ciphertext or plaintext "
                                            "should be not in NTT domain");
                                    }
                                    add_plain_bfv_inplace(input1_, input2_,
                                                          options.stream_);
                                    break;
                                }
                                case 2: // CKKS
                                    add_plain_ckks_inplace(input1_, input2_,
                                                           options.stream_);
                                    break;
                                case 3: // BGV
                                    throw std::invalid_argument(
                                        "Invalid Scheme Type");
                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Scheme Type");
                                    break;
                            }
                        },
                        options, false);
                },
                options, true);
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
        __host__ void
        sub_plain(Ciphertext& input1, Plaintext& input2, Ciphertext& output,
                  const ExecutionOptions& options = ExecutionOptions())
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

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                        {
                                            if (input1_.in_ntt_domain_ ||
                                                input2_.in_ntt_domain_)
                                            {
                                                throw std::logic_error(
                                                    "BFV ciphertext or "
                                                    "plaintext "
                                                    "should be not in NTT "
                                                    "domain");
                                            }
                                            sub_plain_bfv(input1_, input2_,
                                                          output_,
                                                          options.stream_);
                                            break;
                                        }
                                        case 2: // CKKS
                                        {
                                            sub_plain_ckks(input1_, input2_,
                                                           output_,
                                                           options.stream_);
                                            break;
                                        }
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.scale_ = input1_.scale_;
                                    output_.rescale_required_ =
                                        input1_.rescale_required_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Subtracts a plaintext from a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext from which the plaintext will be subtracted.
         * @param input2 Plaintext to be subtracted from the ciphertext.
         */
        __host__ void
        sub_plain_inplace(Ciphertext& input1, Plaintext& input2,
                          const ExecutionOptions& options = ExecutionOptions())
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

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            switch (static_cast<int>(scheme_))
                            {
                                case 1: // BFV
                                {
                                    if (input1.in_ntt_domain_ ||
                                        input2.in_ntt_domain_)
                                    {
                                        throw std::logic_error(
                                            "BFV ciphertext or plaintext "
                                            "should be not in NTT domain");
                                    }
                                    sub_plain_bfv_inplace(input1_, input2_,
                                                          options.stream_);
                                    break;
                                }
                                case 2: // CKKS
                                    sub_plain_ckks_inplace(input1_, input2_,
                                                           options.stream_);
                                    break;
                                case 3: // BGV
                                    throw std::invalid_argument(
                                        "Invalid Scheme Type");
                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Scheme Type");
                                    break;
                            }
                        },
                        options, false);
                },
                options, true);
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
        __host__ void
        multiply(Ciphertext& input1, Ciphertext& input2, Ciphertext& output,
                 const ExecutionOptions& options = ExecutionOptions())
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

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_bfv(input1_, input2_,
                                                         output_,
                                                         options.stream_);
                                            output_.scale_ = 0;
                                            break;
                                        case 2: // CKKS
                                            multiply_ckks(input1_, input2_,
                                                          output_,
                                                          options.stream_);
                                            output_.rescale_required_ = true;
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 3;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.relinearization_required_ = true;
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies two ciphertexts in-place, modifying the first
         * ciphertext.
         *
         * @param input1 Ciphertext to be multiplied, and where the result will
         * be stored.
         * @param input2 Second input ciphertext to be multiplied.
         */
        __host__ void
        multiply_inplace(Ciphertext& input1, Ciphertext& input2,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            multiply(input1, input2, input1, options);
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
        __host__ void
        multiply_plain(Ciphertext& input1, Plaintext& input2,
                       Ciphertext& output,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be multiplied because of "
                    "the non-linear part! Please use relinearization "
                    "operation!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                        {
                                            if (input1_.in_ntt_domain_ !=
                                                input2_.in_ntt_domain_)
                                            {
                                                throw std::logic_error(
                                                    "BFV ciphertext or "
                                                    "plaintext "
                                                    "should be not in same "
                                                    "domain");
                                            }

                                            if (input2_.size() < n)
                                            {
                                                throw std::invalid_argument(
                                                    "Invalid Plaintext size!");
                                            }

                                            multiply_plain_bfv(input1_, input2_,
                                                               output_,
                                                               options.stream_);
                                            output_.rescale_required_ =
                                                input1_.rescale_required_;
                                            break;
                                        }
                                        case 2: // CKKS
                                            if (input2_.size() <
                                                (n * current_decomp_count))
                                            {
                                                throw std::invalid_argument(
                                                    "Invalid Plaintext size!");
                                            }

                                            multiply_plain_ckks(
                                                input1_, input2_, output_,
                                                options.stream_);
                                            output_.rescale_required_ = true;
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Plaintext to be multiplied with the ciphertext.
         */
        __host__ void multiply_plain_inplace(
            Ciphertext& input1, Plaintext& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            multiply_plain(input1, input2, input1, options);
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         */
        __host__ void relinearize_inplace(
            Ciphertext& input1, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions())
        {
            if ((!input1.relinearization_required_))
            {
                throw std::invalid_argument(
                    "Ciphertexts can not use relinearization, since no "
                    "non-linear part!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (3 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    switch (static_cast<int>(relin_key.key_type))
                    {
                        case 1: // KEYSWITCHING_METHOD_I
                            if (scheme_ == scheme_type::bfv)
                            {
                                if (input1_.in_ntt_domain_ != false)
                                {
                                    throw std::invalid_argument(
                                        "Ciphertext should be in intt domain");
                                }

                                relinearize_seal_method_inplace(
                                    input1_, relin_key, options.stream_);
                            }
                            else if (scheme_ == scheme_type::ckks)
                            {
                                relinearize_seal_method_inplace_ckks(
                                    input1_, relin_key, options.stream_);
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    "Invalid Key Switching Type");
                            }
                            break;
                        case 2: // KEYSWITCHING_METHOD_II

                            if (scheme_ == scheme_type::bfv)
                            {
                                if (input1_.in_ntt_domain_ != false)
                                {
                                    throw std::invalid_argument(
                                        "Ciphertext should be in intt domain");
                                }

                                relinearize_external_product_method2_inplace(
                                    input1_, relin_key, options.stream_);
                            }
                            else if (scheme_ == scheme_type::ckks)
                            {
                                relinearize_external_product_method2_inplace_ckks(
                                    input1_, relin_key, options.stream_);
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    "Invalid Key Switching Type");
                            }

                            break;
                        case 3: // KEYSWITCHING_METHOD_III

                            if (scheme_ == scheme_type::bfv)
                            {
                                if (input1_.in_ntt_domain_ != false)
                                {
                                    throw std::invalid_argument(
                                        "Ciphertext should be in intt domain");
                                }

                                relinearize_external_product_method_inplace(
                                    input1_, relin_key, options.stream_);
                            }
                            else if (scheme_ == scheme_type::ckks)
                            {
                                relinearize_external_product_method_inplace_ckks(
                                    input1_, relin_key, options.stream_);
                            }
                            else
                            {
                                throw std::invalid_argument(
                                    "Invalid Key Switching Type");
                            }

                            break;
                        default:
                            throw std::invalid_argument(
                                "Invalid Key Switching Type");
                            break;
                    }

                    input1_.relinearization_required_ = false;
                },
                options, true);
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
        __host__ void
        rotate_rows(Ciphertext& input1, Ciphertext& output,
                    Galoiskey& galois_key, int shift,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            if (shift == 0)
            {
                output = input1;
                return;
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        rotate_method_I(input1_, output_,
                                                        galois_key, shift,
                                                        options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        rotate_ckks_method_I(input1_, output_,
                                                             galois_key, shift,
                                                             options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        rotate_method_II(input1_, output_,
                                                         galois_key, shift,
                                                         options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        rotate_ckks_method_II(input1_, output_,
                                                              galois_key, shift,
                                                              options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 3: // KEYSWITCHING_METHOD_III

                                    throw std::invalid_argument(
                                        "KEYSWITCHING_METHOD_III are not "
                                        "supported because of "
                                        "high memory consumption for rotation "
                                        "operation!");

                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Key Switching Type");
                                    break;
                            }

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Rotates the rows of a ciphertext in-place by a given shift
         * value, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be rotated.
         * @param galois_key Galois key used for the rotation operation.
         * @param shift Number of positions to shift the rows.
         */
        __host__ void rotate_rows_inplace(
            Ciphertext& input1, Galoiskey& galois_key, int shift,
            const ExecutionOptions& options = ExecutionOptions())
        {
            if (shift == 0)
            {
                return;
            }

            rotate_rows(input1, input1, galois_key, shift, options);
        }

        /**
         * @brief Rotates the columns of a ciphertext and stores the result in
         * the output.
         *
         * @param input1 Input ciphertext to be rotated.
         * @param output Ciphertext where the result of the rotation is stored.
         * @param galois_key Galois key used for the rotation operation.
         */
        __host__ void
        rotate_columns(Ciphertext& input1, Ciphertext& output,
                       Galoiskey& galois_key,
                       const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        rotate_columns_method_I(
                                            input1_, output_, galois_key,
                                            options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        throw std::invalid_argument(
                                            "Unsupported scheme");
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        rotate_columns_method_II(
                                            input1_, output_, galois_key,
                                            options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        throw std::invalid_argument(
                                            "Unsupported scheme");
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 3: // KEYSWITCHING_METHOD_III

                                    throw std::invalid_argument(
                                        "KEYSWITCHING_METHOD_III are not "
                                        "supported because of "
                                        "high memory consumption for rotation "
                                        "operation!");

                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Key Switching Type");
                                    break;
                            }

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                        },
                        options);
                },
                options, (&input1 == &output));
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
        __host__ void
        apply_galois(Ciphertext& input1, Ciphertext& output,
                     Galoiskey& galois_key, int galois_elt,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        apply_galois_method_I(
                                            input1_, output_, galois_key,
                                            galois_elt, options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        apply_galois_ckks_method_I(
                                            input1_, output_, galois_key,
                                            galois_elt, options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        apply_galois_method_II(
                                            input1_, output_, galois_key,
                                            galois_elt, options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        apply_galois_ckks_method_II(
                                            input1_, output_, galois_key,
                                            galois_elt, options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 3: // KEYSWITCHING_METHOD_III

                                    throw std::invalid_argument(
                                        "KEYSWITCHING_METHOD_III are not "
                                        "supported because of "
                                        "high memory consumption for rotation "
                                        "operation!");

                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Key Switching Type");
                                    break;
                            }

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                        },
                        options);
                },
                options, (&input1 == &output));
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
        __host__ void apply_galois_inplace(
            Ciphertext& input1, Galoiskey& galois_key, int galois_elt,
            const ExecutionOptions& options = ExecutionOptions())
        {
            apply_galois(input1, input1, galois_key, galois_elt, options);
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
        __host__ void
        keyswitch(Ciphertext& input1, Ciphertext& output, Switchkey& switch_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        {
                            switch (static_cast<int>(switch_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        switchkey_method_I(input1_, output_,
                                                           switch_key,
                                                           options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        switchkey_ckks_method_I(
                                            input1_, output_, switch_key,
                                            options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        if (input1_.in_ntt_domain_ != false)
                                        {
                                            throw std::invalid_argument(
                                                "Ciphertext should be in intt "
                                                "domain");
                                        }

                                        switchkey_method_II(input1_, output_,
                                                            switch_key,
                                                            options.stream_);
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        switchkey_ckks_method_II(
                                            input1_, output_, switch_key,
                                            options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 3: // KEYSWITCHING_METHOD_III

                                    throw std::invalid_argument(
                                        "KEYSWITCHING_METHOD_III are not "
                                        "supported because of "
                                        "high memory consumption for keyswitch "
                                        "operation!");

                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Key Switching Type");
                                    break;
                            }

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs conjugation on the ciphertext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be conjugated.
         * @param output Ciphertext where the result of the conjugation is
         * stored.
         * @param conjugate_key Switch key used for the conjugation operation.
         */
        __host__ void
        conjugate(Ciphertext& input1, Ciphertext& output,
                  Galoiskey& conjugate_key,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        {
                            switch (static_cast<int>(conjugate_key.key_type))
                            {
                                case 1: // KEYSWITHING_METHOD_I
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        throw std::invalid_argument(
                                            "BFV Does Not Support!");
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        conjugate_ckks_method_I(
                                            input1_, output_, conjugate_key,
                                            options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 2: // KEYSWITHING_METHOD_II
                                    if (scheme_ == scheme_type::bfv)
                                    {
                                        throw std::invalid_argument(
                                            "BFV Does Not Support!");
                                    }
                                    else if (scheme_ == scheme_type::ckks)
                                    {
                                        conjugate_ckks_method_II(
                                            input1_, output_, conjugate_key,
                                            options.stream_);
                                    }
                                    else
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Key Switching Type");
                                    }
                                    break;
                                case 3: // KEYSWITHING_METHOD_III

                                    throw std::invalid_argument(
                                        "KEYSWITHING_METHOD_III are not "
                                        "supported because of "
                                        "high memory consumption for keyswitch "
                                        "operation!");

                                    break;
                                default:
                                    throw std::invalid_argument(
                                        "Invalid Key Switching Type");
                                    break;
                            }

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.scale_ = input1_.scale_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Rescales a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to be rescaled.
         */
        __host__ void
        rescale_inplace(Ciphertext& input1,
                        const ExecutionOptions& options = ExecutionOptions())
        {
            if ((!input1.rescale_required_) || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertexts can not be rescaled!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument("BFV Does Not Support!");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        input1,
                        [&](Ciphertext& input1_) {
                            rescale_inplace_ckks_leveled(input1_,
                                                         options.stream_);
                        },
                        options, true);
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
        __host__ void
        mod_drop(Ciphertext& input1, Ciphertext& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext's modulus can not be dropped!!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        input1,
                        [&](Ciphertext& input1_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    mod_drop_ckks_leveled(input1_, output_,
                                                          options.stream_);

                                    output.scheme_ = scheme_;
                                    output.ring_size_ = n;
                                    output.coeff_modulus_count_ = Q_size_;
                                    output.cipher_size_ = 2;
                                    output.depth_ = input1.depth_ + 1;
                                    output.scale_ = input1.scale_;
                                    output.in_ntt_domain_ =
                                        input1.in_ntt_domain_;
                                    output.rescale_required_ =
                                        input1.rescale_required_;
                                    output.relinearization_required_ =
                                        input1.relinearization_required_;
                                },
                                options);
                        },
                        options, (&input1 == &output));
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        /**
         * @brief Drop the last modulus of plaintext and stores the result in
         * the output.(CKKS)
         *
         * @param input1 Input plaintext from which modulus will be dropped.
         * @param output Plaintext where the result of the modulus drop is
         * stored.
         */
        __host__ void
        mod_drop(Plaintext& input1, Plaintext& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.size() < (n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Plaintext size!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        input1,
                        [&](Plaintext& input1_)
                        {
                            output_storage_manager(
                                output,
                                [&](Plaintext& output_)
                                {
                                    mod_drop_ckks_plaintext(input1_, output_,
                                                            options.stream_);

                                    output.scheme_ = input1.scheme_;
                                    output.plain_size_ =
                                        (n * (current_decomp_count - 1));
                                    output.depth_ = input1.depth_ + 1;
                                    output.scale_ = input1.scale_;
                                    output.in_ntt_domain_ =
                                        input1.in_ntt_domain_;
                                },
                                options);
                        },
                        options, (&input1 == &output));
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
         */
        __host__ void
        mod_drop_inplace(Plaintext& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.size() < (n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Plaintext size!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        input1,
                        [&](Plaintext& input1_) {
                            mod_drop_ckks_plaintext_inplace(input1_,
                                                            options.stream_);
                        },
                        options, true);
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
        __host__ void
        mod_drop_inplace(Ciphertext& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    // TODO: implement leveled BFV.
                    throw std::invalid_argument(
                        "BFV does dot support modulus dropping!");
                    break;
                case 2: // CKKS
                    input_storage_manager(
                        input1,
                        [&](Ciphertext& input1_) {
                            mod_drop_ckks_leveled_inplace(input1_,
                                                          options.stream_);
                        },
                        options, true);
                    break;
                default:
                    throw std::invalid_argument("Invalid Scheme Type");
                    break;
            }
        }

        __host__ void multiply_power_of_X(
            Ciphertext& input1, Ciphertext& output, int index,
            const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (index != 0)
                    {
                        if (input1.in_ntt_domain_ != false)
                        {
                            throw std::invalid_argument(
                                "Ciphertext should be in intt domain");
                        }

                        if (input1.memory_size() < (2 * n * Q_size_))
                        {
                            throw std::invalid_argument(
                                "Invalid Ciphertexts size!");
                        }

                        input_storage_manager(
                            input1,
                            [&](Ciphertext& input1_)
                            {
                                output_storage_manager(
                                    output,
                                    [&](Ciphertext& output_)
                                    {
                                        negacyclic_shift_poly_coeffmod(
                                            input1_, output_, index,
                                            options.stream_);

                                        output.scheme_ = scheme_;
                                        output.ring_size_ = n;
                                        output.coeff_modulus_count_ = Q_size_;
                                        output.cipher_size_ = 2;
                                        output.depth_ = input1.depth_;
                                        output.scale_ = input1.scale_;
                                        output.in_ntt_domain_ =
                                            input1.in_ntt_domain_;
                                        output.rescale_required_ =
                                            input1.rescale_required_;
                                        output.relinearization_required_ =
                                            input1.relinearization_required_;
                                    },
                                    options);
                            },
                            options, (&input1 == &output));
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
        }

        /**
         * @brief Transforms a plaintext to the NTT domain and stores the result
         * in the output.
         *
         * @param input1 Input plaintext to be transformed.
         * @param output Plaintext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_to_ntt(Plaintext& input1, Plaintext& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        if (input1.size() < n)
                        {
                            throw std::invalid_argument(
                                "Invalid Ciphertexts size!");
                        }

                        input_storage_manager(
                            input1,
                            [&](Plaintext& input1_)
                            {
                                output_storage_manager(
                                    output,
                                    [&](Plaintext& output_)
                                    {
                                        transform_to_ntt_bfv_plain(
                                            input1, output, options.stream_);

                                        output.scheme_ = input1.scheme_;
                                        output.plain_size_ = (n * Q_size_);
                                        output.depth_ = input1.depth_;
                                        output.scale_ = input1.scale_;
                                        output.in_ntt_domain_ = true;
                                    },
                                    options);
                            },
                            options, (&input1 == &output));
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
        __host__ void transform_to_ntt_inplace(
            Plaintext& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            transform_to_ntt(input1, input1, options);
        }

        /**
         * @brief Transforms a ciphertext to the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_to_ntt(Ciphertext& input1, Ciphertext& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed to NTT!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (!input1.in_ntt_domain_)
                    {
                        if (input1.memory_size() < (2 * n * Q_size_))
                        {
                            throw std::invalid_argument(
                                "Invalid Ciphertexts size!");
                        }

                        input_storage_manager(
                            input1,
                            [&](Ciphertext& input1_)
                            {
                                output_storage_manager(
                                    output,
                                    [&](Ciphertext& output_)
                                    {
                                        transform_to_ntt_bfv_cipher(
                                            input1_, output_, options.stream_);

                                        output.scheme_ = scheme_;
                                        output.ring_size_ = n;
                                        output.coeff_modulus_count_ = Q_size_;
                                        output.cipher_size_ = 2;
                                        output.depth_ = input1.depth_;
                                        output.scale_ = input1.scale_;
                                        output.in_ntt_domain_ = true;
                                        output.rescale_required_ =
                                            input1.rescale_required_;
                                        output.relinearization_required_ =
                                            input1.relinearization_required_;
                                    },
                                    options);
                            },
                            options, (&input1 == &output));
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
         * @brief Transforms a ciphertext to the NTT domain in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed.
         */
        __host__ void transform_to_ntt_inplace(
            Ciphertext& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            transform_to_ntt(input1, input1, options);
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain and stores the
         * result in the output.
         *
         * @param input1 Input ciphertext to be transformed from the NTT domain.
         * @param output Ciphertext where the result of the transformation is
         * stored.
         */
        __host__ void
        transform_from_ntt(Ciphertext& input1, Ciphertext& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertexts can not be transformed from NTT!");
            }

            switch (static_cast<int>(scheme_))
            {
                case 1: // BFV
                    if (input1.in_ntt_domain_)
                    {
                        if (input1.memory_size() < (2 * n * Q_size_))
                        {
                            throw std::invalid_argument(
                                "Invalid Ciphertexts size!");
                        }

                        input_storage_manager(
                            input1,
                            [&](Ciphertext& input1_)
                            {
                                output_storage_manager(
                                    output,
                                    [&](Ciphertext& output_)
                                    {
                                        transform_from_ntt_bfv_cipher(
                                            input1_, output_, options.stream_);

                                        output.scheme_ = scheme_;
                                        output.ring_size_ = n;
                                        output.coeff_modulus_count_ = Q_size_;
                                        output.cipher_size_ = 2;
                                        output.depth_ = input1.depth_;
                                        output.scale_ = input1.scale_;
                                        output.in_ntt_domain_ = false;
                                        output.rescale_required_ =
                                            input1.rescale_required_;
                                        output.relinearization_required_ =
                                            input1.relinearization_required_;
                                    },
                                    options);
                            },
                            options, (&input1 == &output));
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
        }

        /**
         * @brief Transforms a ciphertext from the NTT domain in-place,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be transformed from the NTT domain.
         */
        __host__ void transform_from_ntt_inplace(
            Ciphertext& input1,
            const ExecutionOptions& options = ExecutionOptions())
        {
            transform_from_ntt(input1, input1, options);
        }

        ////////////////////////////////////////
        // BGV (Coming Soon)

        ////////////////////////////////////////

        HEOperator() = default;
        HEOperator(const HEOperator& copy) = default;
        HEOperator(HEOperator&& source) = default;
        HEOperator& operator=(const HEOperator& assign) = default;
        HEOperator& operator=(HEOperator&& assign) = default;

        // private:
      protected:
        __host__ void add_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output,
                                    const cudaStream_t stream);

        __host__ void add_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2,
                                            const cudaStream_t stream);

        __host__ void add_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output,
                                     const cudaStream_t stream);

        __host__ void add_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2,
                                             const cudaStream_t stream);

        __host__ void sub_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                    Ciphertext& output,
                                    const cudaStream_t stream);

        __host__ void sub_plain_bfv_inplace(Ciphertext& input1,
                                            Plaintext& input2,
                                            const cudaStream_t stream);

        __host__ void sub_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                     Ciphertext& output,
                                     const cudaStream_t stream);

        __host__ void sub_plain_ckks_inplace(Ciphertext& input1,
                                             Plaintext& input2,
                                             const cudaStream_t stream);

        __host__ void multiply_bfv(Ciphertext& input1, Ciphertext& input2,
                                   Ciphertext& output,
                                   const cudaStream_t stream);

        __host__ void multiply_ckks(Ciphertext& input1, Ciphertext& input2,
                                    Ciphertext& output,
                                    const cudaStream_t stream);

        __host__ void multiply_plain_bfv(Ciphertext& input1, Plaintext& input2,
                                         Ciphertext& output,
                                         const cudaStream_t stream);

        __host__ void multiply_plain_ckks(Ciphertext& input1, Plaintext& input2,
                                          Ciphertext& output,
                                          const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        relinearize_seal_method_inplace(Ciphertext& input1, Relinkey& relin_key,
                                        const cudaStream_t stream);

        __host__ void relinearize_external_product_method_inplace(
            Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream);

        __host__ void relinearize_external_product_method2_inplace(
            Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream);

        __host__ void relinearize_seal_method_inplace_ckks(
            Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream);

        __host__ void relinearize_external_product_method_inplace_ckks(
            Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream);

        __host__ void relinearize_external_product_method2_inplace_ckks(
            Ciphertext& input1, Relinkey& relin_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void rotate_method_I(Ciphertext& input1, Ciphertext& output,
                                      Galoiskey& galois_key, int shift,
                                      const cudaStream_t stream);

        __host__ void rotate_method_II(Ciphertext& input1, Ciphertext& output,
                                       Galoiskey& galois_key, int shift,
                                       const cudaStream_t stream);

        __host__ void rotate_ckks_method_I(Ciphertext& input1,
                                           Ciphertext& output,
                                           Galoiskey& galois_key, int shift,
                                           const cudaStream_t stream);

        __host__ void rotate_ckks_method_II(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key, int shift,
                                            const cudaStream_t stream);

        ///////////////////////////////////////////////////

        // TODO: Merge with rotation, provide code integrity
        __host__ void apply_galois_method_I(Ciphertext& input1,
                                            Ciphertext& output,
                                            Galoiskey& galois_key,
                                            int galois_elt,
                                            const cudaStream_t stream);

        __host__ void apply_galois_method_II(Ciphertext& input1,
                                             Ciphertext& output,
                                             Galoiskey& galois_key,
                                             int galois_elt,
                                             const cudaStream_t stream);

        __host__ void apply_galois_ckks_method_I(Ciphertext& input1,
                                                 Ciphertext& output,
                                                 Galoiskey& galois_key,
                                                 int galois_elt,
                                                 const cudaStream_t stream);

        __host__ void apply_galois_ckks_method_II(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  Galoiskey& galois_key,
                                                  int galois_elt,
                                                  const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void rotate_columns_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& galois_key,
                                              const cudaStream_t stream);

        __host__ void rotate_columns_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& galois_key,
                                               const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void switchkey_method_I(Ciphertext& input1, Ciphertext& output,
                                         Switchkey& switch_key,
                                         const cudaStream_t stream);

        __host__ void switchkey_method_II(Ciphertext& input1,
                                          Ciphertext& output,
                                          Switchkey& switch_key,
                                          const cudaStream_t stream);

        __host__ void switchkey_ckks_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Switchkey& switch_key,
                                              const cudaStream_t stream);

        __host__ void switchkey_ckks_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Switchkey& switch_key,
                                               const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void conjugate_ckks_method_I(Ciphertext& input1,
                                              Ciphertext& output,
                                              Galoiskey& conjugate_key,
                                              const cudaStream_t stream);

        __host__ void conjugate_ckks_method_II(Ciphertext& input1,
                                               Ciphertext& output,
                                               Galoiskey& conjugate_key,
                                               const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void rescale_inplace_ckks_leveled(Ciphertext& input1,
                                                   const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void mod_drop_ckks_leveled(Ciphertext& input1,
                                            Ciphertext& output,
                                            const cudaStream_t stream);

        __host__ void mod_drop_ckks_plaintext(Plaintext& input1,
                                              Plaintext& output,
                                              const cudaStream_t stream);

        __host__ void
        mod_drop_ckks_plaintext_inplace(Plaintext& input1,
                                        const cudaStream_t stream);

        __host__ void mod_drop_ckks_leveled_inplace(Ciphertext& input1,
                                                    const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void negacyclic_shift_poly_coeffmod(Ciphertext& input1,
                                                     Ciphertext& output,
                                                     int index,
                                                     const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void transform_to_ntt_bfv_plain(Plaintext& input1,
                                                 Plaintext& output,
                                                 const cudaStream_t stream);

        __host__ void transform_to_ntt_bfv_cipher(Ciphertext& input1,
                                                  Ciphertext& output,
                                                  const cudaStream_t stream);

        __host__ void transform_from_ntt_bfv_cipher(Ciphertext& input1,
                                                    Ciphertext& output,
                                                    const cudaStream_t stream);

        // private:
      protected:
        scheme_type scheme_;

        int n;

        int n_power;

        int bsk_mod_count_;

        // New
        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;

        std::shared_ptr<DeviceVector<Modulus64>> base_Bsk_;
        std::shared_ptr<DeviceVector<Root64>> bsk_ntt_tables_; // check
        std::shared_ptr<DeviceVector<Root64>> bsk_intt_tables_; // check
        std::shared_ptr<DeviceVector<Ninverse64>> bsk_n_inverse_; // check

        Modulus64 m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_Bsk_;
        std::shared_ptr<DeviceVector<Data64>>
            inv_punctured_prod_mod_base_array_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_m_tilde_;

        Data64 inv_prod_q_mod_m_tilde_;
        std::shared_ptr<DeviceVector<Data64>> inv_m_tilde_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> prod_q_mod_Bsk_;
        std::shared_ptr<DeviceVector<Data64>> inv_prod_q_mod_Bsk_;

        Modulus64 plain_modulus_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_q_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_msk_;

        std::shared_ptr<DeviceVector<Data64>> inv_punctured_prod_mod_B_array_;
        Data64 inv_prod_B_mod_m_sk_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_mod_q_;

        std::shared_ptr<DeviceVector<Modulus64>> q_Bsk_merge_modulus_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> q_Bsk_merge_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> q_Bsk_n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> half_p_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;

        Data64 upper_threshold_;
        std::shared_ptr<DeviceVector<Data64>> upper_halfincrement_;

        Data64 Q_mod_t_;
        std::shared_ptr<DeviceVector<Data64>> coeeff_div_plainmod_;

        /////////

        int d;
        int d_tilda;
        int r_prime;

        std::shared_ptr<DeviceVector<Modulus64>> B_prime_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_ntt_tables_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_intt_tables_;
        std::shared_ptr<DeviceVector<Ninverse64>> B_prime_n_inverse_;

        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_B_to_D_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_B_;
        std::shared_ptr<DeviceVector<Data64>> prod_B_to_D_;

        // Method2
        std::shared_ptr<DeviceVector<Data64>> base_change_matrix_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_D_to_Q_tilda_;
        std::shared_ptr<DeviceVector<Data64>> prod_D_to_Q_tilda_;

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

        std::shared_ptr<DeviceVector<Modulus64>> B_prime_leveled_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_ntt_tables_leveled_;
        std::shared_ptr<DeviceVector<Root64>> B_prime_intt_tables_leveled_;
        std::shared_ptr<DeviceVector<Ninverse64>> B_prime_n_inverse_leveled_;

        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            Mi_inv_D_to_B_leveled_;
        std::shared_ptr<DeviceVector<Data64>> Mi_inv_B_to_D_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>> prod_D_to_B_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>> prod_B_to_D_leveled_;

        // Method2
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            base_change_matrix_D_to_Qtilda_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            Mi_inv_D_to_Qtilda_leveled_;
        std::shared_ptr<std::vector<DeviceVector<Data64>>>
            prod_D_to_Qtilda_leveled_;

        std::shared_ptr<std::vector<DeviceVector<int>>> I_j_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> I_location_leveled_;
        std::shared_ptr<std::vector<DeviceVector<int>>> Sk_pair_leveled_;

        std::shared_ptr<DeviceVector<int>> prime_location_leveled_;

        /////////

        // Leveled Rescale
        std::shared_ptr<DeviceVector<Data64>> rescaled_last_q_modinv_;
        std::shared_ptr<DeviceVector<Data64>> rescaled_half_;
        std::shared_ptr<DeviceVector<Data64>> rescaled_half_mod_;

        std::vector<Modulus64> prime_vector_; // in CPU

        // Temp(to avoid allocation time)

        // new method
        DeviceVector<int> new_prime_locations_;
        DeviceVector<int> new_input_locations_;
        int* new_prime_locations;
        int* new_input_locations;

        // private:
      protected:
        __host__ Plaintext
        operator_plaintext(cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Plaintext operator_from_plaintext(
            Plaintext& input, cudaStream_t stream = cudaStreamDefault);

        __host__ Ciphertext operator_ciphertext(
            double scale, cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Ciphertext operator_from_ciphertext(
            Ciphertext& input, cudaStream_t stream = cudaStreamDefault);

        class Vandermonde
        {
            friend class HEOperator;
            friend class HEArithmeticOperator;
            friend class HELogicOperator;

          public:
            __host__ Vandermonde(const int poly_degree, const int CtoS_piece,
                                 const int StoC_piece,
                                 const bool less_key_mode);

            __host__ void generate_E_diagonals_index();

            __host__ void generate_E_inv_diagonals_index();

            __host__ void split_E();

            __host__ void split_E_inv();

            __host__ void generate_E_diagonals();

            __host__ void generate_E_inv_diagonals();

            __host__ void generate_V_n_lists();

            __host__ void generate_pre_comp_V();

            __host__ void generate_pre_comp_V_inv();

            __host__ void generate_key_indexs(const bool less_key_mode);

            Vandermonde() = delete;

          private:
            int poly_degree_;
            int num_slots_;
            int log_num_slots_;

            int CtoS_piece_;
            int StoC_piece_;

            std::vector<int> E_size_;
            std::vector<int> E_inv_size_;

            std::vector<int> E_index_;
            std::vector<int> E_inv_index_;

            std::vector<int> E_splitted_;
            std::vector<int> E_inv_splitted_;

            std::vector<std::vector<int>> E_splitted_index_;
            std::vector<std::vector<int>> E_inv_splitted_index_;

            std::vector<std::vector<int>> E_splitted_diag_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_diag_index_gpu_;

            std::vector<std::vector<int>> E_splitted_input_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_input_index_gpu_;

            std::vector<std::vector<int>> E_splitted_output_index_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_output_index_gpu_;

            std::vector<std::vector<int>> E_splitted_iteration_gpu_;
            std::vector<std::vector<int>> E_inv_splitted_iteration_gpu_;

            std::vector<std::vector<int>> V_matrixs_index_;
            std::vector<std::vector<int>> V_inv_matrixs_index_;

            std::vector<heongpu::DeviceVector<Complex64>> V_matrixs_;
            std::vector<heongpu::DeviceVector<Complex64>> V_inv_matrixs_;

            std::vector<heongpu::DeviceVector<Complex64>> V_matrixs_rotated_;
            std::vector<heongpu::DeviceVector<Complex64>>
                V_inv_matrixs_rotated_;

            std::vector<std::vector<std::vector<int>>> diags_matrices_bsgs_;
            std::vector<std::vector<std::vector<int>>> diags_matrices_inv_bsgs_;

            std::vector<std::vector<std::vector<int>>> real_shift_n2_bsgs_;
            std::vector<std::vector<std::vector<int>>> real_shift_n2_inv_bsgs_;

            std::vector<int> key_indexs_;
        };

        double scale_boot_;
        bool boot_context_generated_ = false;

        int CtoS_piece_;
        int StoC_piece_;
        int taylor_number_;
        bool less_key_mode_;

        std::vector<int> key_indexs_;

        std::vector<heongpu::DeviceVector<Data64>> V_matrixs_rotated_encoded_;
        std::vector<heongpu::DeviceVector<Data64>>
            V_inv_matrixs_rotated_encoded_;

        std::vector<std::vector<int>> V_matrixs_index_;
        std::vector<std::vector<int>> V_inv_matrixs_index_;

        std::vector<std::vector<std::vector<int>>> diags_matrices_bsgs_;
        std::vector<std::vector<std::vector<int>>> diags_matrices_inv_bsgs_;

        std::vector<std::vector<std::vector<int>>> real_shift_n2_bsgs_;
        std::vector<std::vector<std::vector<int>>> real_shift_n2_inv_bsgs_;

        ///////// Operator Class Encode Fuctions //////////

        int slot_count_;
        int log_slot_count_;

        // BFV
        std::shared_ptr<DeviceVector<Modulus64>>
            plain_modulus_pointer_; // we already have it
        std::shared_ptr<DeviceVector<Ninverse64>> n_plain_inverse_;
        std::shared_ptr<DeviceVector<Root64>> plain_intt_tables_;
        std::shared_ptr<DeviceVector<Data64>> encoding_location_;

        // CKKS
        double two_pow_64_;
        std::shared_ptr<DeviceVector<int>> reverse_order_;
        std::shared_ptr<DeviceVector<Complex64>> special_ifft_roots_table_;

        __host__ void
        quick_ckks_encoder_vec_complex(Complex64* input, Data64* output,
                                       const double scale,
                                       bool use_all_bases = false);

        __host__ void
        quick_ckks_encoder_constant_complex(Complex64 input, Data64* output,
                                            const double scale,
                                            bool use_all_bases = false);

        __host__ void
        quick_ckks_encoder_constant_double(double input, Data64* output,
                                           const double scale,
                                           bool use_all_bases = false);

        __host__ void
        quick_ckks_encoder_constant_integer(std::int64_t input, Data64* output,
                                            const double scale,
                                            bool use_all_bases = false);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_matrixs(Vandermonde& vandermonde, const double scale,
                         bool use_all_bases = false);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_inv_matrixs(Vandermonde& vandermonde, const double scale,
                             bool use_all_bases = false);

        ///////////////////////////////////////////////////

        __host__ Ciphertext multiply_matrix(
            Ciphertext& cipher,
            std::vector<heongpu::DeviceVector<Data64>>& matrix,
            std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
            Galoiskey& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext multiply_matrix_less_memory(
            Ciphertext& cipher,
            std::vector<heongpu::DeviceVector<Data64>>& matrix,
            std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
            std::vector<std::vector<std::vector<int>>>& real_shift,
            Galoiskey& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ std::vector<Ciphertext>
        coeff_to_slot(Ciphertext& cipher, Galoiskey& galois_key,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext solo_coeff_to_slot(
            Ciphertext& cipher, Galoiskey& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext slot_to_coeff(
            Ciphertext& cipher0, Ciphertext& cipher1, Galoiskey& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext solo_slot_to_coeff(
            Ciphertext& cipher, Galoiskey& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext
        exp_scaled(Ciphertext& cipher, Relinkey& relin_key,
                   const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext exp_taylor_approximation(
            Ciphertext& cipher, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        // Double-hoisting BSGS matrix×vector algorithm
        __host__ DeviceVector<Data64> fast_single_hoisting_rotation_ckks(
            Ciphertext& input1, std::vector<int>& bsgs_shift, int n1,
            Galoiskey& galois_key, const cudaStream_t stream)
        {
            if (input1.rescale_required_ || input1.relinearization_required_)
            {
                throw std::invalid_argument("Ciphertext can not be rotated!");
            }

            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            switch (static_cast<int>(galois_key.key_type))
            {
                case 1: // KEYSWITHING_METHOD_I
                    if (scheme_ == scheme_type::ckks)
                    {
                        DeviceVector<Data64> result =
                            fast_single_hoisting_rotation_ckks_method_I(
                                input1, bsgs_shift, n1, galois_key, stream);
                        return result;
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Scheme Type");
                    }
                    break;
                case 2: // KEYSWITHING_METHOD_II
                    if (scheme_ == scheme_type::ckks)
                    {
                        DeviceVector<Data64> result =
                            fast_single_hoisting_rotation_ckks_method_II(
                                input1, bsgs_shift, n1, galois_key, stream);
                        return result;
                    }
                    else
                    {
                        throw std::invalid_argument("Invalid Scheme Type");
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
        }

        __host__ DeviceVector<Data64>
        fast_single_hoisting_rotation_ckks_method_I(
            Ciphertext& first_cipher, std::vector<int>& bsgs_shift, int n1,
            Galoiskey& galois_key, const cudaStream_t stream);

        __host__ DeviceVector<Data64>
        fast_single_hoisting_rotation_ckks_method_II(
            Ciphertext& first_cipher, std::vector<int>& bsgs_shift, int n1,
            Galoiskey& galois_key, const cudaStream_t stream);

        // Pre-computed encoded parameters
        // CtoS part
        DeviceVector<Data64> encoded_constant_1over2_;
        DeviceVector<Data64> encoded_complex_minus_iover2_;
        // StoC part
        DeviceVector<Data64> encoded_complex_i_;
        // Scale part
        DeviceVector<Data64> encoded_complex_minus_iscale_;
        // Exponentiate part
        DeviceVector<Data64> encoded_complex_iscaleoverr_;
        // Sinus taylor part
        DeviceVector<Data64> encoded_constant_1_;
        // DeviceVector<Data64> encoded_constant_1over2_; // we already have it.
        DeviceVector<Data64> encoded_constant_1over6_;
        DeviceVector<Data64> encoded_constant_1over24_;
        DeviceVector<Data64> encoded_constant_1over120_;
        DeviceVector<Data64> encoded_constant_1over720_;
        DeviceVector<Data64> encoded_constant_1over5040_;
    };

    /**
     * @brief HEArithmeticOperator performs arithmetic operations on
     * ciphertexts.
     */
    class HEArithmeticOperator : public HEOperator
    {
      public:
        /**
         * @brief Constructs a new HEArithmeticOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for arithmetic operations.
         */
        HEArithmeticOperator(Parameters& context, HEEncoder& encoder);

        /**
         * @brief Generates bootstrapping parameters.
         *
         * @param scale Scaling factor.
         * @param config Bootstrapping configuration.
         */
        __host__ void
        generate_bootstrapping_params(const double scale,
                                      const BootstrappingConfig& config);

        __host__ std::vector<int> bootstrapping_key_indexs()
        {
            if (!boot_context_generated_)
            {
                throw std::invalid_argument(
                    "Bootstrapping key indexs can not be returned before "
                    "generating Bootstrapping parameters!");
            }

            return key_indexs_;
        }

        /**
         * @brief Performs regular bootstrapping on a ciphertext.(For more
         * detail please check README.md)
         *
         * @param input1 Input ciphertext.
         * @param galois_key Galois key.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped ciphertext.
         */
        __host__ Ciphertext regular_bootstrapping(
            Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Performs slim bootstrapping on a ciphertext.(For more detail
         * please check README.md)
         *
         * @param input1 Input ciphertext.
         * @param galois_key Galois key.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped ciphertext.
         */
        __host__ Ciphertext slim_bootstrapping(
            Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());
    };

    /**
     * @brief HELogicOperator performs homomorphic logical operations on
     * ciphertexts.
     */
    class HELogicOperator : private HEOperator
    {
      public:
        /**
         * @brief Constructs a new HELogicOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for homomorphic operations.
         */
        HELogicOperator(Parameters& context, HEEncoder& encoder,
                        double scale = 0);

        /**
         * @brief Performs logical NOT on ciphertext.
         *
         * @param input1 First input ciphertext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOT(Ciphertext& input1, Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext& output_)
                        { one_minus_cipher(input1_, output_, options_inner); },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOT on ciphertext.
         *
         * @param input1 Ciphertext updated with result.
         * @param options Execution options.
         */
        __host__ void
        NOT_inplace(Ciphertext& input1,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                { one_minus_cipher_inplace(input1_, options_inner); },
                options, true);
        }

        /**
         * @brief Performs logical AND on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void AND(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_, output_,
                                                     options_inner);
                                            relinearize_inplace(output_,
                                                                relin_key,
                                                                options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_, output_,
                                                     options_inner);
                                            relinearize_inplace(output_,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(output_,
                                                            options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical AND on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        AND_inplace(Ciphertext& input1, Ciphertext& input2, Relinkey& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            AND(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical AND on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void AND(Ciphertext& input1, Plaintext& input2,
                          Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           output_,
                                                           options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           output_,
                                                           options_inner);
                                            rescale_inplace(output_,
                                                            options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical AND on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        AND_inplace(Ciphertext& input1, Plaintext& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            AND(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical OR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void OR(Ciphertext& input1, Ciphertext& input2,
                         Ciphertext& output, Relinkey& relin_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical OR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        OR_inplace(Ciphertext& input1, Ciphertext& input2, Relinkey& relin_key,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            OR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical OR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void OR(Ciphertext& input1, Plaintext& input2,
                         Ciphertext& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical OR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        OR_inplace(Ciphertext& input1, Plaintext& input2,
                   const ExecutionOptions& options = ExecutionOptions())
        {
            OR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical XOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void XOR(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        XOR_inplace(Ciphertext& input1, Ciphertext& input2, Relinkey& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            XOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical XOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void XOR(Ciphertext& input1, Plaintext& input2,
                          Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XOR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        XOR_inplace(Ciphertext& input1, Plaintext& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            XOR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical NAND on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void NAND(Ciphertext& input1, Ciphertext& input2,
                           Ciphertext& output, Relinkey& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_, output_,
                                                     options_inner);
                                            relinearize_inplace(output_,
                                                                relin_key,
                                                                options_inner);
                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_, output_,
                                                     options_inner);
                                            relinearize_inplace(output_,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(output_,
                                                            options_inner);
                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NAND on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext& input1, Ciphertext& input2,
                     Relinkey& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            NAND(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical NAND on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NAND(Ciphertext& input1, Plaintext& input2,
                           Ciphertext& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           output_,
                                                           options_inner);
                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           output_,
                                                           options_inner);
                                            rescale_inplace(output_,
                                                            options_inner);
                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NAND on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        NAND_inplace(Ciphertext& input1, Plaintext& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            NAND(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical NOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void NOR(Ciphertext& input1, Ciphertext& input2,
                          Ciphertext& output, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        NOR_inplace(Ciphertext& input1, Ciphertext& input2, Relinkey& relin_key,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            NOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical NOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOR(Ciphertext& input1, Plaintext& input2,
                          Ciphertext& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical NOR on a ciphertext and a plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        NOR_inplace(Ciphertext& input1, Plaintext& input2,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            NOR(input1, input2, input1, options);
        }

        /**
         * @brief Performs logical XNOR on two ciphertexts.
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param output Output ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void XNOR(Ciphertext& input1, Ciphertext& input2,
                           Ciphertext& output, Relinkey& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply(input1_, input2_,
                                                     inner_input1,
                                                     options_inner);
                                            relinearize_inplace(inner_input1,
                                                                relin_key,
                                                                options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add(input1_, input2_, inner_input2,
                                                options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, (&input2 == &output));
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XNOR on two ciphertexts.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Second input ciphertext.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         */
        __host__ void
        XNOR_inplace(Ciphertext& input1, Ciphertext& input2,
                     Relinkey& relin_key,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            XNOR(input1, input2, input1, relin_key, options);
        }

        /**
         * @brief Performs logical XNOR on a ciphertext and a plaintext.
         *
         * @param input1 Input ciphertext.
         * @param input2 Input plaintext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void XNOR(Ciphertext& input1, Plaintext& input2,
                           Ciphertext& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext& output_)
                                {
                                    switch (static_cast<int>(scheme_))
                                    {
                                        case 1: // BFV
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 2: // CKKS
                                            multiply_plain(input1_, input2_,
                                                           inner_input1,
                                                           options_inner);
                                            rescale_inplace(inner_input1,
                                                            options_inner);
                                            add(inner_input1, inner_input1,
                                                inner_input1, options_inner);

                                            add_plain(input1_, input2_,
                                                      inner_input2,
                                                      options_inner);

                                            mod_drop_inplace(inner_input2);

                                            sub(inner_input2, inner_input1,
                                                output_, options_inner);

                                            one_minus_cipher_inplace(
                                                output_, options_inner);
                                            break;
                                        case 3: // BGV
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                        default:
                                            throw std::invalid_argument(
                                                "Invalid Scheme Type");
                                            break;
                                    }
                                },
                                options);
                        },
                        options, false);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Performs in-place logical XNOR on a ciphertext and a
         * plaintext.
         *
         * @param input1 Ciphertext updated with result.
         * @param input2 Input plaintext.
         * @param options Execution options.
         */
        __host__ void
        XNOR_inplace(Ciphertext& input1, Plaintext& input2,
                     const ExecutionOptions& options = ExecutionOptions())
        {
            XNOR(input1, input2, input1, options);
        }

        /**
         * @brief Generates bootstrapping parameters.
         *
         * @param scale Scaling factor.
         * @param config Bootstrapping configuration.
         */
        __host__ void generate_bootstrapping_params(
            const double scale, const BootstrappingConfig& config,
            const logic_bootstrapping_type& boot_type);

        /**
         * @brief Retrieves galois key indexes required for bootstrapping.
         *
         * @return std::vector<int> Bootstrapping key indexes.
         * @throws std::invalid_argument if parameters are not generated.
         */
        __host__ std::vector<int> bootstrapping_key_indexs()
        {
            if (!boot_context_generated_)
            {
                throw std::invalid_argument(
                    "Bootstrapping key indexs can not be returned before "
                    "generating Bootstrapping parameters!");
            }

            return key_indexs_;
        }

        /**
         * @brief Performs bit-level bootstrapping on a ciphertext.(For more
         * detail please check README.md)
         *
         * @param input1 Input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped ciphertext.
         */
        __host__ Ciphertext bit_bootstrapping(
            Ciphertext& input1, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Bootstrapped logical AND gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        AND_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                          Galoiskey& galois_key, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::AND, input1, input2,
                                      galois_key, relin_key, options);
        }

        /**
         * @brief Bootstrapped logical OR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        OR_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                         Galoiskey& galois_key, Relinkey& relin_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::OR, input1, input2,
                                      galois_key, relin_key, options);
        }

        /**
         * @brief Bootstrapped logical XOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        XOR_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                          Galoiskey& galois_key, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::XOR, input1, input2,
                                      galois_key, relin_key, options);
        }

        /**
         * @brief Bootstrapped logical NAND gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        NAND_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                           Galoiskey& galois_key, Relinkey& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::NAND, input1, input2,
                                      galois_key, relin_key, options);
        }

        /**
         * @brief Bootstrapped logical NOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        NOR_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                          Galoiskey& galois_key, Relinkey& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::NOR, input1, input2,
                                      galois_key, relin_key, options);
        }

        /**
         * @brief Bootstrapped logical XNOR gate.(For more detail please check
         * README.md)
         *
         * @param input1 First input ciphertext.
         * @param input2 Second input ciphertext.
         * @param galois_key Galois key for rotations.
         * @param relin_key Relinearization key.
         * @param options Execution options.
         * @return Ciphertext Bootstrapped result.
         */
        __host__ Ciphertext
        XNOR_bootstrapping(Ciphertext& input1, Ciphertext& input2,
                           Galoiskey& galois_key, Relinkey& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::XNOR, input1, input2,
                                      galois_key, relin_key, options);
        }

        using HEOperator::apply_galois;
        using HEOperator::apply_galois_inplace;
        using HEOperator::keyswitch;
        using HEOperator::mod_drop;
        using HEOperator::mod_drop_inplace;
        using HEOperator::rotate_rows;
        using HEOperator::rotate_rows_inplace;

      private:
        enum class logic_gate
        {
            AND,
            OR,
            XOR,
            NAND,
            NOR,
            XNOR
        };

        __host__ Ciphertext gate_bootstrapping(
            logic_gate gate_type, Ciphertext& input1, Ciphertext& input2,
            Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext AND_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext OR_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext XOR_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext NAND_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext NOR_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext XNOR_approximation(
            Ciphertext& cipher, Galoiskey& galois_key, Relinkey& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        one_minus_cipher(Ciphertext& input1, Ciphertext& output,
                         const ExecutionOptions& options = ExecutionOptions());

        __host__ void one_minus_cipher_inplace(
            Ciphertext& input1,
            const ExecutionOptions& options = ExecutionOptions());

        // Encoded One
        DeviceVector<Data64> encoded_constant_one_;

        // Bit bootstrapping
        DeviceVector<Data64> encoded_constant_minus_1over4_;

        // Gate bootstrapping
        DeviceVector<Data64> encoded_constant_1over3_;
        DeviceVector<Data64> encoded_constant_2over3_;
        DeviceVector<Data64> encoded_complex_minus_2over6j_;
        DeviceVector<Data64> encoded_constant_minus_2over6_;
        DeviceVector<Data64> encoded_complex_2over6j_;
        DeviceVector<Data64> encoded_constant_2over6_;
        // DeviceVector<Data64> we have -> encoded_complex_minus_iscale_
        DeviceVector<Data64> encoded_constant_pioversome_;
        DeviceVector<Data64> encoded_constant_minus_pioversome_;
    };

} // namespace heongpu

#endif // OPERATOR_H
