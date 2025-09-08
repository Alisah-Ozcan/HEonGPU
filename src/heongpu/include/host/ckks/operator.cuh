// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_OPERATOR_H
#define HEONGPU_CKKS_OPERATOR_H

#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include "ntt.cuh"
#include "fft.cuh"
#include "addition.cuh"
#include "multiplication.cuh"
#include "switchkey.cuh"
#include "keygeneration.cuh"
#include "bootstrapping.cuh"

#include "ckks/context.cuh"
#include "ckks/encoder.cuh"
#include "ckks/plaintext.cuh"
#include "ckks/ciphertext.cuh"
#include "ckks/evaluationkey.cuh"

namespace heongpu
{
    /**
     * @brief HEOperator is responsible for performing homomorphic operations on
     * encrypted data, such as addition, subtraction, multiplication, and other
     * functions.
     *
     * The HEOperator class is initialized with encryption parameters and
     * provides various functions for performing operations on ciphertexts,
     * including CKKS scheme. It supports both in-place and
     * out-of-place operations, as well as asynchronous processing using CUDA
     * streams.
     */
    template <> class HEOperator<Scheme::CKKS>
    {
      protected:
        /**
         * @brief Construct a new HEOperator object with the given parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters for the operator.
         */
        __host__ HEOperator(HEContext<Scheme::CKKS>& context,
                            HEEncoder<Scheme::CKKS>& encoder);

      public:
        /**
         * @brief Adds two ciphertexts and stores the result in the output.
         *
         * @param input1 First input ciphertext to be added.
         * @param input2 Second input ciphertext to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void add(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
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
        add_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
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
        __host__ void sub(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Subtracts the second ciphertext from the first, modifying the
         * first ciphertext with the result.
         *
         * @param input1 The ciphertext from which input2 will be subtracted.
         * @param input2 The ciphertext to subtract from input1.
         */
        __host__ void
        sub_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
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
        negate(Ciphertext<Scheme::CKKS>& input1,
               Ciphertext<Scheme::CKKS>& output,
               const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Negates a ciphertext in-place, modifying the input ciphertext.
         *
         * @param input1 Ciphertext to be negated.
         */
        __host__ void
        negate_inplace(Ciphertext<Scheme::CKKS>& input1,
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
        add_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  Ciphertext<Scheme::CKKS>& output,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    add_plain_ckks(input1_, input2_, output_,
                                                   options.stream_);

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
                                    output_.ciphertext_generated_ = true;
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
        add_plain_inplace(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_) {
                            add_plain_ckks_inplace(input1_, input2_,
                                                   options.stream_);
                        },
                        options, false);
                },
                options, true);
        }

        //

        /**
         * @brief Adds a ciphertext and a plaintext and stores the result in the
         * output.
         *
         * @param input1 Input ciphertext to be added.
         * @param input2 Input constant plaintext(double) to be added.
         * @param output Ciphertext where the result of the addition is stored.
         */
        __host__ void
        add_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            add_constant_plain_ckks(input1_,
                                                    input2 * input1_.scale_,
                                                    output_, options.stream_);

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Adds a plaintext to a ciphertext in-place, modifying the input
         * ciphertext.
         *
         * @param input1 Ciphertext to which the plaintext will be added.
         * @param input2 Input constant plaintext(double) to be added.
         */
        __host__ void
        add_plain_inplace(Ciphertext<Scheme::CKKS>& input1, double input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    add_constant_plain_ckks_inplace(
                        input1_, input2 * input1_.scale_, options.stream_);
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
        sub_plain(Ciphertext<Scheme::CKKS>& input1,
                  Plaintext<Scheme::CKKS>& input2,
                  Ciphertext<Scheme::CKKS>& output,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    sub_plain_ckks(input1_, input2_, output_,
                                                   options.stream_);

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
                                    output_.ciphertext_generated_ = true;
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
        sub_plain_inplace(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_) {
                            sub_plain_ckks_inplace(input1_, input2_,
                                                   options.stream_);
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
        sub_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                  Ciphertext<Scheme::CKKS>& output,
                  const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            sub_constant_plain_ckks(input1_,
                                                    input2 * input1_.scale_,
                                                    output_, options.stream_);

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.scale_ = input1_.scale_;
                            output_.rescale_required_ =
                                input1_.rescale_required_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
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
        sub_plain_inplace(Ciphertext<Scheme::CKKS>& input1, double input2,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            if (input1.relinearization_required_)
            {
                throw std::invalid_argument(
                    "Ciphertext and Plaintext can not be added because "
                    "ciphertext has non-linear partl!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    sub_constant_plain_ckks_inplace(
                        input1_, input2 * input1_.scale_, options.stream_);
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
        multiply(Ciphertext<Scheme::CKKS>& input1,
                 Ciphertext<Scheme::CKKS>& input2,
                 Ciphertext<Scheme::CKKS>& output,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_ckks(input1_, input2_, output_,
                                                  options.stream_);
                                    output_.rescale_required_ = true;

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 3;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.relinearization_required_ = true;
                                    output_.ciphertext_generated_ = true;
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
        multiply_inplace(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
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
        multiply_plain(Ciphertext<Scheme::CKKS>& input1,
                       Plaintext<Scheme::CKKS>& input2,
                       Ciphertext<Scheme::CKKS>& output,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    if (input2_.size() <
                                        (n * current_decomp_count))
                                    {
                                        throw std::invalid_argument(
                                            "Invalid Plaintext size!");
                                    }

                                    multiply_plain_ckks(input1_, input2_,
                                                        output_,
                                                        options.stream_);
                                    output_.rescale_required_ = true;

                                    output_.scheme_ = scheme_;
                                    output_.ring_size_ = n;
                                    output_.coeff_modulus_count_ = Q_size_;
                                    output_.cipher_size_ = 2;
                                    output_.depth_ = input1_.depth_;
                                    output_.in_ntt_domain_ =
                                        input1_.in_ntt_domain_;
                                    output_.relinearization_required_ =
                                        input1_.relinearization_required_;
                                    output_.ciphertext_generated_ = true;
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
            Ciphertext<Scheme::CKKS>& input1, Plaintext<Scheme::CKKS>& input2,
            const ExecutionOptions& options = ExecutionOptions())
        {
            multiply_plain(input1, input2, input1, options);
        }

        /**
         * @brief Multiplies a ciphertext and a plaintext and stores the result
         * in the output.
         *
         * @param input1 Input ciphertext to be multiplied.
         * @param input2 Input constant plaintext(double) to be multiplied.
         * @param output Ciphertext where the result of the multiplication is
         * stored.
         */
        __host__ void
        multiply_plain(Ciphertext<Scheme::CKKS>& input1, double input2,
                       Ciphertext<Scheme::CKKS>& output, double scale,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            multiply_const_plain_ckks(input1_, input2, output_,
                                                      scale, options.stream_);
                            output_.rescale_required_ = true;

                            output_.scheme_ = scheme_;
                            output_.ring_size_ = n;
                            output_.coeff_modulus_count_ = Q_size_;
                            output_.cipher_size_ = 2;
                            output_.depth_ = input1_.depth_;
                            output_.in_ntt_domain_ = input1_.in_ntt_domain_;
                            output_.relinearization_required_ =
                                input1_.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Multiplies a plaintext with a ciphertext in-place, modifying
         * the input ciphertext.
         *
         * @param input1 Ciphertext to be multiplied by the plaintext, and where
         * the result will be stored.
         * @param input2 Input constant plaintext(double) to be multiplied.
         */
        __host__ void multiply_plain_inplace(
            Ciphertext<Scheme::CKKS>& input1, double input2, double scale,
            const ExecutionOptions& options = ExecutionOptions())
        {
            multiply_plain(input1, input2, input1, scale, options);
        }

        /**
         * @brief Performs in-place relinearization of the given ciphertext
         * using the provided relin key.
         *
         * @param input1 Ciphertext to be relinearized.
         * @param relin_key The Relinkey object used for relinearization.
         */
        __host__ void relinearize_inplace(
            Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    switch (static_cast<int>(relin_key.key_type))
                    {
                        case 1: // KEYSWITCHING_METHOD_I
                            relinearize_seal_method_inplace_ckks(
                                input1_, relin_key, options.stream_);
                            break;
                        case 2: // KEYSWITCHING_METHOD_II
                            relinearize_external_product_method2_inplace_ckks(
                                input1_, relin_key, options.stream_);
                            break;
                        case 3: // KEYSWITCHING_METHOD_III
                            relinearize_external_product_method_inplace_ckks(
                                input1_, relin_key, options.stream_);
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
        rotate_rows(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& output,
                    Galoiskey<Scheme::CKKS>& galois_key, int shift,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    rotate_ckks_method_I(input1_, output_,
                                                         galois_key, shift,
                                                         options.stream_);
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    rotate_ckks_method_II(input1_, output_,
                                                          galois_key, shift,
                                                          options.stream_);
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
                            output_.ciphertext_generated_ = true;
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
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key, int shift,
            const ExecutionOptions& options = ExecutionOptions())
        {
            if (shift == 0)
            {
                return;
            }

            rotate_rows(input1, input1, galois_key, shift, options);
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
        apply_galois(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& output,
                     Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            switch (static_cast<int>(galois_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    apply_galois_ckks_method_I(
                                        input1_, output_, galois_key,
                                        galois_elt, options.stream_);
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    apply_galois_ckks_method_II(
                                        input1_, output_, galois_key,
                                        galois_elt, options.stream_);
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
                            output_.ciphertext_generated_ = true;
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
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key, int galois_elt,
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
        keyswitch(Ciphertext<Scheme::CKKS>& input1,
                  Ciphertext<Scheme::CKKS>& output,
                  Switchkey<Scheme::CKKS>& switch_key,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            switch (static_cast<int>(switch_key.key_type))
                            {
                                case 1: // KEYSWITCHING_METHOD_I
                                    switchkey_ckks_method_I(input1_, output_,
                                                            switch_key,
                                                            options.stream_);
                                    break;
                                case 2: // KEYSWITCHING_METHOD_II
                                    switchkey_ckks_method_II(input1_, output_,
                                                             switch_key,
                                                             options.stream_);
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
                            output_.ciphertext_generated_ = true;
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
        conjugate(Ciphertext<Scheme::CKKS>& input1,
                  Ciphertext<Scheme::CKKS>& output,
                  Galoiskey<Scheme::CKKS>& conjugate_key,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            switch (static_cast<int>(conjugate_key.key_type))
                            {
                                case 1: // KEYSWITHING_METHOD_I
                                    conjugate_ckks_method_I(input1_, output_,
                                                            conjugate_key,
                                                            options.stream_);
                                    break;
                                case 2: // KEYSWITHING_METHOD_II
                                    conjugate_ckks_method_II(input1_, output_,
                                                             conjugate_key,
                                                             options.stream_);
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
                            output_.ciphertext_generated_ = true;
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
        rescale_inplace(Ciphertext<Scheme::CKKS>& input1,
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

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                { rescale_inplace_ckks_leveled(input1_, options.stream_); },
                options, true);

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
        mod_drop(Ciphertext<Scheme::CKKS>& input1,
                 Ciphertext<Scheme::CKKS>& output,
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

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
                        {
                            mod_drop_ckks_leveled(input1_, output_,
                                                  options.stream_);

                            output.scheme_ = scheme_;
                            output.ring_size_ = n;
                            output.coeff_modulus_count_ = Q_size_;
                            output.cipher_size_ = 2;
                            output.depth_ = input1.depth_ + 1;
                            output.scale_ = input1.scale_;
                            output.in_ntt_domain_ = input1.in_ntt_domain_;
                            output.rescale_required_ = input1.rescale_required_;
                            output.relinearization_required_ =
                                input1.relinearization_required_;
                            output_.ciphertext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
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
        mod_drop(Plaintext<Scheme::CKKS>& input1,
                 Plaintext<Scheme::CKKS>& output,
                 const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.size() < (n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Plaintext size!");
            }

            input_storage_manager(
                input1,
                [&](Plaintext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Plaintext<Scheme::CKKS>& output_)
                        {
                            mod_drop_ckks_plaintext(input1_, output_,
                                                    options.stream_);

                            output.scheme_ = input1.scheme_;
                            output.plain_size_ =
                                (n * (current_decomp_count - 1));
                            output.depth_ = input1.depth_ + 1;
                            output.scale_ = input1.scale_;
                            output.in_ntt_domain_ = input1.in_ntt_domain_;
                            output.plaintext_generated_ = true;
                        },
                        options);
                },
                options, (&input1 == &output));
        }

        /**
         * @brief Drop the last modulus of plaintext in-place on a plaintext,
         * modifying the input plaintext.
         *
         * @param input1 Plaintext to perform modulus dropping on.
         */
        __host__ void
        mod_drop_inplace(Plaintext<Scheme::CKKS>& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.size() < (n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Plaintext size!");
            }

            input_storage_manager(
                input1,
                [&](Plaintext<Scheme::CKKS>& input1_)
                { mod_drop_ckks_plaintext_inplace(input1_, options.stream_); },
                options, true);
        }

        /**
         * @brief Drop the last modulus of ciphertext in-place on a ciphertext,
         * modifying the input ciphertext.
         *
         * @param input1 Ciphertext to perform modulus dropping on.
         */
        __host__ void
        mod_drop_inplace(Ciphertext<Scheme::CKKS>& input1,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            int current_decomp_count = Q_size_ - input1.depth_;

            if (input1.memory_size() < (2 * n * current_decomp_count))
            {
                throw std::invalid_argument("Invalid Ciphertexts size!");
            }

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                { mod_drop_ckks_leveled_inplace(input1_, options.stream_); },
                options, true);
        }

        HEOperator() = default;
        HEOperator(const HEOperator& copy) = default;
        HEOperator(HEOperator&& source) = default;
        HEOperator& operator=(const HEOperator& assign) = default;
        HEOperator& operator=(HEOperator&& assign) = default;

      protected:
        __host__ void add_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                     Plaintext<Scheme::CKKS>& input2,
                                     Ciphertext<Scheme::CKKS>& output,
                                     const cudaStream_t stream);

        __host__ void add_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                             Plaintext<Scheme::CKKS>& input2,
                                             const cudaStream_t stream);

        __host__ void add_constant_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                              double input2,
                                              Ciphertext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        add_constant_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                        double input2,
                                        const cudaStream_t stream);

        __host__ void sub_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                     Plaintext<Scheme::CKKS>& input2,
                                     Ciphertext<Scheme::CKKS>& output,
                                     const cudaStream_t stream);

        __host__ void sub_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                             Plaintext<Scheme::CKKS>& input2,
                                             const cudaStream_t stream);

        __host__ void sub_constant_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                              double input2,
                                              Ciphertext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        sub_constant_plain_ckks_inplace(Ciphertext<Scheme::CKKS>& input1,
                                        double input2,
                                        const cudaStream_t stream);

        __host__ void multiply_ckks(Ciphertext<Scheme::CKKS>& input1,
                                    Ciphertext<Scheme::CKKS>& input2,
                                    Ciphertext<Scheme::CKKS>& output,
                                    const cudaStream_t stream);

        __host__ void multiply_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                          Plaintext<Scheme::CKKS>& input2,
                                          Ciphertext<Scheme::CKKS>& output,
                                          const cudaStream_t stream);

        __host__ void
        multiply_const_plain_ckks(Ciphertext<Scheme::CKKS>& input1,
                                  double input2,
                                  Ciphertext<Scheme::CKKS>& output,
                                  double scale, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        relinearize_seal_method_inplace_ckks(Ciphertext<Scheme::CKKS>& input1,
                                             Relinkey<Scheme::CKKS>& relin_key,
                                             const cudaStream_t stream);

        __host__ void relinearize_external_product_method_inplace_ckks(
            Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
            const cudaStream_t stream);

        __host__ void relinearize_external_product_method2_inplace_ckks(
            Ciphertext<Scheme::CKKS>& input1, Relinkey<Scheme::CKKS>& relin_key,
            const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void rotate_ckks_method_I(Ciphertext<Scheme::CKKS>& input1,
                                           Ciphertext<Scheme::CKKS>& output,
                                           Galoiskey<Scheme::CKKS>& galois_key,
                                           int shift,
                                           const cudaStream_t stream);

        __host__ void rotate_ckks_method_II(Ciphertext<Scheme::CKKS>& input1,
                                            Ciphertext<Scheme::CKKS>& output,
                                            Galoiskey<Scheme::CKKS>& galois_key,
                                            int shift,
                                            const cudaStream_t stream);

        ///////////////////////////////////////////////////

        // TODO: Merge with rotation, provide code integrity
        __host__ void
        apply_galois_ckks_method_I(Ciphertext<Scheme::CKKS>& input1,
                                   Ciphertext<Scheme::CKKS>& output,
                                   Galoiskey<Scheme::CKKS>& galois_key,
                                   int galois_elt, const cudaStream_t stream);

        __host__ void
        apply_galois_ckks_method_II(Ciphertext<Scheme::CKKS>& input1,
                                    Ciphertext<Scheme::CKKS>& output,
                                    Galoiskey<Scheme::CKKS>& galois_key,
                                    int galois_elt, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void switchkey_ckks_method_I(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Switchkey<Scheme::CKKS>& switch_key, const cudaStream_t stream);

        __host__ void switchkey_ckks_method_II(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Switchkey<Scheme::CKKS>& switch_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void conjugate_ckks_method_I(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Galoiskey<Scheme::CKKS>& conjugate_key, const cudaStream_t stream);

        __host__ void conjugate_ckks_method_II(
            Ciphertext<Scheme::CKKS>& input1, Ciphertext<Scheme::CKKS>& output,
            Galoiskey<Scheme::CKKS>& conjugate_key, const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void
        rescale_inplace_ckks_leveled(Ciphertext<Scheme::CKKS>& input1,
                                     const cudaStream_t stream);

        ///////////////////////////////////////////////////

        __host__ void mod_drop_ckks_leveled(Ciphertext<Scheme::CKKS>& input1,
                                            Ciphertext<Scheme::CKKS>& output,
                                            const cudaStream_t stream);

        __host__ void mod_drop_ckks_plaintext(Plaintext<Scheme::CKKS>& input1,
                                              Plaintext<Scheme::CKKS>& output,
                                              const cudaStream_t stream);

        __host__ void
        mod_drop_ckks_plaintext_inplace(Plaintext<Scheme::CKKS>& input1,
                                        const cudaStream_t stream);

        __host__ void
        mod_drop_ckks_leveled_inplace(Ciphertext<Scheme::CKKS>& input1,
                                      const cudaStream_t stream);

      protected:
        scheme_type scheme_;

        int n;

        int n_power;

        // New
        int Q_prime_size_;
        int Q_size_;
        int P_size_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;
        std::shared_ptr<DeviceVector<Root64>> intt_table_;
        std::shared_ptr<DeviceVector<Ninverse64>> n_inverse_;
        std::shared_ptr<DeviceVector<Data64>> last_q_modinv_;

        std::shared_ptr<DeviceVector<Data64>> half_p_;
        std::shared_ptr<DeviceVector<Data64>> half_mod_;

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
        __host__ Plaintext<Scheme::CKKS>
        operator_plaintext(cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Plaintext<Scheme::CKKS>
        operator_from_plaintext(Plaintext<Scheme::CKKS>& input,
                                cudaStream_t stream = cudaStreamDefault);

        __host__ Ciphertext<Scheme::CKKS>
        operator_ciphertext(double scale,
                            cudaStream_t stream = cudaStreamDefault);

        // Just for copy parameters, not memory!
        __host__ Ciphertext<Scheme::CKKS>
        operator_from_ciphertext(Ciphertext<Scheme::CKKS>& input,
                                 cudaStream_t stream = cudaStreamDefault);

        __host__ std::vector<int> rotation_index_generator(uint64_t n, int K, int M);

        class Vandermonde
        {
            template <Scheme S> friend class HEOperator;
            template <Scheme S> friend class HEArithmeticOperator;
            template <Scheme S> friend class HELogicOperator;

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
        int CtoS_level_;
        int StoC_level_;

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

        // CKKS
        double two_pow_64_;
        std::shared_ptr<DeviceVector<int>> reverse_order_;
        std::shared_ptr<DeviceVector<Complex64>> special_ifft_roots_table_;

        __host__ void quick_ckks_encoder_vec_complex(Complex64* input,
                                                     Data64* output,
                                                     const double scale,
                                                     int rns_count);

        __host__ void quick_ckks_encoder_constant_complex(Complex64 input,
                                                          Data64* output,
                                                          const double scale);

        __host__ void quick_ckks_encoder_constant_double(double input,
                                                         Data64* output,
                                                         const double scale);

        __host__ void quick_ckks_encoder_constant_integer(std::int64_t input,
                                                          Data64* output,
                                                          const double scale);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_matrixs(Vandermonde& vandermonde, const double scale,
                         int rns_count);

        __host__ std::vector<heongpu::DeviceVector<Data64>>
        encode_V_inv_matrixs(Vandermonde& vandermonde, const double scale,
                             int rns_count);

        ///////////////////////////////////////////////////

        __host__ Ciphertext<Scheme::CKKS> multiply_matrix(
            Ciphertext<Scheme::CKKS>& cipher,
            std::vector<heongpu::DeviceVector<Data64>>& matrix,
            std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> multiply_matrix_less_memory(
            Ciphertext<Scheme::CKKS>& cipher,
            std::vector<heongpu::DeviceVector<Data64>>& matrix,
            std::vector<std::vector<std::vector<int>>>& diags_matrices_bsgs_,
            std::vector<std::vector<std::vector<int>>>& real_shift,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ std::vector<Ciphertext<Scheme::CKKS>>
        coeff_to_slot(Ciphertext<Scheme::CKKS>& cipher,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> solo_coeff_to_slot(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        slot_to_coeff(Ciphertext<Scheme::CKKS>& cipher0,
                      Ciphertext<Scheme::CKKS>& cipher1,
                      Galoiskey<Scheme::CKKS>& galois_key,
                      const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> solo_slot_to_coeff(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        exp_scaled(Ciphertext<Scheme::CKKS>& cipher,
                   Relinkey<Scheme::CKKS>& relin_key,
                   const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> exp_taylor_approximation(
            Ciphertext<Scheme::CKKS>& cipher, Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        // Double-hoisting BSGS matrix×vector algorithm
        __host__ DeviceVector<Data64>
        fast_single_hoisting_rotation_ckks(Ciphertext<Scheme::CKKS>& input1,
                                           std::vector<int>& bsgs_shift, int n1,
                                           Galoiskey<Scheme::CKKS>& galois_key,
                                           const cudaStream_t stream)
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
            Ciphertext<Scheme::CKKS>& first_cipher,
            std::vector<int>& bsgs_shift, int n1,
            Galoiskey<Scheme::CKKS>& galois_key, const cudaStream_t stream);

        __host__ DeviceVector<Data64>
        fast_single_hoisting_rotation_ckks_method_II(
            Ciphertext<Scheme::CKKS>& first_cipher,
            std::vector<int>& bsgs_shift, int n1,
            Galoiskey<Scheme::CKKS>& galois_key, const cudaStream_t stream);

        // Pre-computed encoded parameters
        // CtoS part
        DeviceVector<Data64> encoded_complex_minus_iover2_;
        // StoC part
        DeviceVector<Data64> encoded_complex_i_;
        // Scale part
        DeviceVector<Data64> encoded_complex_minus_iscale_;
        // Exponentiate part
        DeviceVector<Data64> encoded_complex_iscaleoverr_;
        // Sinus taylor part
    };

    /**
     * @brief HEArithmeticOperator performs arithmetic operations on
     * ciphertexts.
     */
    template <>
    class HEArithmeticOperator<Scheme::CKKS> : public HEOperator<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HEArithmeticOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for arithmetic operations.
         */
        HEArithmeticOperator(HEContext<Scheme::CKKS>& context,
                             HEEncoder<Scheme::CKKS>& encoder);

        /**
         * @brief Generates bootstrapping parameters.
         *
         * @param scale Scaling factor.
         * @param config Bootstrapping configuration.
         */
        __host__ void generate_bootstrapping_params(
            const double scale, const BootstrappingConfig& config,
            const arithmetic_bootstrapping_type& boot_type);

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
        __host__ Ciphertext<Scheme::CKKS> regular_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS> slim_bootstrapping(
            Ciphertext<Scheme::CKKS>& input1,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());
    };

    /**
     * @brief HELogicOperator performs homomorphic logical operations on
     * ciphertexts.
     */
    template <>
    class HELogicOperator<Scheme::CKKS> : private HEOperator<Scheme::CKKS>
    {
      public:
        /**
         * @brief Constructs a new HELogicOperator object.
         *
         * @param context Encryption parameters.
         * @param encoder Encoder for homomorphic operations.
         */
        HELogicOperator(HEContext<Scheme::CKKS>& context,
                        HEEncoder<Scheme::CKKS>& encoder, double scale);

        /**
         * @brief Performs logical NOT on ciphertext.
         *
         * @param input1 First input ciphertext.
         * @param output Output ciphertext.
         * @param options Execution options.
         */
        __host__ void NOT(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    output_storage_manager(
                        output,
                        [&](Ciphertext<Scheme::CKKS>& output_)
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
        NOT_inplace(Ciphertext<Scheme::CKKS>& input1,
                    const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
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
        __host__ void AND(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
                                                        options_inner);
                                    rescale_inplace(output_, options_inner);
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
        AND_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void AND(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_, output_,
                                                   options_inner);
                                    rescale_inplace(output_, options_inner);
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
        AND_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
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
        __host__ void OR(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         Ciphertext<Scheme::CKKS>& output,
                         Relinkey<Scheme::CKKS>& relin_key,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
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
        OR_inplace(Ciphertext<Scheme::CKKS>& input1,
                   Ciphertext<Scheme::CKKS>& input2,
                   Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void OR(Ciphertext<Scheme::CKKS>& input1,
                         Plaintext<Scheme::CKKS>& input2,
                         Ciphertext<Scheme::CKKS>& output,
                         const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
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
        OR_inplace(Ciphertext<Scheme::CKKS>& input1,
                   Plaintext<Scheme::CKKS>& input2,
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
        __host__ void XOR(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
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
        XOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void XOR(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);
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
        XOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
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
        __host__ void NAND(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           Relinkey<Scheme::CKKS>& relin_key,
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
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, output_,
                                             options_inner);
                                    relinearize_inplace(output_, relin_key,
                                                        options_inner);
                                    rescale_inplace(output_, options_inner);
                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        NAND_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& input2,
                     Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void NAND(Ciphertext<Scheme::CKKS>& input1,
                           Plaintext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_, output_,
                                                   options_inner);
                                    rescale_inplace(output_, options_inner);
                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        NAND_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Plaintext<Scheme::CKKS>& input2,
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
        __host__ void NOR(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        NOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Ciphertext<Scheme::CKKS>& input2,
                    Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void NOR(Ciphertext<Scheme::CKKS>& input1,
                          Plaintext<Scheme::CKKS>& input2,
                          Ciphertext<Scheme::CKKS>& output,
                          const ExecutionOptions& options = ExecutionOptions())
        {
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        NOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                    Plaintext<Scheme::CKKS>& input2,
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
        __host__ void XNOR(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           Relinkey<Scheme::CKKS>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Ciphertext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply(input1_, input2_, inner_input1,
                                             options_inner);
                                    relinearize_inplace(inner_input1, relin_key,
                                                        options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add(input1_, input2_, inner_input2,
                                        options_inner);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        XNOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Ciphertext<Scheme::CKKS>& input2,
                     Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ void XNOR(Ciphertext<Scheme::CKKS>& input1,
                           Plaintext<Scheme::CKKS>& input2,
                           Ciphertext<Scheme::CKKS>& output,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            // TODO: Make it efficient
            ExecutionOptions options_inner =
                ExecutionOptions()
                    .set_stream(options.stream_)
                    .set_storage_type(storage_type::DEVICE)
                    .set_initial_location(true);

            Ciphertext<Scheme::CKKS> inner_input1 =
                operator_from_ciphertext(input1, options.stream_);
            Ciphertext<Scheme::CKKS> inner_input2 =
                operator_from_ciphertext(input1, options.stream_);

            input_storage_manager(
                input1,
                [&](Ciphertext<Scheme::CKKS>& input1_)
                {
                    input_storage_manager(
                        input2,
                        [&](Plaintext<Scheme::CKKS>& input2_)
                        {
                            output_storage_manager(
                                output,
                                [&](Ciphertext<Scheme::CKKS>& output_)
                                {
                                    multiply_plain(input1_, input2_,
                                                   inner_input1, options_inner);
                                    rescale_inplace(inner_input1,
                                                    options_inner);
                                    add(inner_input1, inner_input1,
                                        inner_input1, options_inner);

                                    add_plain(input1_, input2_, inner_input2,
                                              options_inner);

                                    mod_drop_inplace(inner_input2);

                                    sub(inner_input2, inner_input1, output_,
                                        options_inner);

                                    one_minus_cipher_inplace(output_,
                                                             options_inner);
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
        XNOR_inplace(Ciphertext<Scheme::CKKS>& input1,
                     Plaintext<Scheme::CKKS>& input2,
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
        __host__ Ciphertext<Scheme::CKKS>
        bit_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        AND_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        OR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& input2,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        XOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        NAND_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Galoiskey<Scheme::CKKS>& galois_key,
                           Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        NOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                          Ciphertext<Scheme::CKKS>& input2,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
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
        __host__ Ciphertext<Scheme::CKKS>
        XNOR_bootstrapping(Ciphertext<Scheme::CKKS>& input1,
                           Ciphertext<Scheme::CKKS>& input2,
                           Galoiskey<Scheme::CKKS>& galois_key,
                           Relinkey<Scheme::CKKS>& relin_key,
                           const ExecutionOptions& options = ExecutionOptions())
        {
            return gate_bootstrapping(logic_gate::XNOR, input1, input2,
                                      galois_key, relin_key, options);
        }

        using HEOperator<Scheme::CKKS>::apply_galois;
        using HEOperator<Scheme::CKKS>::apply_galois_inplace;
        using HEOperator<Scheme::CKKS>::keyswitch;
        using HEOperator<Scheme::CKKS>::mod_drop;
        using HEOperator<Scheme::CKKS>::mod_drop_inplace;
        using HEOperator<Scheme::CKKS>::rotate_rows;
        using HEOperator<Scheme::CKKS>::rotate_rows_inplace;

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

        __host__ Ciphertext<Scheme::CKKS> gate_bootstrapping(
            logic_gate gate_type, Ciphertext<Scheme::CKKS>& input1,
            Ciphertext<Scheme::CKKS>& input2,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        AND_approximation(Ciphertext<Scheme::CKKS>& cipher,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        OR_approximation(Ciphertext<Scheme::CKKS>& cipher,
                         Galoiskey<Scheme::CKKS>& galois_key,
                         Relinkey<Scheme::CKKS>& relin_key,
                         const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        XOR_approximation(Ciphertext<Scheme::CKKS>& cipher,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> NAND_approximation(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS>
        NOR_approximation(Ciphertext<Scheme::CKKS>& cipher,
                          Galoiskey<Scheme::CKKS>& galois_key,
                          Relinkey<Scheme::CKKS>& relin_key,
                          const ExecutionOptions& options = ExecutionOptions());

        __host__ Ciphertext<Scheme::CKKS> XNOR_approximation(
            Ciphertext<Scheme::CKKS>& cipher,
            Galoiskey<Scheme::CKKS>& galois_key,
            Relinkey<Scheme::CKKS>& relin_key,
            const ExecutionOptions& options = ExecutionOptions());

        __host__ void
        one_minus_cipher(Ciphertext<Scheme::CKKS>& input1,
                         Ciphertext<Scheme::CKKS>& output,
                         const ExecutionOptions& options = ExecutionOptions());

        __host__ void one_minus_cipher_inplace(
            Ciphertext<Scheme::CKKS>& input1,
            const ExecutionOptions& options = ExecutionOptions());

        // Encoded One
        DeviceVector<Data64> encoded_constant_one_;

        // Gate bootstrapping
        DeviceVector<Data64> encoded_complex_minus_2over6j_;
        DeviceVector<Data64> encoded_complex_2over6j_;
    };

} // namespace heongpu

#endif // HEONGPU_CKKS_OPERATOR_H
