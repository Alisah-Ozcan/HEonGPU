// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef SECRETKEY_H
#define SECRETKEY_H

#include "context.cuh"

namespace heongpu
{
    /**
     * @brief Secretkey represents a secret key used for decrypting data and
     * generating other keys in homomorphic encryption schemes.
     *
     * The Secretkey class is initialized with encryption parameters and
     * provides a method to access the underlying secret key data. This key is
     * essential for decryption operations as well as for generating other keys
     * like public keys, relinearization keys, and Galois keys.
     */
    class Secretkey
    {
        friend class HEKeyGenerator;

      public:
        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Secretkey(Parameters& context);

        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param hamming_weight Parameter defining hamming weight of secret
         * key, try to use it as (ring size / 2) for maximum security.
         */
        __host__ Secretkey(Parameters& context, int hamming_weight);

        /**
         * @brief Returns a pointer to the underlying secret key data.
         *
         * @return Data* Pointer to the secret key data.
         */
        Data* data();

        /**
         * @brief Returns the size of the polynomial ring used in the
         * homomorphic encryption scheme.
         *
         * @return int Size of the polynomial ring.
         */
        inline int ring_size() const noexcept { return ring_size_; }

        /**
         * @brief Returns the number of coefficient modulus primes used in the
         * encryption parameters.
         *
         * @return int Number of coefficient modulus primes.
         */
        inline int coeff_modulus_count() const noexcept
        {
            return coeff_modulus_count_;
        }

        Secretkey() = default;

        Secretkey(const Secretkey& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_)
        {
            location_.resize(copy.location_.size(), cudaStreamLegacy);
            cudaMemcpyAsync(location_.data(), copy.location_.data(),
                            copy.location_.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        Secretkey(Secretkey&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              location_(std::move(assign.location_))
        {
            // location_ = std::move(assign.location_);
        }

        Secretkey& operator=(const Secretkey& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;

                location_.resize(copy.location_.size(), cudaStreamLegacy);
                cudaMemcpyAsync(
                    location_.data(), copy.location_.data(),
                    copy.location_.size() * sizeof(Data),
                    cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy); // TODO: use cudaStreamPerThread
            }
            return *this;
        }

        Secretkey& operator=(Secretkey&& assign) noexcept
        {
            if (this != &assign)
            {
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);

                location_ = std::move(assign.location_);
            }
            return *this;
        }

      private:
        int ring_size_;
        int coeff_modulus_count_;
        int hamming_weight_;

        DeviceVector<int> secretkey_; // coefficients are in {-1, 0, 1}
        DeviceVector<Data> location_; // coefficients are RNS domain
    };
} // namespace heongpu
#endif // SECRETKEY_H