// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef PUBLICKEY_H
#define PUBLICKEY_H

#include "context.cuh"

namespace heongpu
{
    /**
     * @brief Publickey represents a public key used for encrypting data in
     * homomorphic encryption schemes.
     *
     * The Publickey class is initialized with encryption parameters and
     * provides a method to access the underlying public key data. This key is
     * used in conjunction with the HEEncryptor class to encrypt plaintexts,
     * making them suitable for homomorphic operations.
     */
    class Publickey
    {
      public:
        /**
         * @brief Constructs a new Publickey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Publickey(Parameters& context);

        /**
         * @brief Returns a pointer to the underlying public key data.
         *
         * @return Data* Pointer to the public key data.
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

        Publickey() = default;

        Publickey(const Publickey& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_)
        {
            locations_.resize(copy.locations_.size(), cudaStreamLegacy);
            cudaMemcpyAsync(locations_.data(), copy.locations_.data(),
                            copy.locations_.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        Publickey(Publickey&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              locations_(std::move(assign.locations_))
        {
            // locations_ = std::move(assign.locations_);
        }

        Publickey& operator=(const Publickey& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;

                locations_.resize(copy.locations_.size(), cudaStreamLegacy);
                cudaMemcpyAsync(
                    locations_.data(), copy.locations_.data(),
                    copy.locations_.size() * sizeof(Data),
                    cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy); // TODO: use cudaStreamPerThread
            }
            return *this;
        }

        Publickey& operator=(Publickey&& assign) noexcept
        {
            if (this != &assign)
            {
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);

                locations_ = std::move(assign.locations_);
            }
            return *this;
        }

      private:
        int ring_size_;
        int coeff_modulus_count_;

        DeviceVector<Data> locations_;
    };
} // namespace heongpu
#endif // PUBLICKEY_H