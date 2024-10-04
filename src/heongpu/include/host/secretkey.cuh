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
      public:
        /**
         * @brief Constructs a new Secretkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Secretkey(Parameters& context);

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
            location.resize(copy.location.size(), cudaStreamLegacy);
            cudaMemcpyAsync(location.data(), copy.location.data(),
                            copy.location.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        Secretkey(Secretkey&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              location(std::move(assign.location))
        {
            // location = std::move(assign.location);
        }

        Secretkey& operator=(const Secretkey& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;

                location.resize(copy.location.size(), cudaStreamLegacy);
                cudaMemcpyAsync(
                    location.data(), copy.location.data(),
                    copy.location.size() * sizeof(Data),
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

                location = std::move(assign.location);
            }
            return *this;
        }

      private:
        int ring_size_;
        int coeff_modulus_count_;

        DeviceVector<Data> location;
    };
} // namespace heongpu
#endif // SECRETKEY_H