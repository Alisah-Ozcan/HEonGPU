// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef SECRETKEY_H
#define SECRETKEY_H

#include "context.cuh"
#include "keygeneration.cuh"

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
        friend class HEDecryptor;

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
         * @return Data64* Pointer to the secret key data.
         */
        Data64* data();

        /**
         * @brief Transfers the secretkey data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param secret_key Vector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void device_to_host(std::vector<int>& secret_key,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the secretkey data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param secret_key Vector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void host_to_device(std::vector<int>& secret_key,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the secretkey data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param secret_key HostVector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void device_to_host(HostVector<int>& secret_key,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the secretkey data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param secret_key HostVector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void host_to_device(HostVector<int>& public_key,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Switches the secretkey CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream)
        {
            location_.set_stream(stream);
        }

        /**
         * @brief Retrieves the CUDA stream associated with the secretkey.
         *
         * This function returns the CUDA stream that was used to create or last
         * modify the secretkey.
         *
         * @return The CUDA stream associated with the secretkey.
         */
        cudaStream_t stream() const { return location_.stream(); }

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

        Secretkey() = delete;

        /**
         * @brief Copy constructor for creating a new Secretkey object by
         * copying an existing one.
         *
         * This constructor performs a deep copy of the secret key data,
         * ensuring that the new object has its own independent copy of the
         * data. GPU memory operations are handled using `cudaMemcpyAsync` for
         * asynchronous data transfer.
         *
         * @param copy The source Secretkey object to copy from.
         */
        Secretkey(const Secretkey& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_)
        {
            location_.resize(copy.location_.size(), copy.location_.stream());
            cudaMemcpyAsync(
                location_.data(), copy.location_.data(),
                copy.location_.size() * sizeof(Data64),
                cudaMemcpyDeviceToDevice,
                copy.location_.stream()); // TODO: use cudaStreamPerThread
        }

        /**
         * @brief Move constructor for transferring ownership of a Secretkey
         * object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the newly constructed object. The source object is left in a valid
         * but undefined state.
         *
         * @param assign The source Secretkey object to move from.
         */
        Secretkey(Secretkey&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              location_(std::move(assign.location_))
        {
        }

        /**
         * @brief Copy assignment operator for assigning one Secretkey object to
         * another.
         *
         * Performs a deep copy of the secret key data, ensuring that the target
         * object has its own independent copy. GPU memory is copied using
         * `cudaMemcpyAsync`.
         *
         * @param copy The source Secretkey object to copy from.
         * @return Reference to the assigned object.
         */
        Secretkey& operator=(const Secretkey& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;

                location_.resize(copy.location_.size(),
                                 copy.location_.stream());
                cudaMemcpyAsync(
                    location_.data(), copy.location_.data(),
                    copy.location_.size() * sizeof(Data64),
                    cudaMemcpyDeviceToDevice,
                    copy.location_.stream()); // TODO: use cudaStreamPerThread
            }
            return *this;
        }

        /**
         * @brief Move assignment operator for transferring ownership of a
         * Secretkey object.
         *
         * Transfers all resources, including GPU memory, from the source object
         * to the target object. The source object is left in a valid but
         * undefined state.
         *
         * @param assign The source Secretkey object to move from.
         * @return Reference to the assigned object.
         */
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
        int n_power_;

        std::shared_ptr<DeviceVector<Modulus64>> modulus_;
        std::shared_ptr<DeviceVector<Root64>> ntt_table_;

        int hamming_weight_;
        bool in_ntt_domain_;

        DeviceVector<int> secretkey_; // coefficients are in {-1, 0, 1}
        DeviceVector<Data64> location_; // coefficients are RNS domain
    };
} // namespace heongpu
#endif // SECRETKEY_H