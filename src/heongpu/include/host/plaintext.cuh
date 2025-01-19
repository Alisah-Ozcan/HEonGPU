// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef PLAINTEXT_H
#define PLAINTEXT_H

#include "context.cuh"

namespace heongpu
{
    /**
     * @brief Plaintext represents the unencrypted data used in homomorphic
     * encryption.
     *
     * The Plaintext class is initialized with encryption parameters and
     * optionally a CUDA stream. It provides methods to manage the underlying
     * data, such as resizing, and transferring between host (CPU) and device
     * (GPU). This class is used for preparing data for encoding, encryption,
     * and other homomorphic operations.
     */
    class Plaintext
    {
        friend class HEEncoder;
        friend class HEEncryptor;
        friend class HEDecryptor;
        friend class HEOperator;

      public:
        /**
         * @brief Constructs an empty Plaintext object.
         */
        __host__ Plaintext();

        /**
         * @brief Constructs a new Plaintext object with specified parameters
         * and an optional CUDA stream.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        explicit __host__ Plaintext(Parameters& context,
                                    cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Constructs a new Plaintext object from a vector of data with
         * specified parameters and an optional CUDA stream.
         *
         * @param plain Vector of Data representing the initial value of the
         * plaintext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        explicit __host__ Plaintext(const std::vector<Data>& plain,
                                    Parameters& context,
                                    cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Constructs a new Plaintext object from a HostVector of data
         * with specified parameters and an optional CUDA stream.
         *
         * @param plain Vector of Data representing the initial value of the
         * plaintext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        explicit __host__ Plaintext(const HostVector<Data>& plain,
                                    Parameters& context,
                                    cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Returns a pointer to the underlying plaintext data.
         *
         * @return Data* Pointer to the plaintext data.
         */
        Data* data();

        /**
         * @brief Resizes the plaintext to the new specified size using the
         * given CUDA stream.
         *
         * @param new_size The new size of the plaintext.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void resize(int new_size, cudaStream_t stream = cudaStreamDefault)
        {
            locations_.resize(new_size, stream);
            plain_size_ = new_size;
        }

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param plain Vector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void device_to_host(std::vector<Data>& plain,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param plain Vector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void host_to_device(std::vector<Data>& plain,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param plain HostVector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void device_to_host(HostVector<Data>& plain,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param plain HostVector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void host_to_device(HostVector<Data>& plain,
                            cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Returns the size of the plaintext.
         *
         * @return int Size of the plaintext.
         */
        inline int size() const noexcept { return plain_size_; }

        /**
         * @brief Returns the current depth level of the plaintext.
         *
         * @return int Depth level of the plaintext.
         */
        inline int depth() const noexcept { return depth_; }

        /**
         * @brief Returns the scaling factor used for encoding in CKKS scheme.
         *
         * @return double Scaling factor.
         */
        inline double scale() const noexcept { return scale_; }

        /**
         * @brief Indicates whether the plaintext is in the NTT (Number
         * Theoretic Transform) domain.
         *
         * @return bool True if in NTT domain, false otherwise.
         */
        inline bool in_ntt_domain() const noexcept { return in_ntt_domain_; }

        Plaintext(const Plaintext& copy)
            : scheme_(copy.scheme_), plain_size_(copy.plain_size_),
              depth_(copy.depth_), scale_(copy.scale_),
              in_ntt_domain_(copy.in_ntt_domain_)
        {
            locations_.resize(copy.locations_.size(), copy.locations_.stream());
            cudaMemcpyAsync(
                locations_.data(), copy.locations_.data(),
                copy.locations_.size() * sizeof(Data), cudaMemcpyDeviceToDevice,
                copy.locations_.stream()); // TODO: use cudaStreamPerThread
        }

        Plaintext(Plaintext&& assign) noexcept
            : scheme_(std::move(assign.scheme_)),
              plain_size_(std::move(assign.plain_size_)),
              depth_(std::move(assign.depth_)),
              scale_(std::move(assign.scale_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              locations_(std::move(assign.locations_))
        {
        }

        Plaintext& operator=(const Plaintext& copy)
        {
            if (this != &copy)
            {
                scheme_ = copy.scheme_;
                plain_size_ = copy.plain_size_;
                depth_ = copy.depth_;
                scale_ = copy.scale_;
                in_ntt_domain_ = copy.in_ntt_domain_;

                locations_.resize(copy.locations_.size(),
                                  copy.locations_.stream());
                cudaMemcpyAsync(
                    locations_.data(), copy.locations_.data(),
                    copy.locations_.size() * sizeof(Data),
                    cudaMemcpyDeviceToDevice,
                    copy.locations_.stream()); // TODO: use cudaStreamPerThread
            }
            return *this;
        }

        Plaintext& operator=(Plaintext&& assign) noexcept
        {
            if (this != &assign)
            {
                scheme_ = std::move(assign.scheme_);
                plain_size_ = std::move(assign.plain_size_);
                depth_ = std::move(assign.depth_);
                scale_ = std::move(assign.scale_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);

                locations_ = std::move(assign.locations_);
            }
            return *this;
        }

      private:
        scheme_type scheme_;

        int plain_size_;

        int depth_;

        double scale_;

        bool in_ntt_domain_;

        DeviceVector<Data> locations_;
    };
} // namespace heongpu
#endif // PLAINTEXT_H
