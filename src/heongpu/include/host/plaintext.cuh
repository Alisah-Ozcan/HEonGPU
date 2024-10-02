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
         * @brief Constructs a new Plaintext object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Plaintext(Parameters& context);

        /**
         * @brief Constructs a new Plaintext object with specified parameters
         * and CUDA stream.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream for operations involving the plaintext.
         */
        __host__ Plaintext(Parameters& context, HEStream& stream);

        /**
         * @brief Constructs a new Plaintext object from a vector of data.
         *
         * @param plain Vector of Data representing the initial value of the
         * plaintext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         */
        __host__ Plaintext(const std::vector<Data>& plain, Parameters& context);

        /**
         * @brief Constructs a new Plaintext object from a vector of data with
         * specified parameters and CUDA stream.
         *
         * @param plain Vector of Data representing the initial value of the
         * plaintext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream for operations involving the plaintext.
         */
        __host__ Plaintext(const std::vector<Data>& plain, Parameters& context,
                           HEStream& stream);

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
         * @param stream CUDA stream to be used for the resize operation,
         * default is cudaStreamLegacy.
         */
        void resize(int new_size, cudaStream_t stream = cudaStreamLegacy)
        {
            locations_.resize(0, stream);
            locations_.resize(new_size, stream);
            plain_size_ = new_size;
        }

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU).
         *
         * @param plain Vector where the device data will be copied to.
         */
        void device_to_host(std::vector<Data>& plain);

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU) asynchronously.
         *
         * @param plain Vector where the device data will be copied to.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous data transfer.
         */
        void device_to_host(std::vector<Data>& plain, HEStream& stream);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU).
         *
         * @param plain Vector of data to be transferred to the device.
         */
        void host_to_device(std::vector<Data>& plain);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU) asynchronously.
         *
         * @param plain Vector of data to be transferred to the device.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous data transfer.
         */
        void host_to_device(std::vector<Data>& plain, HEStream& stream);

        /**
         * @brief Returns the size of the plaintext.
         *
         * @return int Size of the plaintext.
         */
        inline int plain_size() const noexcept { return plain_size_; }

        /**
         * @brief Returns the depth level of the plaintext.
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

        Plaintext() = default;

        Plaintext(const Plaintext& copy)
            : scheme_(copy.scheme_), plain_size_(copy.plain_size_),
              depth_(copy.depth_), scale_(copy.scale_),
              in_ntt_domain_(copy.in_ntt_domain_)
        {
            locations_.resize(copy.locations_.size(), cudaStreamLegacy);
            cudaMemcpyAsync(locations_.data(), copy.locations_.data(),
                            copy.locations_.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        Plaintext(Plaintext&& assign) noexcept
            : scheme_(std::move(assign.scheme_)),
              plain_size_(std::move(assign.plain_size_)),
              depth_(std::move(assign.depth_)),
              scale_(std::move(assign.scale_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              locations_(std::move(assign.locations_))
        {
            // locations_ = std::move(assign.locations_);
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

                locations_.resize(copy.locations_.size(), cudaStreamLegacy);
                cudaMemcpyAsync(
                    locations_.data(), copy.locations_.data(),
                    copy.locations_.size() * sizeof(Data),
                    cudaMemcpyDeviceToDevice,
                    cudaStreamLegacy); // TODO: use cudaStreamPerThread
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
