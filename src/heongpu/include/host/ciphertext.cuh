// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef CIPHERTEXT_H
#define CIPHERTEXT_H

#include "context.cuh"

namespace heongpu
{
    class Ciphertext
    {
        friend class HEEncryptor;
        friend class HEDecryptor;
        friend class HEOperator;

      public:
        /**
         * @brief Constructs an empty Ciphertext object.
         */
        __host__ Ciphertext();

        /**
         * @brief Constructs a new Ciphertext object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param scale The scale to be used for the ciphertext, default
         * is 0.
         * @param depth The depth of the ciphertext, default is 0.
         */
        __host__ Ciphertext(Parameters& context, double scale = 0,
                            int depth = 0);

        /**
         * @brief Constructs a new Ciphertext object with specified parameters
         * and CUDA stream.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream for the ciphertext operations.
         * @param scale The scale to be used for the ciphertext, default
         * is 0.
         * @param depth The depth of the ciphertext, default is 0.
         */
        __host__ Ciphertext(Parameters& context, HEStream& steram,
                            double scale = 0, int depth = 0);

        /**
         * @brief Constructs a new Ciphertext object from a vector of data.
         *
         * @param cipher Vector of Data that represents the initial value of the
         * ciphertext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param scale The scale to be used for the ciphertext, default
         * is 0.
         * @param depth The depth of the ciphertext, default is 0.
         */
        __host__ Ciphertext(const std::vector<Data>& cipher,
                            Parameters& context, double scale = 0,
                            int depth = 0);

        /**
         * @brief Constructs a new Ciphertext object from a vector of data with
         * specified parameters and CUDA stream.
         *
         * @param cipher Vector of Data that represents the initial value of the
         * ciphertext.
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream for the ciphertext operations.
         * @param scale The scale to be used for the ciphertext, default
         * is 0.
         * @param depth The depth of the ciphertext, default is 0.
         */
        __host__ Ciphertext(const std::vector<Data>& cipher,
                            Parameters& context, HEStream& steram,
                            double scale = 0, int depth = 0);

        /**
         * @brief Returns a pointer to the underlying data of the ciphertext.
         *
         * @return Data* Pointer to the data.
         */
        Data* data();

        /**
         * @brief Resizes the ciphertext to the new specified size using the
         * given CUDA stream.
         *
         * @param new_size The new size of the ciphertext.
         * @param stream CUDA stream to be used for the resize operation,
         * default is cudaStreamLegacy.
         */
        void resize(int new_size, cudaStream_t stream = cudaStreamLegacy)
        {
            // locations_.resize(0, stream);
            locations_.resize(new_size, stream);
        }

        /**
         * @brief Copies the data from the device to the host.
         *
         * @param cipher Reference to a vector where the device data will be
         * copied to.
         */
        void device_to_host(std::vector<Data>& cipher);

        /**
         * @brief Copies the data from the device to the host asynchronously.
         *
         * @param cipher Reference to a vector where the device data will be
         * copied to.
         * @param stream Reference to the HEStream object representing the CUDA
         * stream to be used for asynchronous data transfer.
         */
        void device_to_host(std::vector<Data>& cipher, HEStream& stream);

        /**
         * @brief Switches the Ciphertext CUDA stream.
         *
         * @param stream The new CUDA stream to be used.
         */
        void switch_stream(cudaStream_t stream);

        /**
         * @brief Returns the size of the polynomial ring used in ciphertext.
         *
         * @return int Size of the polynomial ring.
         */
        inline int ring_size() const noexcept { return ring_size_; }

        /**
         * @brief Returns the number of coefficient modulus primes used in the
         * ciphertext.
         *
         * @return int Number of coefficient modulus primes.
         */
        inline int coeff_modulus_count() const noexcept
        {
            return coeff_modulus_count_;
        }

        /**
         * @brief Returns the size of the ciphertext.
         *
         * @return int Size of the ciphertext.
         */
        inline int size() const noexcept { return cipher_size_; }

        /**
         * @brief Returns the depth level of the ciphertext.
         *
         * @return int Depth level of the ciphertext.
         */
        inline int depth() const noexcept { return depth_; }

        /**
         * @brief Indicates whether the ciphertext is in the NTT (Number
         * Theoretic Transform) domain.
         *
         * @return bool True if in NTT domain, false otherwise.
         */
        inline bool in_ntt_domain() const noexcept { return in_ntt_domain_; }

        /**
         * @brief Returns the scaling factor used for encoding in CKKS scheme.
         *
         * @return double Scaling factor.
         */
        inline double scale() const noexcept { return scale_; }

        /**
         * @brief Indicates whether rescaling is required for the ciphertext.
         *
         * @return bool True if rescaling is required, false otherwise.
         */
        inline bool rescale_required() const noexcept
        {
            return rescale_required_;
        }

        /**
         * @brief Indicates whether relinearization is required for the
         * ciphertext.
         *
         * @return bool True if relinearization is required, false otherwise.
         */
        inline bool relinearization_required() const noexcept
        {
            return relinearization_required_;
        }

        // Ciphertext() = default;

        Ciphertext(const Ciphertext& copy)
            : ring_size_(copy.ring_size_),
              coeff_modulus_count_(copy.coeff_modulus_count_),
              cipher_size_(copy.cipher_size_), depth_(copy.depth_),
              scheme_(copy.scheme_), in_ntt_domain_(copy.in_ntt_domain_),
              scale_(copy.scale_), rescale_required_(copy.rescale_required_),
              relinearization_required_(copy.relinearization_required_)
        {
            locations_.resize(copy.locations_.size(), cudaStreamLegacy);
            cudaMemcpyAsync(locations_.data(), copy.locations_.data(),
                            copy.locations_.size() * sizeof(Data),
                            cudaMemcpyDeviceToDevice,
                            cudaStreamLegacy); // TODO: use cudaStreamPerThread
        }

        Ciphertext(Ciphertext&& assign) noexcept
            : ring_size_(std::move(assign.ring_size_)),
              coeff_modulus_count_(std::move(assign.coeff_modulus_count_)),
              cipher_size_(std::move(assign.cipher_size_)),
              depth_(std::move(assign.depth_)),
              scheme_(std::move(assign.scheme_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              scale_(std::move(assign.scale_)),
              rescale_required_(std::move(assign.rescale_required_)),
              relinearization_required_(
                  std::move(assign.relinearization_required_)),
              locations_(std::move(assign.locations_))
        {
            // locations_ = std::move(assign.locations_);
        }

        Ciphertext& operator=(const Ciphertext& copy)
        {
            if (this != &copy)
            {
                ring_size_ = copy.ring_size_;
                coeff_modulus_count_ = copy.coeff_modulus_count_;
                cipher_size_ = copy.cipher_size_;
                depth_ = copy.depth_;
                scheme_ = copy.scheme_;
                in_ntt_domain_ = copy.in_ntt_domain_;

                scale_ = copy.scale_;
                rescale_required_ = copy.rescale_required_;
                relinearization_required_ = copy.relinearization_required_;

                // locations_ = copy.locations_;
                // locations_.resize(copy.locations_.size(),
                // cudaStreamLegacy); cudaMemcpyAsync(
                //    locations_.data(), copy.locations_.data(),
                //    copy.locations_.size() * sizeof(Data),
                //    cudaMemcpyDeviceToDevice,
                //    cudaStreamLegacy); // TODO: use cudaStreamPerThread

                locations_.resize(copy.locations_.size(),
                                  copy.locations_.stream());
                cudaMemcpyAsync(locations_.data(), copy.locations_.data(),
                                copy.locations_.size() * sizeof(Data),
                                cudaMemcpyDeviceToDevice,
                                copy.locations_.stream());
            }
            return *this;
        }

        Ciphertext& operator=(Ciphertext&& assign) noexcept
        {
            if (this != &assign)
            {
                ring_size_ = std::move(assign.ring_size_);
                coeff_modulus_count_ = std::move(assign.coeff_modulus_count_);
                cipher_size_ = std::move(assign.cipher_size_);
                depth_ = std::move(assign.depth_);
                scheme_ = std::move(assign.scheme_);
                in_ntt_domain_ = std::move(assign.in_ntt_domain_);

                scale_ = std::move(assign.scale_);
                rescale_required_ = std::move(assign.rescale_required_);
                relinearization_required_ =
                    std::move(assign.relinearization_required_);

                locations_ = std::move(assign.locations_);
            }
            return *this;
        }

      private:
        scheme_type scheme_;

        int ring_size_;
        int coeff_modulus_count_;
        int cipher_size_;
        int depth_;

        bool in_ntt_domain_;

        double scale_;
        bool rescale_required_;
        bool relinearization_required_;

        DeviceVector<Data> locations_;
    };
} // namespace heongpu
#endif // CIPHERTEXT_H