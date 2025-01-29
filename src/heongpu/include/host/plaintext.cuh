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

        template <typename T, typename F>
        friend void input_storage_manager(T& object, F function,
                                          ExecutionOptions options,
                                          bool check_initial_condition);
        template <typename T, typename F>
        friend void input_vector_storage_manager(std::vector<T>& objects,
                                                 F function,
                                                 ExecutionOptions options,
                                                 bool is_input_output_same);

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
        explicit __host__
        Plaintext(Parameters& context,
                  const ExecutionOptions& options = ExecutionOptions());

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
        explicit __host__
        Plaintext(const std::vector<Data64>& plain, Parameters& context,
                  const ExecutionOptions& options = ExecutionOptions());

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
        explicit __host__
        Plaintext(const HostVector<Data64>& plain, Parameters& context,
                  const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Stores the plaintext in the device (GPU) memory.
         */
        void store_in_device(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Stores the plaintext in the host (CPU) memory.
         */
        void store_in_host(cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Checks whether the data is stored on the device (GPU) memory.
         */
        bool is_on_device() const noexcept
        {
            if (storage_type_ == storage_type::DEVICE)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        /**
         * @brief Returns a pointer to the underlying plaintext data.
         *
         * @return Data64* Pointer to the plaintext data.
         */
        Data64* data();

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param plain Vector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void get_data(std::vector<Data64>& plain,
                      cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param plain Vector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void set_data(const std::vector<Data64>& plain,
                      const ExecutionOptions& options = ExecutionOptions());

        /**
         * @brief Transfers the plaintext data from the device (GPU) to the host
         * (CPU) using the specified CUDA stream.
         *
         * @param plain HostVector where the device data will be copied to.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void get_data(HostVector<Data64>& plain,
                      cudaStream_t stream = cudaStreamDefault);

        /**
         * @brief Transfers the plaintext data from the host (CPU) to the device
         * (GPU) using the specified CUDA stream.
         *
         * @param plain HostVector of data to be transferred to the device.
         * @param stream The CUDA stream to be used for asynchronous operations.
         * Defaults to `cudaStreamDefault`.
         */
        void set_data(const HostVector<Data64>& plain,
                      const ExecutionOptions& options = ExecutionOptions());

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
              in_ntt_domain_(copy.in_ntt_domain_),
              storage_type_(copy.storage_type_)
        {
            if (copy.storage_type_ == storage_type::DEVICE)
            {
                device_locations_.resize(copy.device_locations_.size(),
                                         copy.device_locations_.stream());
                cudaMemcpyAsync(device_locations_.data(),
                                copy.device_locations_.data(),
                                copy.device_locations_.size() * sizeof(Data64),
                                cudaMemcpyDeviceToDevice,
                                copy.device_locations_
                                    .stream()); // TODO: use cudaStreamPerThread
            }
            else
            {
                std::memcpy(host_locations_.data(), copy.host_locations_.data(),
                            copy.host_locations_.size() * sizeof(Data64));
            }
        }

        Plaintext(Plaintext&& assign) noexcept
            : scheme_(std::move(assign.scheme_)),
              plain_size_(std::move(assign.plain_size_)),
              depth_(std::move(assign.depth_)),
              scale_(std::move(assign.scale_)),
              in_ntt_domain_(std::move(assign.in_ntt_domain_)),
              storage_type_(std::move(assign.storage_type_)),
              device_locations_(std::move(assign.device_locations_)),
              host_locations_(std::move(assign.host_locations_))
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
                storage_type_ = copy.storage_type_;

                if (copy.storage_type_ == storage_type::DEVICE)
                {
                    device_locations_.resize(copy.device_locations_.size(),
                                             copy.device_locations_.stream());
                    cudaMemcpyAsync(
                        device_locations_.data(), copy.device_locations_.data(),
                        copy.device_locations_.size() * sizeof(Data64),
                        cudaMemcpyDeviceToDevice,
                        copy.device_locations_
                            .stream()); // TODO: use cudaStreamPerThread
                }
                else
                {
                    std::memcpy(host_locations_.data(),
                                copy.host_locations_.data(),
                                copy.host_locations_.size() * sizeof(Data64));
                }
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
                storage_type_ = std::move(assign.storage_type_);

                device_locations_ = std::move(assign.device_locations_);
                host_locations_ = std::move(assign.host_locations_);
            }
            return *this;
        }

      private:
        scheme_type scheme_;

        int plain_size_;

        int depth_;

        double scale_;

        bool in_ntt_domain_;
        storage_type storage_type_;

        DeviceVector<Data64> device_locations_;
        HostVector<Data64> host_locations_;

        int memory_size();
        void memory_clear(cudaStream_t stream);
        void memory_set(DeviceVector<Data64>&& new_device_vector);

        void copy_to_device(cudaStream_t stream);
        void remove_from_device(cudaStream_t stream);
        void remove_from_host();
    };
} // namespace heongpu
#endif // PLAINTEXT_H
