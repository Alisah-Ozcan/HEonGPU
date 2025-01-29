// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ciphertext.cuh"

namespace heongpu
{
    __host__ Ciphertext::Ciphertext() {}

    __host__ Ciphertext::Ciphertext(Parameters& context,
                                    const ExecutionOptions& options)
    {
        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2;
        ring_size_ = context.n;
        depth_ = 0;

        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        storage_type_ = options.storage_;

        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_ =
                DeviceVector<Data64>(cipher_memory_size, options.stream_);
        }
        else
        {
            host_locations_ = HostVector<Data64>(cipher_memory_size);
        }
    }

    __host__ Ciphertext::Ciphertext(const std::vector<Data64>& cipher,
                                    Parameters& context,
                                    const ExecutionOptions& options)
    {
        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2;
        ring_size_ = context.n;
        depth_ = 0;

        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        storage_type_ = options.storage_;

        if (!(cipher.size() == cipher_memory_size))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_ =
                DeviceVector<Data64>(cipher_memory_size, options.stream_);

            cudaMemcpyAsync(device_locations_.data(), cipher.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyHostToDevice, options.stream_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            host_locations_ = HostVector<Data64>(cipher_memory_size);
            std::memcpy(host_locations_.data(), cipher.data(),
                        cipher.size() * sizeof(Data64));
        }
    }

    __host__ Ciphertext::Ciphertext(const HostVector<Data64>& cipher,
                                    Parameters& context,
                                    const ExecutionOptions& options)
    {
        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_size;
        cipher_size_ = 2; // default
        ring_size_ = context.n; // n
        depth_ = 0;

        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        in_ntt_domain_ =
            (static_cast<int>(scheme_) == static_cast<int>(scheme_type::ckks))
                ? true
                : false;

        rescale_required_ = false;
        relinearization_required_ = false;
        scale_ = 0;

        storage_type_ = options.storage_;

        if (!(cipher.size() == cipher_memory_size))
        {
            throw std::invalid_argument("Ciphertext size should be valid!");
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_ =
                DeviceVector<Data64>(cipher_memory_size, options.stream_);

            cudaMemcpyAsync(device_locations_.data(), cipher.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyHostToDevice, options.stream_);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            host_locations_ = cipher;
        }
    }

    void Ciphertext::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                device_locations_ =
                    DeviceVector<Data64>(host_locations_, stream);
                host_locations_.resize(0);
                host_locations_.shrink_to_fit();
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Ciphertext::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                int cipher_memory_size =
                    cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;
                host_locations_ = HostVector<Data64>(cipher_memory_size);
                cudaMemcpyAsync(host_locations_.data(),
                                device_locations_.data(),
                                cipher_memory_size * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                device_locations_.resize(0, stream);
                device_locations_.shrink_to_fit(stream);
            }

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    Data64* Ciphertext::data()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_locations_.data();
        }
        else
        {
            return host_locations_.data();
        }
    }

    void Ciphertext::get_data(std::vector<Data64>& cipher, cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    void Ciphertext::get_data(HostVector<Data64>& cipher, cudaStream_t stream)
    {
        int cipher_memory_size =
            cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;

        if (cipher.size() < cipher_memory_size)
        {
            cipher.resize(cipher_memory_size);
        }

        if (storage_type_ == storage_type::DEVICE)
        {
            cudaMemcpyAsync(cipher.data(), device_locations_.data(),
                            cipher_memory_size * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
        }
        else
        {
            std::memcpy(cipher.data(), host_locations_.data(),
                        host_locations_.size() * sizeof(Data64));
        }
    }

    int Ciphertext::memory_size()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_locations_.size();
        }
        else
        {
            return host_locations_.size();
        }
    }

    void Ciphertext::memory_clear(cudaStream_t stream)
    {
        if (device_locations_.size() > 0)
        {
            device_locations_.resize(0, stream);
            device_locations_.shrink_to_fit(stream);
        }

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Ciphertext::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Ciphertext::copy_to_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                device_locations_ =
                    DeviceVector<Data64>(host_locations_, stream);
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Ciphertext::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            device_locations_.resize(0, stream);
            device_locations_.shrink_to_fit(stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Ciphertext::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu