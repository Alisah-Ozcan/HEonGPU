// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/ckks/ciphertext.cuh>

namespace heongpu
{

    __host__
    Ciphertext<Scheme::CKKS>::Ciphertext(HEContext<Scheme::CKKS>& context,
                                         const ExecutionOptions& options)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

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

    void Ciphertext<Scheme::CKKS>::store_in_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::store_in_host(cudaStream_t stream)
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

    Data64* Ciphertext<Scheme::CKKS>::data()
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

    void Ciphertext<Scheme::CKKS>::get_data(std::vector<Data64>& cipher,
                                            cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::get_data(HostVector<Data64>& cipher,
                                            cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::save(std::ostream& os) const
    {
        if (ciphertext_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &cipher_size_, sizeof(cipher_size_));

            os.write((char*) &depth_, sizeof(depth_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &scale_, sizeof(scale_));

            os.write((char*) &rescale_required_, sizeof(rescale_required_));

            os.write((char*) &relinearization_required_,
                     sizeof(relinearization_required_));

            os.write((char*) &ciphertext_generated_,
                     sizeof(ciphertext_generated_));

            if (storage_type_ == storage_type::DEVICE)
            {
                uint32_t ciphertext_memory_size =
                    cipher_size_ * (coeff_modulus_count_ - depth_) * ring_size_;
                HostVector<Data64> host_locations_temp(ciphertext_memory_size);
                cudaMemcpy(host_locations_temp.data(), device_locations_.data(),
                           ciphertext_memory_size * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
            else
            {
                uint32_t ciphertext_memory_size = host_locations_.size();
                os.write((char*) &ciphertext_memory_size,
                         sizeof(ciphertext_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * ciphertext_memory_size);
            }
        }
        else
        {
            throw std::runtime_error(
                "Ciphertext is not generated so can not be serialized!");
        }
    }

    void Ciphertext<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!ciphertext_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::ckks)
            {
                throw std::runtime_error("Invalid scheme binary!");
            }

            is.read((char*) &ring_size_, sizeof(ring_size_));

            is.read((char*) &coeff_modulus_count_,
                    sizeof(coeff_modulus_count_));

            is.read((char*) &cipher_size_, sizeof(cipher_size_));

            is.read((char*) &depth_, sizeof(depth_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            is.read((char*) &scale_, sizeof(scale_));

            is.read((char*) &rescale_required_, sizeof(rescale_required_));

            is.read((char*) &relinearization_required_,
                    sizeof(relinearization_required_));

            is.read((char*) &ciphertext_generated_,
                    sizeof(ciphertext_generated_));

            storage_type_ = storage_type::DEVICE;
            ciphertext_generated_ = true;

            uint32_t ciphertext_memory_size;
            is.read((char*) &ciphertext_memory_size,
                    sizeof(ciphertext_memory_size));

            if (ciphertext_memory_size !=
                (cipher_size_ * ring_size_ * (coeff_modulus_count_ - depth_)))
            {
                throw std::runtime_error("Invalid ciphertext size!");
            }

            HostVector<Data64> host_locations_temp(ciphertext_memory_size);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * ciphertext_memory_size);

            device_locations_.resize(ciphertext_memory_size);
            cudaMemcpy(device_locations_.data(), host_locations_temp.data(),
                       ciphertext_memory_size * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Ciphertext has been already exist!");
        }
    }

    int Ciphertext<Scheme::CKKS>::memory_size()
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

    void Ciphertext<Scheme::CKKS>::memory_clear(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::memory_set(
        DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Ciphertext<Scheme::CKKS>::copy_to_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::remove_from_device(cudaStream_t stream)
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

    void Ciphertext<Scheme::CKKS>::remove_from_host()
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