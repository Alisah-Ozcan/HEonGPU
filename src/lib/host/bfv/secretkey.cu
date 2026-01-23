// Copyright 2024-2026 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include <heongpu/host/bfv/secretkey.cuh>

namespace heongpu
{
    __host__ Secretkey<Scheme::BFV>::Secretkey(HEContext<Scheme::BFV>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n
        n_power_ = context.n_power;

        hamming_weight_ = ring_size_ >> 1; // default
        in_ntt_domain_ = false;

        storage_type_ = storage_type::DEVICE;
    }

    __host__ Secretkey<Scheme::BFV>::Secretkey(HEContext<Scheme::BFV>& context,
                                               int hamming_weight)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n

        hamming_weight_ = hamming_weight;
        if ((hamming_weight_ <= 0) || (hamming_weight_ > ring_size_))
        {
            throw std::invalid_argument(
                "Hamming weight has to be in range 0 to ring size.");
        }

        in_ntt_domain_ = false;

        storage_type_ = storage_type::DEVICE;
    }

    __host__
    Secretkey<Scheme::BFV>::Secretkey(const std::vector<int>& secret_key,
                                      HEContext<Scheme::BFV>& context,
                                      cudaStream_t stream)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n
        n_power_ = context.n_power;

        hamming_weight_ = ring_size_ >> 1; // default

        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        DeviceVector<int> secret_key_device(ring_size_, stream);
        cudaMemcpyAsync(secret_key_device.data(), secret_key.data(),
                        ring_size_ * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (device_locations_.size() < coeff_modulus_count_ * ring_size_)
        {
            device_locations_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secret_key_device.data(), device_locations_.data(),
            context.modulus_->data(), n_power_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(device_locations_.data(),
                                context.ntt_table_->data(),
                                context.modulus_->data(), cfg_ntt,
                                coeff_modulus_count_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        in_ntt_domain_ = true;
        storage_type_ = storage_type::DEVICE;
        secret_key_generated_ = true;
    }

    __host__
    Secretkey<Scheme::BFV>::Secretkey(const HostVector<int>& secret_key,
                                      HEContext<Scheme::BFV>& context,
                                      cudaStream_t stream)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        coeff_modulus_count_ = context.Q_prime_size;
        ring_size_ = context.n; // n
        n_power_ = context.n_power;

        hamming_weight_ = ring_size_ >> 1; // default

        if (!(secret_key.size() == ring_size_))
        {
            throw std::invalid_argument("Secretkey size should be valid!");
        }

        DeviceVector<int> secret_key_device(secret_key, stream);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        if (device_locations_.size() < coeff_modulus_count_ * ring_size_)
        {
            device_locations_ =
                DeviceVector<Data64>(coeff_modulus_count_ * ring_size_, stream);
        }

        secretkey_rns_kernel<<<dim3((ring_size_ >> 8), 1, 1), 256, 0, stream>>>(
            secret_key_device.data(), device_locations_.data(),
            context.modulus_->data(), n_power_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        gpuntt::ntt_rns_configuration<Data64> cfg_ntt = {
            .n_power = n_power_,
            .ntt_type = gpuntt::FORWARD,
            .ntt_layout = gpuntt::PerPolynomial,
            .reduction_poly = gpuntt::ReductionPolynomial::X_N_plus,
            .zero_padding = false,
            .stream = stream};

        gpuntt::GPU_NTT_Inplace(device_locations_.data(),
                                context.ntt_table_->data(),
                                context.modulus_->data(), cfg_ntt,
                                coeff_modulus_count_, coeff_modulus_count_);
        HEONGPU_CUDA_CHECK(cudaGetLastError());

        in_ntt_domain_ = true;
        storage_type_ = storage_type::DEVICE;
        secret_key_generated_ = true;
    }

    Data64* Secretkey<Scheme::BFV>::data()
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

    void Secretkey<Scheme::BFV>::store_in_device(cudaStream_t stream)
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

    void Secretkey<Scheme::BFV>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (memory_size() == 0)
            {
                // pass
            }
            else
            {
                int secretkey_memory_size = coeff_modulus_count_ * ring_size_;
                host_locations_ = HostVector<Data64>(secretkey_memory_size);
                cudaMemcpyAsync(host_locations_.data(),
                                device_locations_.data(),
                                secretkey_memory_size * sizeof(Data64),
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

    void Secretkey<Scheme::BFV>::save(std::ostream& os) const
    {
        if (secret_key_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &ring_size_, sizeof(ring_size_));

            os.write((char*) &coeff_modulus_count_,
                     sizeof(coeff_modulus_count_));

            os.write((char*) &n_power_, sizeof(n_power_));

            os.write((char*) &hamming_weight_, sizeof(hamming_weight_));

            os.write((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            os.write((char*) &secret_key_generated_,
                     sizeof(secret_key_generated_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            if (storage_type_ == storage_type::DEVICE)
            {
                uint32_t secretkey_memory_size =
                    coeff_modulus_count_ * ring_size_;
                HostVector<Data64> host_locations_temp(secretkey_memory_size);
                cudaMemcpy(host_locations_temp.data(), device_locations_.data(),
                           secretkey_memory_size * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) &secretkey_memory_size,
                         sizeof(secretkey_memory_size));
                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * secretkey_memory_size);
            }
            else
            {
                uint32_t secretkey_memory_size = host_locations_.size();
                os.write((char*) &secretkey_memory_size,
                         sizeof(secretkey_memory_size));
                os.write((char*) host_locations_.data(),
                         sizeof(Data64) * secretkey_memory_size);
            }
        }
        else
        {
            throw std::runtime_error(
                "Secretkey is not generated so can not be serialized!");
        }
    }

    void Secretkey<Scheme::BFV>::load(std::istream& is)
    {
        if ((!secret_key_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::bfv)
            {
                throw std::runtime_error("Invalid scheme binary!");
            }

            is.read((char*) &ring_size_, sizeof(ring_size_));

            is.read((char*) &coeff_modulus_count_,
                    sizeof(coeff_modulus_count_));

            is.read((char*) &n_power_, sizeof(n_power_));

            is.read((char*) &hamming_weight_, sizeof(hamming_weight_));

            is.read((char*) &in_ntt_domain_, sizeof(in_ntt_domain_));

            is.read((char*) &secret_key_generated_,
                    sizeof(secret_key_generated_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            storage_type_ = storage_type::DEVICE;
            secret_key_generated_ = true;

            uint32_t secretkey_memory_size;
            is.read((char*) &secretkey_memory_size,
                    sizeof(secretkey_memory_size));

            if (secretkey_memory_size != (ring_size_ * coeff_modulus_count_))
            {
                throw std::runtime_error("Invalid secretkey size!");
            }

            HostVector<Data64> host_locations_temp(secretkey_memory_size);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * secretkey_memory_size);

            device_locations_.resize(secretkey_memory_size);
            cudaMemcpy(device_locations_.data(), host_locations_temp.data(),
                       secretkey_memory_size * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Secretkey has been already exist!");
        }
    }

    int Secretkey<Scheme::BFV>::memory_size()
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

    void Secretkey<Scheme::BFV>::memory_clear(cudaStream_t stream)
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

    void
    Secretkey<Scheme::BFV>::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_locations_ = std::move(new_device_vector);

        if (host_locations_.size() > 0)
        {
            host_locations_.resize(0);
            host_locations_.shrink_to_fit();
        }
    }

    void Secretkey<Scheme::BFV>::copy_to_device(cudaStream_t stream)
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

    void Secretkey<Scheme::BFV>::remove_from_device(cudaStream_t stream)
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

    void Secretkey<Scheme::BFV>::remove_from_host()
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