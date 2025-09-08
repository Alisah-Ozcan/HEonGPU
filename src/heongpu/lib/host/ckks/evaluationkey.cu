// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "ckks/evaluationkey.cuh"

namespace heongpu
{

    __host__ Relinkey<Scheme::CKKS>::Relinkey(HEContext<Scheme::CKKS>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                relinkey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
            }
            break;
            case 2: // KEYSWITCHING_METHOD_II
            {
                d_ = context.d_leveled->operator[](0);
                relinkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III
            {
                d_ = context.d_leveled->operator[](0);
                d_tilda_ = context.d_tilda_leveled->operator[](0);
                r_prime_ = context.r_prime_leveled;

                int max_depth = Q_size_ - 1;
                for (int i = 0; i < max_depth; i++)
                {
                    relinkey_size_leveled_.push_back(
                        2 * context.d_leveled->operator[](i) *
                        context.d_tilda_leveled->operator[](i) * r_prime_ *
                        ring_size);
                }
            }
            break;
            default:
                break;
        }
    }

    void Relinkey<Scheme::CKKS>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if ((key_type == keyswitching_type::KEYSWITCHING_METHOD_III) &&
                scheme_ == scheme_type::ckks)
            {
                int max_depth = Q_size_ - 1;
                for (int i = 0; i < max_depth; i++)
                {
                    device_location_leveled_.push_back(
                        std::move(DeviceVector<Data64>(
                            host_location_leveled_[i], stream)));
                    host_location_leveled_[i].resize(0);
                    host_location_leveled_[i].shrink_to_fit();
                }
                host_location_leveled_.resize(0);
                host_location_leveled_.shrink_to_fit();
            }
            else
            {
                device_location_ = DeviceVector<Data64>(host_location_, stream);
                host_location_.resize(0);
                host_location_.shrink_to_fit();
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Relinkey<Scheme::CKKS>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if ((key_type == keyswitching_type::KEYSWITCHING_METHOD_III) &&
                scheme_ == scheme_type::ckks)
            {
                int max_depth = Q_size_ - 1;
                for (int i = 0; i < max_depth; i++)
                {
                    host_location_leveled_.push_back(
                        HostVector<Data64>(relinkey_size_leveled_[i]));

                    cudaMemcpyAsync(host_location_leveled_[i].data(),
                                    device_location_leveled_[i].data(),
                                    relinkey_size_leveled_[i] * sizeof(Data64),
                                    cudaMemcpyDeviceToHost, stream);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());

                    device_location_leveled_[i].resize(0, stream);
                }
                device_location_leveled_.resize(0);
                device_location_leveled_.shrink_to_fit();
            }
            else
            {
                host_location_ = HostVector<Data64>(relinkey_size_);
                cudaMemcpyAsync(host_location_.data(), device_location_.data(),
                                relinkey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                device_location_.resize(0, stream);
            }

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    Data64* Relinkey<Scheme::CKKS>::data()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_location_.data();
        }
        else
        {
            return host_location_.data();
        }
    }
    Data64* Relinkey<Scheme::CKKS>::data(size_t i)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_location_leveled_[i].data();
        }
        else
        {
            return host_location_leveled_[i].data();
        }
    }

    void Relinkey<Scheme::CKKS>::save(std::ostream& os) const
    {
        if (key_type == keyswitching_type::KEYSWITCHING_METHOD_III)
        {
            throw std::runtime_error(
                "Relinkey has not serialization for KEYSWITCHING_METHOD_III!");
        }

        if (relin_key_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &key_type, sizeof(key_type));

            os.write((char*) &ring_size, sizeof(ring_size));

            os.write((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            os.write((char*) &Q_size_, sizeof(Q_size_));

            os.write((char*) &d_, sizeof(d_));

            os.write((char*) &d_tilda_, sizeof(d_tilda_));

            os.write((char*) &r_prime_, sizeof(r_prime_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &relin_key_generated_,
                     sizeof(relin_key_generated_));

            os.write((char*) &relinkey_size_, sizeof(relinkey_size_));

            if (storage_type_ == storage_type::DEVICE)
            {
                HostVector<Data64> host_locations_temp(relinkey_size_);
                cudaMemcpy(host_locations_temp.data(), device_location_.data(),
                           relinkey_size_ * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * relinkey_size_);
            }
            else
            {
                os.write((char*) host_location_.data(),
                         sizeof(Data64) * relinkey_size_);
            }
        }
        else
        {
            throw std::runtime_error(
                "Relinkey is not generated so can not be serialized!");
        }
    }

    void Relinkey<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!relin_key_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::ckks)
            {
                throw std::runtime_error("Invalid scheme binary!");
            }

            is.read((char*) &key_type, sizeof(key_type));

            is.read((char*) &ring_size, sizeof(ring_size));

            is.read((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            is.read((char*) &Q_size_, sizeof(Q_size_));

            is.read((char*) &d_, sizeof(d_));

            is.read((char*) &d_tilda_, sizeof(d_tilda_));

            is.read((char*) &r_prime_, sizeof(r_prime_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            is.read((char*) &relin_key_generated_,
                    sizeof(relin_key_generated_));

            is.read((char*) &relinkey_size_, sizeof(relinkey_size_));

            storage_type_ = storage_type::DEVICE;
            relin_key_generated_ = true;

            HostVector<Data64> host_locations_temp(relinkey_size_);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * relinkey_size_);

            device_location_.resize(relinkey_size_);
            cudaMemcpy(device_location_.data(), host_locations_temp.data(),
                       relinkey_size_ * sizeof(Data64), cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Relinkey has been already exist!");
        }
    }

    int Relinkey<Scheme::CKKS>::memory_size()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (relinkey_size_leveled_.size() == 0)
            {
                return device_location_.size();
            }
            else
            {
                return device_location_leveled_[0].size();
            }
        }
        else
        {
            if (relinkey_size_leveled_.size() == 0)
            {
                return host_location_.size();
            }
            else
            {
                return host_location_leveled_[0].size();
            }
        }
    }

    void Relinkey<Scheme::CKKS>::memory_clear(cudaStream_t stream)
    {
        if (device_location_.size() > 0)
        {
            device_location_.resize(0, stream);
            device_location_.shrink_to_fit(stream);
        }

        for (int i = 0; i < device_location_leveled_.size(); i++)
        {
            device_location_leveled_[i].resize(0, stream);
            device_location_leveled_[i].shrink_to_fit(stream);
        }

        device_location_leveled_.resize(0);
        device_location_leveled_.shrink_to_fit();

        if (host_location_.size() > 0)
        {
            host_location_.resize(0);
            host_location_.shrink_to_fit();
        }

        for (int i = 0; i < host_location_leveled_.size(); i++)
        {
            host_location_leveled_[i].resize(0);
            host_location_leveled_[i].shrink_to_fit();
        }

        host_location_leveled_.resize(0);
        host_location_leveled_.shrink_to_fit();
    }

    void
    Relinkey<Scheme::CKKS>::memory_set(DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_location_ = std::move(new_device_vector);

        if (host_location_.size() > 0)
        {
            host_location_.resize(0);
            host_location_.shrink_to_fit();
        }
    }

    void
    Relinkey<Scheme::CKKS>::memory_set(DeviceVector<Data64>&& new_device_vector,
                                       int i)
    {
        storage_type_ = storage_type::DEVICE;
        device_location_leveled_[i] = std::move(new_device_vector);

        for (int i = 0; i < host_location_leveled_.size(); i++)
        {
            host_location_leveled_[i].resize(0);
            host_location_leveled_[i].shrink_to_fit();
        }

        host_location_leveled_.resize(0);
        host_location_leveled_.shrink_to_fit();
    }

    void Relinkey<Scheme::CKKS>::copy_to_device(cudaStream_t stream)
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
                if (relinkey_size_leveled_.size() == 0)
                {
                    device_location_ =
                        DeviceVector<Data64>(host_location_, stream);
                }
                else
                {
                    device_location_leveled_.resize(
                        relinkey_size_leveled_.size());

                    for (int i = 0; i < device_location_leveled_.size(); i++)
                    {
                        device_location_leveled_[i] = DeviceVector<Data64>(
                            host_location_leveled_[i], stream);
                    }
                }
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Relinkey<Scheme::CKKS>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            if (relinkey_size_leveled_.size() == 0)
            {
                device_location_.resize(0, stream);
                device_location_.shrink_to_fit(stream);
            }
            else
            {
                for (int i = 0; i < device_location_leveled_.size(); i++)
                {
                    device_location_leveled_[i].resize(0, stream);
                    device_location_leveled_[i].shrink_to_fit(stream);
                }

                device_location_leveled_.resize(0);
                device_location_leveled_.shrink_to_fit();
            }
        }
        else
        {
            // pass
        }
    }

    void Relinkey<Scheme::CKKS>::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            if (relinkey_size_leveled_.size() == 0)
            {
                host_location_.resize(0);
                host_location_.shrink_to_fit();
            }
            else
            {
                for (int i = 0; i < host_location_leveled_.size(); i++)
                {
                    host_location_leveled_[i].resize(0);
                    host_location_leveled_[i].shrink_to_fit();
                }

                host_location_leveled_.resize(0);
                host_location_leveled_.shrink_to_fit();
            }
        }
    }

    __host__ MultipartyRelinkey<Scheme::CKKS>::MultipartyRelinkey(
        HEContext<Scheme::CKKS>& context, const RNGSeed seed)
        : Relinkey(context), seed_(seed)
    {
    }

    __host__
    Galoiskey<Scheme::CKKS>::Galoiskey(HEContext<Scheme::CKKS>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        customized = false;

        group_order_ = 5;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                galoiskey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;

                for (int i = 0; i < MAX_SHIFT; i++)
                {
                    int power = pow(2, i);
                    galois_elt[power] =
                        steps_to_galois_elt(power, ring_size, group_order_);
                    galois_elt[(-power)] =
                        steps_to_galois_elt((-power), ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                max_shift_ = MAX_SHIFT - 1;
                max_log_slot_ = int(std::log2(ring_size >> 1));
            }
            break;
            case 2: // KEYSWITCHING_METHOD_II
            {
                for (int i = 0; i < MAX_SHIFT; i++)
                {
                    int power = pow(2, i);
                    galois_elt[power] =
                        steps_to_galois_elt(power, ring_size, group_order_);
                    galois_elt[(-power)] =
                        steps_to_galois_elt((-power), ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                max_shift_ = MAX_SHIFT - 1;
                max_log_slot_ = int(std::log2(ring_size >> 1));

                d_ = context.d_leveled->operator[](0);
                galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__
    Galoiskey<Scheme::CKKS>::Galoiskey(HEContext<Scheme::CKKS>& context, int max_shift)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        customized = false;

        group_order_ = 5;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                galoiskey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;

                for (int i = 0; i < max_shift + 1; i++)
                {
                    int power = pow(2, i);
                    galois_elt[power] =
                        steps_to_galois_elt(power, ring_size, group_order_);
                    galois_elt[(-power)] =
                        steps_to_galois_elt((-power), ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                max_shift_ = max_shift;
                max_log_slot_ = int(std::log2(ring_size >> 1));
            }
            break;
            case 2: // KEYSWITCHING_METHOD_II
            {
                for (int i = 0; i < max_shift + 1; i++)
                {
                    int power = pow(2, i);
                    galois_elt[power] =
                        steps_to_galois_elt(power, ring_size, group_order_);
                    galois_elt[(-power)] =
                        steps_to_galois_elt((-power), ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                max_shift_ = max_shift;
                max_log_slot_ = int(std::log2(ring_size >> 1));

                d_ = context.d_leveled->operator[](0);
                galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__
    Galoiskey<Scheme::CKKS>::Galoiskey(HEContext<Scheme::CKKS>& context,
                                       std::vector<int>& shift_vec)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        customized = false;

        group_order_ = 5;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                galoiskey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;

                for (int shift : shift_vec)
                {
                    galois_elt[shift] =
                        steps_to_galois_elt(shift, ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                break;
            }
            case 2: // KEYSWITCHING_METHOD_II
            {
                for (int shift : shift_vec)
                {
                    galois_elt[shift] =
                        steps_to_galois_elt(shift, ring_size, group_order_);
                }

                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);

                d_ = context.d_leveled->operator[](0);
                galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__
    Galoiskey<Scheme::CKKS>::Galoiskey(HEContext<Scheme::CKKS>& context,
                                       std::vector<uint32_t>& galois_elts)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        customized = true;

        group_order_ = 5;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);
                galoiskey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
                custom_galois_elt = galois_elts;
            }
            break;
            case 2: // KEYSWITCHING_METHOD_II
            {
                d_ = context.d_leveled->operator[](0);
                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);
                galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                custom_galois_elt = galois_elts;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    void Galoiskey<Scheme::CKKS>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            for (const auto& galois_ : host_location_)
            {
                device_location_[galois_.first] =
                    DeviceVector<Data64>(galois_.second, stream);
            }

            zero_device_location_ =
                DeviceVector<Data64>(zero_host_location_, stream);

            host_location_.clear();
            zero_host_location_.resize(0);
            zero_host_location_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Galoiskey<Scheme::CKKS>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            for (auto& galois_ : device_location_)
            {
                host_location_[galois_.first] =
                    HostVector<Data64>(galoiskey_size_);
                cudaMemcpyAsync(host_location_[galois_.first].data(),
                                galois_.second.data(),
                                galoiskey_size_ * sizeof(Data64),
                                cudaMemcpyDeviceToHost, stream);
                HEONGPU_CUDA_CHECK(cudaGetLastError());

                galois_.second.resize(0, stream);
            }

            zero_host_location_ = HostVector<Data64>(galoiskey_size_);
            cudaMemcpyAsync(zero_host_location_.data(),
                            zero_device_location_.data(),
                            galoiskey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            device_location_.clear();
            zero_device_location_.resize(0);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    Data64* Galoiskey<Scheme::CKKS>::data(size_t i)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_location_[i].data();
        }
        else
        {
            return host_location_[i].data();
        }
    }

    Data64* Galoiskey<Scheme::CKKS>::c_data()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return zero_device_location_.data();
        }
        else
        {
            return zero_host_location_.data();
        }
    }

    void Galoiskey<Scheme::CKKS>::save(std::ostream& os) const
    {
        if (galois_key_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &key_type, sizeof(key_type));

            os.write((char*) &ring_size, sizeof(ring_size));

            os.write((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            os.write((char*) &Q_size_, sizeof(Q_size_));

            os.write((char*) &d_, sizeof(d_));

            os.write((char*) &customized, sizeof(customized));

            os.write((char*) &group_order_, sizeof(group_order_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &galois_key_generated_,
                     sizeof(galois_key_generated_));

            if (customized)
            {
                uint32_t custom_galois_elt_size = custom_galois_elt.size();
                os.write((char*) &custom_galois_elt_size,
                         sizeof(custom_galois_elt_size));
                os.write((char*) custom_galois_elt.data(),
                         sizeof(u_int32_t) * custom_galois_elt_size);
            }
            else
            {
                uint32_t galois_elt_size = galois_elt.size();
                os.write((char*) &galois_elt_size, sizeof(galois_elt_size));
                for (auto& galois : galois_elt)
                {
                    os.write((char*) &galois.first, sizeof(galois.first));
                    os.write((char*) &galois.second, sizeof(galois.second));
                }
            }

            os.write((char*) &galois_elt_zero, sizeof(galois_elt_zero));

            os.write((char*) &galoiskey_size_, sizeof(galoiskey_size_));

            if (storage_type_ == storage_type::DEVICE)
            {
                uint32_t key_count = device_location_.size();
                os.write((char*) &key_count, sizeof(key_count));

                for (auto& galois_key_mem : device_location_)
                {
                    HostVector<Data64> host_locations_temp(galoiskey_size_);
                    cudaMemcpy(host_locations_temp.data(),
                               galois_key_mem.second.data(),
                               galoiskey_size_ * sizeof(Data64),
                               cudaMemcpyDeviceToHost);
                    HEONGPU_CUDA_CHECK(cudaGetLastError());
                    cudaDeviceSynchronize();

                    os.write((char*) &galois_key_mem.first,
                             sizeof(galois_key_mem.first));
                    os.write((char*) host_locations_temp.data(),
                             sizeof(Data64) * galoiskey_size_);
                }

                HostVector<Data64> host_locations_temp(galoiskey_size_);
                cudaMemcpy(
                    host_locations_temp.data(), zero_device_location_.data(),
                    galoiskey_size_ * sizeof(Data64), cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * galoiskey_size_);
            }
            else
            {
                uint32_t key_count = host_location_.size();
                os.write((char*) &key_count, sizeof(key_count));

                for (auto& galois_key_mem : host_location_)
                {
                    os.write((char*) &galois_key_mem.first,
                             sizeof(galois_key_mem.first));
                    os.write((char*) galois_key_mem.second.data(),
                             sizeof(Data64) * galoiskey_size_);
                }

                os.write((char*) zero_host_location_.data(),
                         sizeof(Data64) * galoiskey_size_);
            }
        }
        else
        {
            throw std::runtime_error(
                "Galoiskey is not generated so can not be serialized!");
        }
    }

    void Galoiskey<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!galois_key_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            if (scheme_ != scheme_type::ckks)
            {
                throw std::runtime_error("Invalid scheme binary!");
            }

            is.read((char*) &key_type, sizeof(key_type));

            is.read((char*) &ring_size, sizeof(ring_size));

            is.read((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            is.read((char*) &Q_size_, sizeof(Q_size_));

            is.read((char*) &d_, sizeof(d_));

            is.read((char*) &customized, sizeof(customized));

            is.read((char*) &group_order_, sizeof(group_order_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            is.read((char*) &galois_key_generated_,
                    sizeof(galois_key_generated_));

            storage_type_ = storage_type::DEVICE;
            galois_key_generated_ = true;

            if (customized)
            {
                uint32_t custom_galois_elt_size;
                is.read((char*) &custom_galois_elt_size,
                        custom_galois_elt_size);
                custom_galois_elt.resize(custom_galois_elt_size);
                is.read((char*) custom_galois_elt.data(),
                        sizeof(u_int32_t) * custom_galois_elt_size);
            }
            else
            {
                uint32_t galois_elt_size;
                is.read((char*) &galois_elt_size, sizeof(galois_elt_size));
                for (int i = 0; i < galois_elt_size; i++)
                {
                    int first;
                    int second;
                    is.read((char*) &first, sizeof(first));
                    is.read((char*) &second, sizeof(second));
                    galois_elt[first] = second;
                }
            }

            is.read((char*) &galois_elt_zero, sizeof(galois_elt_zero));

            is.read((char*) &galoiskey_size_, sizeof(galoiskey_size_));

            uint32_t key_count;
            is.read((char*) &key_count, sizeof(key_count));

            for (int i = 0; i < key_count; i++)
            {
                int first;
                is.read((char*) &first, sizeof(first));
                HostVector<Data64> host_locations_temp(galoiskey_size_);
                is.read((char*) host_locations_temp.data(),
                        sizeof(Data64) * galoiskey_size_);
                device_location_[first] =
                    DeviceVector<Data64>(host_locations_temp);
                cudaDeviceSynchronize();
            }

            HostVector<Data64> host_locations_temp(galoiskey_size_);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * galoiskey_size_);

            zero_device_location_.resize(galoiskey_size_);
            cudaMemcpy(zero_device_location_.data(), host_locations_temp.data(),
                       galoiskey_size_ * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Galoiskey has been already exist!");
        }
    }

    __host__ MultipartyGaloiskey<Scheme::CKKS>::MultipartyGaloiskey(
        HEContext<Scheme::CKKS>& context, const RNGSeed seed)
        : Galoiskey(context), seed_(seed)
    {
    }

    __host__ MultipartyGaloiskey<Scheme::CKKS>::MultipartyGaloiskey(
        HEContext<Scheme::CKKS>& context, std::vector<int>& shift_vec,
        const RNGSeed seed)
        : Galoiskey(context, shift_vec), seed_(seed)
    {
    }

    __host__ MultipartyGaloiskey<Scheme::CKKS>::MultipartyGaloiskey(
        HEContext<Scheme::CKKS>& context, std::vector<uint32_t>& galois_elts,
        const RNGSeed seed)
        : Galoiskey(context, galois_elts), seed_(seed)
    {
    }

    __host__
    Switchkey<Scheme::CKKS>::Switchkey(HEContext<Scheme::CKKS>& context)
    {
        if (!context.context_generated_)
        {
            throw std::invalid_argument("HEContext is not generated!");
        }

        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;

        ring_size = context.n;
        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                switchkey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
            }
            break;
            case 2: // KEYSWITCHING_METHOD_II
            {
                d_ = context.d_leveled->operator[](0);
                switchkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
            }
            break;
            case 3: // KEYSWITCHING_METHOD_III Galoiskey
                throw std::invalid_argument(
                    "Switchkey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    void Switchkey<Scheme::CKKS>::store_in_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            device_location_ = DeviceVector<Data64>(host_location_, stream);
            host_location_.resize(0);
            host_location_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Switchkey<Scheme::CKKS>::store_in_host(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            host_location_ = HostVector<Data64>(switchkey_size_);
            cudaMemcpyAsync(host_location_.data(), device_location_.data(),
                            switchkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            device_location_.resize(0, stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    Data64* Switchkey<Scheme::CKKS>::data()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_location_.data();
        }
        else
        {
            return host_location_.data();
        }
    }

    void Switchkey<Scheme::CKKS>::save(std::ostream& os) const
    {
        if (switch_key_generated_)
        {
            os.write((char*) &scheme_, sizeof(scheme_));

            os.write((char*) &key_type, sizeof(key_type));

            os.write((char*) &ring_size, sizeof(ring_size));

            os.write((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            os.write((char*) &Q_size_, sizeof(Q_size_));

            os.write((char*) &d_, sizeof(d_));

            os.write((char*) &storage_type_, sizeof(storage_type_));

            os.write((char*) &switch_key_generated_,
                     sizeof(switch_key_generated_));

            os.write((char*) &switchkey_size_, sizeof(switchkey_size_));

            if (storage_type_ == storage_type::DEVICE)
            {
                HostVector<Data64> host_locations_temp(switchkey_size_);
                cudaMemcpy(host_locations_temp.data(), device_location_.data(),
                           switchkey_size_ * sizeof(Data64),
                           cudaMemcpyDeviceToHost);
                HEONGPU_CUDA_CHECK(cudaGetLastError());
                cudaDeviceSynchronize();

                os.write((char*) host_locations_temp.data(),
                         sizeof(Data64) * switchkey_size_);
            }
            else
            {
                os.write((char*) host_location_.data(),
                         sizeof(Data64) * switchkey_size_);
            }
        }
        else
        {
            throw std::runtime_error(
                "Switchkey is not generated so can not be serialized!");
        }
    }

    void Switchkey<Scheme::CKKS>::load(std::istream& is)
    {
        if ((!switch_key_generated_))
        {
            is.read((char*) &scheme_, sizeof(scheme_));

            is.read((char*) &key_type, sizeof(key_type));

            is.read((char*) &ring_size, sizeof(ring_size));

            is.read((char*) &Q_prime_size_, sizeof(Q_prime_size_));

            is.read((char*) &Q_size_, sizeof(Q_size_));

            is.read((char*) &d_, sizeof(d_));

            is.read((char*) &storage_type_, sizeof(storage_type_));

            is.read((char*) &switch_key_generated_,
                    sizeof(switch_key_generated_));

            is.read((char*) &switchkey_size_, sizeof(switchkey_size_));

            storage_type_ = storage_type::DEVICE;
            switch_key_generated_ = true;

            HostVector<Data64> host_locations_temp(switchkey_size_);
            is.read((char*) host_locations_temp.data(),
                    sizeof(Data64) * switchkey_size_);

            cudaMemcpy(device_location_.data(), host_locations_temp.data(),
                       switchkey_size_ * sizeof(Data64),
                       cudaMemcpyHostToDevice);
            HEONGPU_CUDA_CHECK(cudaGetLastError());
            cudaDeviceSynchronize();
        }
        else
        {
            throw std::runtime_error("Switchkey has been already exist!");
        }
    }

    int Switchkey<Scheme::CKKS>::memory_size()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            return device_location_.size();
        }
        else
        {
            return host_location_.size();
        }
    }

    void Switchkey<Scheme::CKKS>::memory_clear(cudaStream_t stream)
    {
        if (device_location_.size() > 0)
        {
            device_location_.resize(0, stream);
            device_location_.shrink_to_fit(stream);
        }

        if (host_location_.size() > 0)
        {
            host_location_.resize(0);
            host_location_.shrink_to_fit();
        }
    }

    void Switchkey<Scheme::CKKS>::memory_set(
        DeviceVector<Data64>&& new_device_vector)
    {
        storage_type_ = storage_type::DEVICE;
        device_location_ = std::move(new_device_vector);

        if (host_location_.size() > 0)
        {
            host_location_.resize(0);
            host_location_.shrink_to_fit();
        }
    }

    void Switchkey<Scheme::CKKS>::copy_to_device(cudaStream_t stream)
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
                device_location_ = DeviceVector<Data64>(host_location_, stream);
            }

            storage_type_ = storage_type::DEVICE;
        }
    }

    void Switchkey<Scheme::CKKS>::remove_from_device(cudaStream_t stream)
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            device_location_.resize(0, stream);
            device_location_.shrink_to_fit(stream);

            storage_type_ = storage_type::HOST;
        }
        else
        {
            // pass
        }
    }

    void Switchkey<Scheme::CKKS>::remove_from_host()
    {
        if (storage_type_ == storage_type::DEVICE)
        {
            // pass
        }
        else
        {
            host_location_.resize(0);
            host_location_.shrink_to_fit();

            storage_type_ = storage_type::DEVICE;
        }
    }

} // namespace heongpu