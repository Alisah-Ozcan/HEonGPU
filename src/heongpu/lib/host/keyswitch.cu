// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "keyswitch.cuh"

namespace heongpu
{
    __host__ Relinkey::Relinkey(Parameters& context, bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_ = context.d;
        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                // It can use for both leveled and no leveled.
                relinkey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
                break;
            }
            case 2: // KEYSWITCHING_METHOD_II

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    relinkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
                    d_ = context.d_leveled->operator[](0);
                    relinkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }

                break;
            case 3: // KEYSWITCHING_METHOD_III

                if (scheme_ == scheme_type::bfv)
                { // no leveled
                    d_ = context.d;
                    d_tilda_ = context.d_tilda;
                    r_prime_ = context.r_prime;
                    relinkey_size_ = 2 * d_ * d_tilda_ * r_prime_ * ring_size;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
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
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }

                break;
            default:
                break;
        }
    }

    __host__ Relinkey::Relinkey(Parameters& context, HostVector<Data64>& key,
                                bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_ = context.d;
        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                // It can use for both leveled and no leveled.
                relinkey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
                if (relinkey_size_ != key.size())
                {
                    throw std::invalid_argument("Invalid Key Size");
                }
                device_location_ = DeviceVector<Data64>(key);
                break;
            }
            case 2: // KEYSWITCHING_METHOD_II

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    relinkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                    if (relinkey_size_ != key.size())
                    {
                        throw std::invalid_argument("Invalid Key Size");
                    }
                    device_location_ = DeviceVector<Data64>(key);
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
                    d_ = context.d_leveled->operator[](0);
                    relinkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                    if (relinkey_size_ != key.size())
                    {
                        throw std::invalid_argument("Invalid Key Size");
                    }
                    device_location_ = DeviceVector<Data64>(key);
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }

                break;
            case 3: // KEYSWITCHING_METHOD_III

                if (scheme_ == scheme_type::bfv)
                { // no leveled
                    d_ = context.d;
                    d_tilda_ = context.d_tilda;
                    r_prime_ = context.r_prime;
                    relinkey_size_ = 2 * d_ * d_tilda_ * r_prime_ * ring_size;
                    if (relinkey_size_ != key.size())
                    {
                        throw std::invalid_argument("Invalid Key Size");
                    }
                    device_location_ = DeviceVector<Data64>(key);
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
                    throw std::invalid_argument("Invalid Key Size");
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }

                break;
            default:
                break;
        }
    }

    __host__ Relinkey::Relinkey(Parameters& context,
                                std::vector<HostVector<Data64>>& key,
                                bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_ = context.d;
        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                throw std::invalid_argument("Invalid Key Size");
            }
            case 2: // KEYSWITCHING_METHOD_II
                throw std::invalid_argument("Invalid Key Size");
                break;
            case 3: // KEYSWITCHING_METHOD_III

                if (scheme_ == scheme_type::bfv)
                { // no leveled
                    throw std::invalid_argument("Invalid Key Size");
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
                    d_ = context.d_leveled->operator[](0);
                    d_tilda_ = context.d_tilda_leveled->operator[](0);
                    r_prime_ = context.r_prime_leveled;

                    int max_depth = Q_size_ - 1;
                    if (max_depth != key.size())
                    {
                        throw std::invalid_argument("Invalid Key Size");
                    }
                    for (int i = 0; i < max_depth; i++)
                    {
                        relinkey_size_leveled_.push_back(
                            2 * context.d_leveled->operator[](i) *
                            context.d_tilda_leveled->operator[](i) * r_prime_ *
                            ring_size);

                        if (max_depth != key[i].size())
                        {
                            throw std::invalid_argument("Invalid Key Size");
                        }

                        device_location_leveled_.push_back(
                            std::move(DeviceVector<Data64>(key[i])));
                    }
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
                }

                break;
            default:
                break;
        }
    }

    void Relinkey::store_in_device(cudaStream_t stream)
    {
        if (store_in_gpu_)
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

            store_in_gpu_ = true;
        }
    }

    void Relinkey::store_in_host(cudaStream_t stream)
    {
        if (store_in_gpu_)
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

            store_in_gpu_ = false;
        }
        else
        {
            // pass
        }
    }

    Data64* Relinkey::data()
    {
        if (store_in_gpu_)
        {
            return device_location_.data();
        }
        else
        {
            return host_location_.data();
        }
    }
    Data64* Relinkey::data(size_t i)
    {
        if (store_in_gpu_)
        {
            return device_location_leveled_[i].data();
        }
        else
        {
            return host_location_leveled_[i].data();
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    __host__ MultipartyRelinkey::MultipartyRelinkey(Parameters& context,
                                                    const int seed,
                                                    bool store_in_gpu)
        : Relinkey(context, store_in_gpu), seed_(seed)
    {
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    __host__ Galoiskey::Galoiskey(Parameters& context, bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        customized = false;

        group_order_ = (scheme_ == scheme_type::bfv) ? 3 : 5;

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

                break;
            }
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

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled

                    d_ = context.d_leveled->operator[](0);
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else
                {
                    throw std::invalid_argument("Invalid Scheme Type");
                }

                break;
            }
            case 3: // KEYSWITCHING_METHOD_III Galoiskey
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__ Galoiskey::Galoiskey(Parameters& context,
                                  std::vector<int>& shift_vec,
                                  bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        customized = false;

        group_order_ = (scheme_ == scheme_type::bfv) ? 3 : 5;

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

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled

                    d_ = context.d_leveled->operator[](0);
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else
                {
                    throw std::invalid_argument("Invalid Scheme Type");
                }

                break;
            }
            case 3: // KEYSWITCHING_METHOD_III Galoiskey
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    __host__ Galoiskey::Galoiskey(Parameters& context,
                                  std::vector<uint32_t>& galois_elts,
                                  bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;
        customized = true;

        group_order_ = (scheme_ == scheme_type::bfv) ? 3 : 5;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I
            {
                galois_elt_zero =
                    steps_to_galois_elt(0, ring_size, group_order_);
                galoiskey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
                custom_galois_elt = galois_elts;

                break;
            }
            case 2: // KEYSWITCHING_METHOD_II

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    galois_elt_zero =
                        steps_to_galois_elt(0, ring_size, group_order_);
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                    custom_galois_elt = galois_elts;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled
                    d_ = context.d_leveled->operator[](0);
                    galois_elt_zero =
                        steps_to_galois_elt(0, ring_size, group_order_);
                    galoiskey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                    custom_galois_elt = galois_elts;
                }
                else
                {
                    throw std::invalid_argument("Invalid Scheme Type");
                }

                break;
            case 3: // KEYSWITCHING_METHOD_III Galoiskey
                throw std::invalid_argument(
                    "Galoiskey does not support KEYSWITCHING_METHOD_III");
                break;
            default:
                throw std::invalid_argument("Invalid Key Switching Type");
                break;
        }
    }

    void Galoiskey::store_in_device(cudaStream_t stream)
    {
        if (store_in_gpu_)
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

            store_in_gpu_ = true;
        }
    }

    void Galoiskey::store_in_host(cudaStream_t stream)
    {
        if (store_in_gpu_)
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

            store_in_gpu_ = false;
        }
        else
        {
            // pass
        }
    }

    Data64* Galoiskey::data(size_t i)
    {
        if (store_in_gpu_)
        {
            return device_location_[i].data();
        }
        else
        {
            return host_location_[i].data();
        }
    }

    Data64* Galoiskey::c_data()
    {
        if (store_in_gpu_)
        {
            return zero_device_location_.data();
        }
        else
        {
            return zero_host_location_.data();
        }
    }

    ////////////////////////////////////////
    ////////////////////////////////////////

    __host__ MultipartyGaloiskey::MultipartyGaloiskey(Parameters& context,
                                                      const int seed,
                                                      bool store_in_gpu)
        : Galoiskey(context, store_in_gpu), seed_(seed)
    {
    }

    __host__
    MultipartyGaloiskey::MultipartyGaloiskey(Parameters& context,
                                             std::vector<int>& shift_vec,
                                             const int seed, bool store_in_gpu)
        : Galoiskey(context, shift_vec, store_in_gpu), seed_(seed)
    {
    }

    __host__
    MultipartyGaloiskey::MultipartyGaloiskey(Parameters& context,
                                             std::vector<uint32_t>& galois_elts,
                                             const int seed, bool store_in_gpu)
        : Galoiskey(context, galois_elts, store_in_gpu), seed_(seed)
    {
    }

    ////////////////////////////////////////
    ////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////////////

    __host__ Switchkey::Switchkey(Parameters& context, bool store_in_gpu)
    {
        scheme_ = context.scheme_;
        key_type = context.keyswitching_type_;
        store_in_gpu_ = store_in_gpu;

        ring_size = context.n;

        Q_prime_size_ = context.Q_prime_size;
        Q_size_ = context.Q_size;

        d_ = context.d;
        d_tilda_ = context.d_tilda;
        r_prime_ = context.r_prime;

        switch (static_cast<int>(context.keyswitching_type_))
        {
            case 1: // KEYSWITCHING_METHOD_I

                // It can use for both leveled and no leveled.
                switchkey_size_ = 2 * Q_size_ * Q_prime_size_ * ring_size;
                break;
            case 2: // KEYSWITCHING_METHOD_II

                if (scheme_ == scheme_type::bfv)
                { // no leveled

                    d_ = context.d;
                    switchkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else if (scheme_ == scheme_type::ckks)
                { // leveled

                    d_ = context.d_leveled->operator[](0);
                    switchkey_size_ = 2 * d_ * Q_prime_size_ * ring_size;
                }
                else
                {
                    throw std::invalid_argument("Invalid Key Switching Type");
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

    //////////////////////////

    void Switchkey::store_in_device(cudaStream_t stream)
    {
        if (store_in_gpu_)
        {
            // pass
        }
        else
        {
            device_location_ = DeviceVector<Data64>(host_location_, stream);
            host_location_.resize(0);
            host_location_.shrink_to_fit();

            store_in_gpu_ = true;
        }
    }

    void Switchkey::store_in_host(cudaStream_t stream)
    {
        if (store_in_gpu_)
        {
            host_location_ = HostVector<Data64>(switchkey_size_);
            cudaMemcpyAsync(host_location_.data(), device_location_.data(),
                            switchkey_size_ * sizeof(Data64),
                            cudaMemcpyDeviceToHost, stream);
            HEONGPU_CUDA_CHECK(cudaGetLastError());

            device_location_.resize(0, stream);

            store_in_gpu_ = false;
        }
        else
        {
            // pass
        }
    }

    //////////////////////////

    Data64* Switchkey::data()
    {
        if (store_in_gpu_)
        {
            return device_location_.data();
        }
        else
        {
            return host_location_.data();
        }
    }

} // namespace heongpu