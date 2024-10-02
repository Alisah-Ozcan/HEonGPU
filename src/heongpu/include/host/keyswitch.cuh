// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef KEYSWITCH_H
#define KEYSWITCH_H

#include "context.cuh"
#include "keygeneration.cuh"

namespace heongpu
{
    /**
     * @brief Relinkey represents a relinearization key used for homomorphic
     * encryption operations.
     *
     * The Relinkey class is initialized with encryption parameters and
     * optionally with key data. It provides methods for storing the
     * relinearization key in GPU or CPU memory, which is essential for
     * performing homomorphic operations that require relinearization, such as
     * reducing the size of ciphertexts after multiplication.
     */
    class Relinkey
    {
        friend class HEKeyGenerator;
        friend class HEOperator;

      public:
        /**
         * @brief Constructs a new Relinkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Relinkey(Parameters& context, bool store_in_gpu = true);

        /**
         * @brief Constructs a new Relinkey object with specified parameters and
         * a key in HostVector format.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param key HostVector containing the relinearization key data.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Relinkey(Parameters& context, HostVector<Data>& key,
                          bool store_in_gpu = true);

        /**
         * @brief Constructs a new Relinkey object with specified parameters and
         * a key in vector format.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param key Vector of HostVector containing the relinearization key
         * data for multiple parts.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Relinkey(Parameters& context,
                          std::vector<HostVector<Data>>& key,
                          bool store_in_gpu = true);

        /**
         * @brief Stores the relinearization key in the GPU memory.
         */
        void store_key_in_device();

        /**
         * @brief Stores the relinearization key in the host (CPU) memory.
         */
        void store_key_in_host();

        /**
         * @brief Returns a pointer to the underlying relinearization key data.
         *
         * @return Data* Pointer to the relinearization key data.
         */
        Data* data();

        /**
         * @brief Returns a pointer to the specific part of the relinearization
         * key data.
         *
         * @param i Index of the key level to access.
         * @return Data* Pointer to the specified part of the relinearization
         * key data.
         */
        Data* data(size_t i);

        Relinkey() = delete;
        Relinkey(const Relinkey& copy) = delete;
        Relinkey(Relinkey&& source) = delete;
        Relinkey& operator=(const Relinkey& assign) = delete;
        Relinkey& operator=(Relinkey&& assign) = delete;

      private:
        scheme_type scheme_;
        keyswitching_type key_type;

        int ring_size;

        int Q_prime_size_;
        int Q_size_;

        int d_;
        int d_tilda_;
        int r_prime_;

        bool store_in_gpu_;
        size_t relinkey_size_;
        std::vector<size_t> relinkey_size_leveled_;

        // store key in device (GPU)
        DeviceVector<Data> device_location_;
        std::vector<DeviceVector<Data>> device_location_leveled_;

        // store key in host (CPU)
        HostVector<Data> host_location_;
        std::vector<HostVector<Data>> host_location_leveled_;
    };

    /**
     * @brief Galoiskey represents a Galois key used for performing homomorphic
     * operations such as rotations on encrypted data.
     *
     * The Galoiskey class is initialized with encryption parameters and
     * provides options to allow certain types of rotations. The Galois key can
     * be used to rotate encrypted data by specific shifts or elements, which is
     * useful in operations like ciphertext multiplication or encoding
     * manipulations. The class also offers flexibility to store the key in
     * either GPU or CPU memory.
     */
    class Galoiskey
    {
        friend class HEKeyGenerator;
        friend class HEOperator;

      public:
        /**
         * @brief Constructs a new Galoiskey object with default settings that
         * allow rotations in the range (-255, 255).
         *
         * This range can be increased by modifying the definition in
         * kernel/defines.h (MAX_SHIFT).
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Galoiskey(Parameters& context, bool store_in_gpu = true);

        /**
         * @brief Constructs a new Galoiskey object allowing specific rotations
         * based on the given vector of shifts.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param shift_vec Vector of integers representing the allowed shifts
         * for rotations.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Galoiskey(Parameters& context, std::vector<int>& shift_vec,
                           bool store_in_gpu = true);

        /**
         * @brief Constructs a new Galoiskey object allowing certain Galois
         * elements given in a vector.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param galois_elts Vector of Galois elements (represented as unsigned
         * 32-bit integers) specifying the rotations allowed.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Galoiskey(Parameters& context,
                           std::vector<uint32_t>& galois_elts,
                           bool store_in_gpu = true);

        /**
         * @brief Stores the Galois key in the GPU memory.
         */
        void store_key_in_device();

        /**
         * @brief Stores the Galois key in the host (CPU) memory.
         */
        void store_key_in_host();

        /**
         * @brief Returns a pointer to the specified part of the Galois key
         * data.
         *
         * @param i Index of the key elvel to access.
         * @return Data* Pointer to the specified part of the Galois key data.
         */
        Data* data(size_t i);

        /**
         * @brief Returns a pointer to the Galois key data for column
         * rotation(for BFV).
         *
         * @return Data* Pointer to the Galois key data for column rotation.
         */
        Data* c_data();

        Galoiskey() = default;
        Galoiskey(const Galoiskey& copy) = default;
        Galoiskey(Galoiskey&& source) = default;
        Galoiskey& operator=(const Galoiskey& assign) = default;
        Galoiskey& operator=(Galoiskey&& assign) = default;

      private:
        scheme_type scheme_;
        keyswitching_type key_type;

        int ring_size;

        int Q_prime_size_;
        int Q_size_;

        int d_;
        int d_tilda_;
        int r_prime_;

        bool customized;

        bool store_in_gpu_;
        size_t galoiskey_size_;
        std::vector<u_int32_t> custom_galois_elt;

        // for rotate_rows
        std::unordered_map<int, int> galois_elt;
        std::unordered_map<int, DeviceVector<Data>> device_location_;
        std::unordered_map<int, HostVector<Data>> host_location_;

        // for rotate_columns
        int galois_elt_zero;
        DeviceVector<Data> zero_device_location_;
        HostVector<Data> zero_host_location_;
    };

    /**
     * @brief Switchkey represents a key used for switching between different
     * secret keys in homomorphic encryption.
     *
     * The Switchkey class is initialized with encryption parameters and
     * provides the functionality to store the switching key either in GPU or
     * CPU memory. Key switching is an essential operation in homomorphic
     * encryption schemes, allowing ciphertexts encrypted under one key to be
     * transformed for use with a different secret key, which is particularly
     * useful for multi-party computation and other advanced encryption
     * scenarios.
     */
    class Switchkey
    {
        friend class HEKeyGenerator;
        friend class HEOperator;

      public:
        /**
         * @brief Constructs a new Switchkey object with specified parameters.
         *
         * @param context Reference to the Parameters object that sets the
         * encryption parameters.
         * @param store_in_gpu A boolean value indicating whether to store the
         * key in GPU memory. Default is true.
         */
        __host__ Switchkey(Parameters& context, bool store_in_gpu = true);

        /**
         * @brief Stores the switch key in the GPU memory.
         */
        void store_key_in_device();

        /**
         * @brief Stores the switch key in the host (CPU) memory.
         */
        void store_key_in_host();

        /**
         * @brief Returns a pointer to the underlying switch key data.
         *
         * @return Data* Pointer to the switch key data.
         */
        Data* data();

        Switchkey() = default;
        Switchkey(const Switchkey& copy) = default;
        Switchkey(Switchkey&& source) = default;
        Switchkey& operator=(const Switchkey& assign) = default;
        Switchkey& operator=(Switchkey&& assign) = default;

      private:
        scheme_type scheme_;
        keyswitching_type key_type;

        int ring_size;

        int Q_prime_size_;
        int Q_size_;

        int d_;
        int d_tilda_;
        int r_prime_;

        bool store_in_gpu_;
        size_t switchkey_size_;

        DeviceVector<Data> device_location_;
        HostVector<Data> host_location_;
    };

} // namespace heongpu
#endif // KEYSWITCH_H
