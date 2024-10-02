// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HOST_VECTOR_H
#define HOST_VECTOR_H

#include "memorypool.cuh"

namespace heongpu
{
    template <typename T>
    class HostVector : public std::vector<T, rmm_pinned_allocator<T>>
    {
      public:
        using std::vector<T, rmm_pinned_allocator<T>>::vector;
    };

} // namespace heongpu
#endif // HOST_VECTOR_H