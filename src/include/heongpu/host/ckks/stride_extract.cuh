// Copyright 2024-2025 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#ifndef HEONGPU_CKKS_STRIDE_EXTRACT_H
#define HEONGPU_CKKS_STRIDE_EXTRACT_H

#include <heongpu/host/ckks/operator.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace heongpu
{
    struct StrideExtractStats
    {
        int mul_plain_mask = 0;
        int rotate = 0;
        int add = 0;
    };

    /**
     * @brief Slot-index mapping used by stride extraction.
     *
     * We interpret a 2D tensor (valid output) as row-major indices:
     *   linear(i,j) = i * w + j
     *
     * and map it into CKKS slots by:
     *   slot(i,j) = base_stride * linear(i,j) + base_offset
     *
     * This matches the "BatchConv+PackCoeffs" layout after a coeff->slot
     * transform if coeff-index == slot-index:
     *   slot(i,j,out=b) = B * (i*w + j) + b
     *
     * where base_stride=B and base_offset=b.
     *
     * Note: This function assumes the ciphertext is in *slot domain* and uses
     * slot rotations (rotate_rows). In HEonGPU, rotate_rows_inplace(ct, shift)
     * is assumed to move value at slot s to slot (s + shift) mod slot_count.
     */
    struct SlotIndexMap2D
    {
        int w = 0;            // original width used in linear(i,j)
        int base_stride = 1;  // interleave stride in slots
        int base_offset = 0;  // lane offset within the interleave

        __host__ inline int slot_of(int i, int j) const
        {
            return base_stride * (i * w + j) + base_offset;
        }
    };

    /**
     * @brief Stride-2 downsample of a d×d VALID output stored in slots.
     *
     * Keeps positions (i,j) where i%2==0 and j%2==0, and compacts them into the
     * first d2*d2 slots in row-major order, where d2 = ceil(d/2).
     *
     * Implementation: mask + rotate + sum.
     * - Group elements by rotation delta so that each delta uses one mask/plain
     *   multiply and one rotation.
     *
     * @param ct_slots Slot-domain ciphertext containing the (possibly sparse)
     *                 2D output according to map.
     * @param map Slot mapping from (i,j) to slot index.
     * @param d Valid output width/height.
     * @param galois_key Galois key for slot rotations.
     * @return Ciphertext with compacted stride-2 output in slots [0..d2*d2).
     */
    __host__ inline Ciphertext<Scheme::CKKS> stride2_extract_slots(
        HEArithmeticOperator<Scheme::CKKS>& ops, Ciphertext<Scheme::CKKS>& ct_slots,
        const SlotIndexMap2D& map, int d, Galoiskey<Scheme::CKKS>& galois_key,
        const ExecutionOptions& options = ExecutionOptions(),
        StrideExtractStats* stats = nullptr)
    {
        if (d <= 0)
        {
            throw std::invalid_argument("Invalid d");
        }
        if (map.w <= 0 || map.base_stride <= 0)
        {
            throw std::invalid_argument("Invalid SlotIndexMap2D");
        }
        if (ct_slots.rescale_required() || ct_slots.relinearization_required())
        {
            throw std::invalid_argument(
                "stride2_extract_slots requires a clean ciphertext (no pending "
                "rescale/relin).");
        }

        const int d2 = (d + 1) / 2;
        const int needed = d2 * d2;
        if (needed <= 0)
        {
            throw std::invalid_argument("Invalid d2");
        }

        // Group by delta = dst - src in slot indices.
        std::unordered_map<int, std::vector<int>> delta_to_src;
        delta_to_src.reserve(static_cast<size_t>(needed));

        for (int i = 0; i < d; i += 2)
        {
            for (int j = 0; j < d; j += 2)
            {
                const int src = map.slot_of(i, j);
                const int dst = (i / 2) * d2 + (j / 2);
                const int delta = dst - src;
                delta_to_src[delta].push_back(src);
            }
        }

        // Deterministic order for reproducibility.
        std::vector<int> deltas;
        deltas.reserve(delta_to_src.size());
        for (const auto& kv : delta_to_src)
            deltas.push_back(kv.first);
        std::sort(deltas.begin(), deltas.end());

        Ciphertext<Scheme::CKKS> acc = ct_slots;
        // Zero-out accumulator via mask (all zeros).
        {
            std::vector<double> zero_mask(static_cast<size_t>(ops.slot_count()),
                                          0.0);
            Plaintext<Scheme::CKKS> Pz;
            ops.encode_slots_leveled(Pz, zero_mask, 1.0, ct_slots.depth(), options);
            ops.multiply_plain_mask_inplace(acc, Pz, options);
            if (stats)
                stats->mul_plain_mask += 1;
        }

        bool first = true;
        for (int delta : deltas)
        {
            std::vector<double> mask(static_cast<size_t>(ops.slot_count()), 0.0);
            for (int src : delta_to_src[delta])
            {
                if (src < 0 || src >= ops.slot_count())
                {
                    throw std::invalid_argument("Source slot out of range");
                }
                mask[static_cast<size_t>(src)] = 1.0;
            }

            Plaintext<Scheme::CKKS> Pm;
            ops.encode_slots_leveled(Pm, mask, 1.0, ct_slots.depth(), options);

            Ciphertext<Scheme::CKKS> tmp = ct_slots;
            ops.multiply_plain_mask_inplace(tmp, Pm, options);
            if (stats)
                stats->mul_plain_mask += 1;

            if (delta != 0)
            {
                ops.rotate_rows_inplace(tmp, galois_key, delta, options);
                if (stats)
                    stats->rotate += 1;
            }

            if (first)
            {
                acc = std::move(tmp);
                first = false;
            }
            else
            {
                ops.add_inplace(acc, tmp, options);
                if (stats)
                    stats->add += 1;
            }
        }

        return acc;
    }

} // namespace heongpu

#endif // HEONGPU_CKKS_STRIDE_EXTRACT_H

