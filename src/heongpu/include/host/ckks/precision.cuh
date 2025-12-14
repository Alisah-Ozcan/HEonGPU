// Copyright 2025 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

#ifndef HEONGPU_CKKS_PRECISION_H
#define HEONGPU_CKKS_PRECISION_H

#include "complex.cuh"
#include <vector>
#include <string>
#include <cmath>

namespace heongpu
{
    // Stats structure storing real, imaginary and L2 norm about precision
    struct Stats {
        double real;
        double imag;
        double l2;

        Stats() : real(0.0), imag(0.0), l2(0.0) {}
        Stats(double r, double i, double l2_val) : real(r), imag(i), l2(l2_val) {}
    };

    // Distribution entry for CDF
    struct DistEntry {
        double prec;
        int count;

        DistEntry() : prec(0.0), count(0) {}
    };

    // PrecisionStats structure storing statistics about CKKS precision
    struct PrecisionStats {
        Stats max_delta;
        Stats min_delta;
        Stats max_precision;
        Stats min_precision;
        Stats mean_delta;
        Stats mean_precision;
        Stats median_delta;
        Stats median_precision;

        std::vector<DistEntry> real_dist;
        std::vector<DistEntry> imag_dist;
        std::vector<DistEntry> l2_dist;

        int cdf_resol;

        PrecisionStats() :
            max_delta(0.0, 0.0, 0.0),
            min_delta(1.0, 1.0, 1.0),
            max_precision(0.0, 0.0, 0.0),
            min_precision(0.0, 0.0, 0.0),
            mean_delta(0.0, 0.0, 0.0),
            mean_precision(0.0, 0.0, 0.0),
            median_delta(0.0, 0.0, 0.0),
            median_precision(0.0, 0.0, 0.0),
            cdf_resol(500) {}

        // Print precision statistics
        std::string to_string() const;
    };

    // Get precision statistics comparing expected values with actual values
    // values_want: expected/reference values
    // values_test: actual/decrypted values
    PrecisionStats get_precision_stats(
        const std::vector<Complex64>& values_want,
        const std::vector<Complex64>& values_test);

    // Helper function: convert delta to precision (log2(1/delta))
    Stats delta_to_precision(const Stats& delta);

    // Helper function: calculate median of Stats array
    Stats calc_median(std::vector<Stats>& values);

    // Helper function: calculate CDF for precision distribution
    void calc_cdf(const std::vector<double>& precs,
                  std::vector<DistEntry>& res,
                  int cdf_resol);

} // namespace heongpu

#endif // HEONGPU_CKKS_PRECISION_H
