// Copyright 2024-2025 Yanbin Li
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Yanbin Li

#include "ckks/precision.cuh"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace heongpu
{
    // Convert delta to precision (log2(1/delta))
    Stats delta_to_precision(const Stats& delta)
    {
        return Stats(
            std::log2(1.0 / delta.real),
            std::log2(1.0 / delta.imag),
            std::log2(1.0 / delta.l2)
        );
    }

    // Calculate median of Stats array
    Stats calc_median(std::vector<Stats>& values)
    {
        std::vector<double> tmp(values.size());

        // Sort real values
        for (size_t i = 0; i < values.size(); i++) {
            tmp[i] = values[i].real;
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < values.size(); i++) {
            values[i].real = tmp[i];
        }

        // Sort imaginary values
        for (size_t i = 0; i < values.size(); i++) {
            tmp[i] = values[i].imag;
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < values.size(); i++) {
            values[i].imag = tmp[i];
        }

        // Sort L2 values
        for (size_t i = 0; i < values.size(); i++) {
            tmp[i] = values[i].l2;
        }
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < values.size(); i++) {
            values[i].l2 = tmp[i];
        }

        size_t index = values.size() / 2;

        if ((values.size() & 1) == 1 || index + 1 == values.size()) {
            return Stats(values[index].real, values[index].imag, values[index].l2);
        }

        return Stats(
            (values[index].real + values[index + 1].real) / 2.0,
            (values[index].imag + values[index + 1].imag) / 2.0,
            (values[index].l2 + values[index + 1].l2) / 2.0
        );
    }

    // Calculate CDF for precision distribution
    void calc_cdf(const std::vector<double>& precs,
                  std::vector<DistEntry>& res,
                  int cdf_resol)
    {
        std::vector<double> sorted_precs(precs);
        std::sort(sorted_precs.begin(), sorted_precs.end());

        double min_prec = sorted_precs[0];
        double max_prec = sorted_precs[sorted_precs.size() - 1];

        for (int i = 0; i < cdf_resol; i++) {
            double cur_prec = min_prec + static_cast<double>(i) * (max_prec - min_prec) / static_cast<double>(cdf_resol);

            for (size_t j = 0; j < sorted_precs.size(); j++) {
                if (sorted_precs[j] >= cur_prec) {
                    res[i].prec = cur_prec;
                    res[i].count = static_cast<int>(j);
                    break;
                }
            }
        }
    }

    // Get precision statistics comparing expected values with actual values
    PrecisionStats get_precision_stats(
        const std::vector<Complex64>& values_want,
        const std::vector<Complex64>& values_test)
    {
        PrecisionStats prec;

        size_t slots = values_want.size();

        if (values_test.size() != slots) {
            throw std::invalid_argument("values_want and values_test must have the same size");
        }

        std::vector<Stats> diff(slots);

        prec.max_delta = Stats(0.0, 0.0, 0.0);
        prec.min_delta = Stats(1.0, 1.0, 1.0);
        prec.mean_delta = Stats(0.0, 0.0, 0.0);

        prec.real_dist.resize(prec.cdf_resol);
        prec.imag_dist.resize(prec.cdf_resol);
        prec.l2_dist.resize(prec.cdf_resol);

        std::vector<double> prec_real(slots);
        std::vector<double> prec_imag(slots);
        std::vector<double> prec_l2(slots);

        for (size_t i = 0; i < slots; i++) {
            double delta_real = std::abs(values_test[i].real() - values_want[i].real());
            double delta_imag = std::abs(values_test[i].imag() - values_want[i].imag());
            double delta_l2 = std::sqrt(delta_real * delta_real + delta_imag * delta_imag);

            prec_real[i] = std::log2(1.0 / delta_real);
            prec_imag[i] = std::log2(1.0 / delta_imag);
            prec_l2[i] = std::log2(1.0 / delta_l2);

            diff[i].real = delta_real;
            diff[i].imag = delta_imag;
            diff[i].l2 = delta_l2;

            prec.mean_delta.real += delta_real;
            prec.mean_delta.imag += delta_imag;
            prec.mean_delta.l2 += delta_l2;

            if (delta_real > prec.max_delta.real) {
                prec.max_delta.real = delta_real;
            }
            if (delta_imag > prec.max_delta.imag) {
                prec.max_delta.imag = delta_imag;
            }
            if (delta_l2 > prec.max_delta.l2) {
                prec.max_delta.l2 = delta_l2;
            }

            if (delta_real < prec.min_delta.real) {
                prec.min_delta.real = delta_real;
            }
            if (delta_imag < prec.min_delta.imag) {
                prec.min_delta.imag = delta_imag;
            }
            if (delta_l2 < prec.min_delta.l2) {
                prec.min_delta.l2 = delta_l2;
            }
        }

        calc_cdf(prec_real, prec.real_dist, prec.cdf_resol);
        calc_cdf(prec_imag, prec.imag_dist, prec.cdf_resol);
        calc_cdf(prec_l2, prec.l2_dist, prec.cdf_resol);

        prec.min_precision = delta_to_precision(prec.max_delta);
        prec.max_precision = delta_to_precision(prec.min_delta);
        prec.mean_delta.real /= static_cast<double>(slots);
        prec.mean_delta.imag /= static_cast<double>(slots);
        prec.mean_delta.l2 /= static_cast<double>(slots);
        prec.mean_precision = delta_to_precision(prec.mean_delta);
        prec.median_delta = calc_median(diff);
        prec.median_precision = delta_to_precision(prec.median_delta);

        return prec;
    }

    // Print precision statistics
    std::string PrecisionStats::to_string() const
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        oss << "\n┌─────────┬───────┬───────┬───────┐\n";
        oss << "│    Log2 │ REAL  │ IMAG  │ L2    │\n";
        oss << "├─────────┼───────┼───────┼───────┤\n";
        oss << "│MIN Prec │ " << std::setw(5) << min_precision.real
            << " │ " << std::setw(5) << min_precision.imag
            << " │ " << std::setw(5) << min_precision.l2 << " │\n";
        oss << "│MAX Prec │ " << std::setw(5) << max_precision.real
            << " │ " << std::setw(5) << max_precision.imag
            << " │ " << std::setw(5) << max_precision.l2 << " │\n";
        oss << "│AVG Prec │ " << std::setw(5) << mean_precision.real
            << " │ " << std::setw(5) << mean_precision.imag
            << " │ " << std::setw(5) << mean_precision.l2 << " │\n";
        oss << "│MED Prec │ " << std::setw(5) << median_precision.real
            << " │ " << std::setw(5) << median_precision.imag
            << " │ " << std::setw(5) << median_precision.l2 << " │\n";
        oss << "└─────────┴───────┴───────┴───────┘\n";

        return oss.str();
    }

} // namespace heongpu
