#ifndef LOWMEMORYFHERESNET20_UTILS_H
#define LOWMEMORYFHERESNET20_UTILS_H

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define YELLOW_TEXT "\033[1;33m"
#define RESET_COLOR "\033[0m"

namespace utils {

using namespace std;
using namespace std::chrono;

static inline chrono::time_point<steady_clock, nanoseconds> start_time() {
    return steady_clock::now();
}

static duration<long long, ratio<1, 1000>> total_time;

static inline string get_class(int max_index) {
    switch (max_index) {
        case 0:
            return "Airplane";
        case 1:
            return "Automobile";
        case 2:
            return "Bird";
        case 3:
            return "Cat";
        case 4:
            return "Deer";
        case 5:
            return "Dog";
        case 6:
            return "Frog";
        case 7:
            return "Horse";
        case 8:
            return "Ship";
        case 9:
            return "Truck";
    }
    return "?";
}

static inline void print_duration(chrono::time_point<steady_clock, nanoseconds> start,
                                  const string& title) {
    auto ms = duration_cast<milliseconds>(steady_clock::now() - start);
    total_time += ms;

    auto secs = duration_cast<seconds>(ms);
    ms -= duration_cast<milliseconds>(secs);
    auto mins = duration_cast<minutes>(secs);
    secs -= duration_cast<seconds>(mins);

    if (mins.count() < 1) {
        cout << "⌛(" << title << "): " << secs.count() << ":" << ms.count()
             << "s" << " (Total: " << duration_cast<seconds>(total_time).count()
             << "s)" << endl;
    } else {
        cout << "⌛(" << title << "): " << mins.count() << "." << secs.count()
             << ":" << ms.count() << endl;
    }
}

static inline void print_duration_yellow(
    chrono::time_point<steady_clock, nanoseconds> start, const string& title) {
    auto ms = duration_cast<milliseconds>(steady_clock::now() - start);
    total_time += ms;

    auto secs = duration_cast<seconds>(ms);
    ms -= duration_cast<milliseconds>(secs);
    auto mins = duration_cast<minutes>(secs);
    secs -= duration_cast<seconds>(mins);

    if (mins.count() < 1) {
        cout << "⌛(" << title << "): " << secs.count() << ":" << ms.count()
             << "s" << " (Total: " << duration_cast<seconds>(total_time).count()
             << "s)" << endl;
    } else {
        cout << "⌛(" << title << "): " << YELLOW_TEXT << mins.count() << "."
             << secs.count() << ":" << ms.count() << RESET_COLOR << endl;
    }
}

static inline vector<double> read_values_from_file(const string& filename,
                                                   double scale = 1) {
    vector<double> values;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Can not open " << filename << endl;
        return values;
    }

    string row;
    while (getline(file, row)) {
        istringstream stream(row);
        string value;
        while (getline(stream, value, ',')) {
            try {
                double num = stod(value);
                values.push_back(num * scale);
            } catch (const invalid_argument&) {
                cerr << "Can not convert: " << value << endl;
            }
        }
    }

    file.close();
    return values;
}

static inline vector<double> read_fc_weight(const string& filename) {
    vector<double> weight = read_values_from_file(filename);
    vector<double> weight_corrected;

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 10; j++) {
            weight_corrected.push_back(weight[(10 * i) + j]);
        }
        for (int j = 0; j < 64 - 10; j++) {
            weight_corrected.push_back(0);
        }
    }

    return weight_corrected;
}

static inline int get_relu_depth(int degree) {
    switch (degree) {
        case 5:
            return 3;
        case 13:
            return 4;
        case 27:
            return 5;
        case 59:
            return 6;
        case 119:
            return 7;
        case 200:
        case 247:
            return 8;
        case 495:
            return 9;
        case 1007:
            return 10;
        case 2031:
            return 11;
    }

    cerr << "Set a valid degree for ReLU" << endl;
    exit(1);
}

} // namespace utils

#endif // LOWMEMORYFHERESNET20_UTILS_H
