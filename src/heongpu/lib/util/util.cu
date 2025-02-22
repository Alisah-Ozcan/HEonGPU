// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "util.cuh"

namespace heongpu
{
    Data64 extendedGCD(Data64 a, Data64 b, Data64& x, Data64& y)
    {
        if (a == 0)
        {
            x = 0;
            y = 1;
            return b;
        }

        Data64 x1, y1;
        Data64 gcd = extendedGCD(b % a, a, x1, y1);

        x = y1 - (b / a) * x1;
        y = x1;

        return gcd;
    }

    Data64 modInverse(Data64 a, Data64 m)
    {
        Data64 x, y;
        Data64 gcd = extendedGCD(a, m, x, y);

        if (gcd != 1)
        {
            // Modular inverse does not exist
            return 0;
        }
        else
        {
            // Ensure the result is positive
            Data64 result = (x % m + m) % m;
            return result;
        }
    }

    int countBits(Data64 input)
    {
        return (int) log2(input) + 1;
    }

    bool is_power_of_two(size_t number)
    {
        return (number > 0) && ((number & (number - 1)) == 0);
    }

    int calculate_bit_count(Data64 number)
    {
        return log2(number) + 1;
    }

    int calculate_big_integer_bit_count(Data64* number, int word_count)
    {
        int size = word_count;
        for (int i = (word_count - 1); i > (-1); i--)
        {
            if (number[i] == 0)
            {
                size--;
            }
            else
            {
                break;
            }
        }

        return ((size - 1) * 64) + calculate_bit_count(number[size - 1]);
    }

    bool miller_rabin(const Data64& value, size_t num_rounds)
    {
        Modulus64 modulus_in(value);

        Data64 d = value - 1;
        Data64 r = 0;
        while (0 == (d & 0x1)) // #true while the last bit of r is zero
        {
            d >>= 1;
            r++;
        }
        if (r == 0)
        {
            return false;
        }

        // apply miller_rabin primality test
        std::random_device rand;
        std::uniform_int_distribution<Data64> dist(3, value - 1);
        for (size_t i = 0; i < num_rounds; i++)
        {
            Data64 a = i ? dist(rand) : 2;
            Data64 x = OPERATOR64::exp(a, d, modulus_in);
            if (x == 1 || x == value - 1)
            {
                continue;
            }
            Data64 count = 0;
            do
            {
                x = OPERATOR64::mult(x, x, modulus_in);
                count++;
            } while (x != value - 1 && count < r - 1);
            if (x != value - 1)
            {
                return false;
            }
        }
        return true;
    }

    bool is_prime(const Data64& value)
    {
        size_t num_rounds = 11;

        // First check the prime under 1000.
        std::vector<Data64> low_primes = {
            3ULL,   5ULL,   7ULL,   11ULL,  13ULL,  17ULL,  19ULL,  23ULL,
            29ULL,  31ULL,  37ULL,  41ULL,  43ULL,  47ULL,  53ULL,  59ULL,
            61ULL,  67ULL,  71ULL,  73ULL,  79ULL,  83ULL,  89ULL,  97ULL,
            101ULL, 103ULL, 107ULL, 109ULL, 113ULL, 127ULL, 131ULL, 137ULL,
            139ULL, 149ULL, 151ULL, 157ULL, 163ULL, 167ULL, 173ULL, 179ULL,
            181ULL, 191ULL, 193ULL, 197ULL, 199ULL, 211ULL, 223ULL, 227ULL,
            229ULL, 233ULL, 239ULL, 241ULL, 251ULL, 257ULL, 263ULL, 269ULL,
            271ULL, 277ULL, 281ULL, 283ULL, 293ULL, 307ULL, 311ULL, 313ULL,
            317ULL, 331ULL, 337ULL, 347ULL, 349ULL, 353ULL, 359ULL, 367ULL,
            373ULL, 379ULL, 383ULL, 389ULL, 397ULL, 401ULL, 409ULL, 419ULL,
            421ULL, 431ULL, 433ULL, 439ULL, 443ULL, 449ULL, 457ULL, 461ULL,
            463ULL, 467ULL, 479ULL, 487ULL, 491ULL, 499ULL, 503ULL, 509ULL,
            521ULL, 523ULL, 541ULL, 547ULL, 557ULL, 563ULL, 569ULL, 571ULL,
            577ULL, 587ULL, 593ULL, 599ULL, 601ULL, 607ULL, 613ULL, 617ULL,
            619ULL, 631ULL, 641ULL, 643ULL, 647ULL, 653ULL, 659ULL, 661ULL,
            673ULL, 677ULL, 683ULL, 691ULL, 701ULL, 709ULL, 719ULL, 727ULL,
            733ULL, 739ULL, 743ULL, 751ULL, 757ULL, 761ULL, 769ULL, 773ULL,
            787ULL, 797ULL, 809ULL, 811ULL, 821ULL, 823ULL, 827ULL, 829ULL,
            839ULL, 853ULL, 857ULL, 859ULL, 863ULL, 877ULL, 881ULL, 883ULL,
            887ULL, 907ULL, 911ULL, 919ULL, 929ULL, 937ULL, 941ULL, 947ULL,
            953ULL, 967ULL, 971ULL, 977ULL, 983ULL, 991ULL, 997ULL};

        if (value >= 3)
        {
            if ((value & 0x1) != 0)
            {
                for (int i = 0; i < low_primes.size(); i++)
                {
                    if (value == low_primes[i])
                    {
                        return true;
                    }
                    if ((value % low_primes[i]) == 0)
                    {
                        return false;
                    }
                }

                return miller_rabin(value, num_rounds);
            }
        }

        return false;
    }

    std::vector<Data64> generate_proper_primes(Data64 factor, int bit_size,
                                               size_t count)
    {
        std::vector<Data64> destination;

        // Start with (2^bit_size - 1) / factor * factor + 1
        Data64 value = ((Data64(0x1) << bit_size) - 1) / factor * factor + 1;

        Data64 lower_bound = Data64(0x1) << (bit_size - 1);
        while (count > 0 && value > lower_bound)
        {
            if (is_prime(value))
            {
                destination.emplace_back(std::move(value));
                count--;
            }
            value -= factor;
        }
        if (count > 0)
        {
            throw std::logic_error("failed to find enough qualifying primes");
        }
        return destination;
    }

    std::vector<Modulus64>
    generate_primes(size_t poly_modulus_degree,
                    const std::vector<int> prime_bit_sizes)
    {
        std::vector<Modulus64> prime_vector_;
        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data64>> prime_table;
        for (int size : prime_bit_sizes)
        {
            if ((size > MAX_USER_DEFINED_MOD_BIT_COUNT) ||
                (size < MIN_USER_DEFINED_MOD_BIT_COUNT))
            {
                throw std::logic_error("invalid modulus bit size");
            }

            ++count_table[size];
        }

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        for (const auto& table_elt : count_table)
        {
            prime_table[table_elt.first] = generate_proper_primes(
                factor, table_elt.first, table_elt.second);
        }

        for (int size : prime_bit_sizes)
        {
            prime_vector_.emplace_back(Modulus(prime_table[size].back()));
            prime_table[size].pop_back();
        }

        return prime_vector_;
    }

    std::vector<Modulus64> generate_internal_primes(size_t poly_modulus_degree,
                                                    const int prime_count)
    {
        std::vector<Modulus64> all_primes;

        std::vector<int> prime_bit_sizes;
        for (int i = 0; i < prime_count; i++)
        {
            prime_bit_sizes.push_back(MAX_MOD_BIT_COUNT);
        }

        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data64>> prime_table;
        for (int size : prime_bit_sizes)
        {
            ++count_table[size];
        }

        Data64 factor = Data64(2) * Data64(poly_modulus_degree);
        for (const auto& table_elt : count_table)
        {
            prime_table[table_elt.first] = generate_proper_primes(
                factor, table_elt.first, table_elt.second);
        }

        for (int size : prime_bit_sizes)
        {
            all_primes.emplace_back(Modulus(prime_table[size].back()));
            prime_table[size].pop_back();
        }

        return all_primes;
    }

    bool is_primitive_root(Data64 root, size_t degree, Modulus64& modulus)
    {
        // root^(degree/2) = modulus - 1 .
        Data64 degree_over2 = degree >> 1;

        return OPERATOR64::exp(root, degree_over2, modulus) ==
               (modulus.value - 1);
    }

    bool find_primitive_root(size_t degree, Modulus64& modulus,
                             Data64& destination)
    {
        Data64 size_entire_group = modulus.value - 1;

        Data64 size_quotient_group = size_entire_group / degree;

        if (size_entire_group - size_quotient_group * degree != 0)
        {
            return false;
        }

        std::random_device rd;

        int attempt_counter = 0;
        int attempt_counter_max = 100;
        do
        {
            attempt_counter++;

            Data64 random_num =
                (static_cast<Data64>(rd()) << 32) | static_cast<Data64>(rd());
            // destination = OPERATOR64::reduce(random_num, modulus);
            destination = random_num % modulus.value;

            // Raise the random number to power the size of the quotient
            // to get rid of irrelevant part
            destination =
                OPERATOR64::exp(destination, size_quotient_group, modulus);
        } while (!is_primitive_root(destination, degree, modulus) &&
                 (attempt_counter < attempt_counter_max));

        return is_primitive_root(destination, degree, modulus);
    }

    Data64 find_minimal_primitive_root(size_t degree, Modulus64& modulus)
    {
        Data64 root;
        if (!find_primitive_root(degree, modulus, root))
        {
            throw std::logic_error("no sufficient root unity");
        }

        Data64 generator_sq = OPERATOR64::mult(root, root, modulus);

        Data64 current_generator = root;

        for (size_t i = 0; i < degree; i += 2)
        {
            if (current_generator < root)
            {
                root = current_generator;
            }

            current_generator =
                OPERATOR64::mult(current_generator, generator_sq, modulus);
        }

        return root;
    }

    std::vector<Data64>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus64> primes)
    {
        std::vector<Data64> root_of_unity;

        // 2nth root of unity
        for (int i = 0; i < primes.size(); i++)
        {
            root_of_unity.push_back(find_minimal_primitive_root(
                2 * poly_modulus_degree, primes[i]));
        }

        return root_of_unity;
    }

    std::vector<Root64> generate_ntt_table(std::vector<Data64> psi,
                                           std::vector<Modulus64> primes,
                                           int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root64> forward_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root64> table;
            table.push_back(1);

            for (int j = 1; j < n_; j++)
            {
                Data64 exp =
                    OPERATOR64::mult(table[(j - 1)], psi[i], primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                forward_table.push_back(table[gpuntt::bitreverse(j, n_power)]);
            }
        }

        return forward_table;
    }

    std::vector<Root64> generate_intt_table(std::vector<Data64> psi,
                                            std::vector<Modulus64> primes,
                                            int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root64> inverse_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root64> table;
            table.push_back(1);

            Data64 inv_root = OPERATOR64::modinv(psi[i], primes[i]);
            for (int j = 1; j < n_; j++)
            {
                Data64 exp =
                    OPERATOR64::mult(table[(j - 1)], inv_root, primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                inverse_table.push_back(table[gpuntt::bitreverse(j, n_power)]);
            }
        }

        return inverse_table;
    }

    std::vector<Ninverse64> generate_n_inverse(size_t poly_modulus_degree,
                                               std::vector<Modulus64> primes)
    {
        Data64 n_ = poly_modulus_degree;
        std::vector<Ninverse64> n_inverse;
        for (int i = 0; i < primes.size(); i++)
        {
            n_inverse.push_back(OPERATOR64::modinv(n_, primes[i]));
        }

        return n_inverse;
    }

    __global__ void unsigned_signed_convert(Data64* input, Data64* output,
                                            Modulus64* modulus)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size

        int64_t threshold = (modulus[0].value + 1) >> 1;
        int64_t input_reg = static_cast<int64_t>(input[idx]);
        input_reg = (input_reg > threshold)
                        ? input_reg - static_cast<int64_t>(modulus[0].value)
                        : input_reg;

        output[idx] = static_cast<Data64>(input_reg);
    }

    __global__ void fill_device_vector(Data64* vector, Data64 number, int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size

        if (idx < size)
        {
            vector[idx] = number;
        }
    }

    int find_closest_divisor(int N)
    {
        double target = std::sqrt(N);
        int closest_div = 1;
        double min_diff = std::abs(closest_div - target);

        for (int k = 1; k <= std::sqrt(N); ++k)
        {
            if (N % k == 0)
            {
                for (int divisor : {k, N / k})
                {
                    double diff = std::abs(divisor - target);
                    if (diff < min_diff)
                    {
                        min_diff = diff;
                        closest_div = divisor;
                    }
                }
            }
        }
        return closest_div;
    }

    std::vector<std::vector<int>> split_array(const std::vector<int>& array,
                                              int chunk_size)
    {
        std::vector<std::vector<int>> result;
        int n = array.size();
        for (int i = 0; i < n; i += chunk_size)
        {
            result.emplace_back(array.begin() + i,
                                array.begin() + min(i + chunk_size, n));
        }
        return result;
    }

    std::vector<std::vector<int>> seperate_func(const std::vector<int>& A)
    {
        int initial_size = A.size();
        int counter = 2;
        int offset = A[1] - A[0];

        for (size_t i = 1; i < A.size() - 1; ++i)
        {
            if (A[i + 1] - A[i] != offset)
            {
                break;
            }
            counter++;
        }

        int real_n1 = heongpu::find_closest_divisor(counter);

        if (counter == initial_size)
        {
            return split_array(A, real_n1);
        }
        else
        {
            auto first_part = split_array(
                std::vector<int>(A.begin(), A.begin() + counter), real_n1);
            auto second_part = split_array(
                std::vector<int>(A.begin() + counter, A.end()), real_n1);

            first_part.insert(first_part.end(), second_part.begin(),
                              second_part.end());
            return first_part;
        }
    }

    std::vector<int> unique_sort(const std::vector<int>& input)
    {
        std::set<int> result(input.begin(), input.end());

        return std::vector<int>(result.begin(), result.end());
    }

    BootstrappingConfig::BootstrappingConfig(int CtoS, int StoC, int taylor,
                                             bool less_key_mode)
        : CtoS_piece_(CtoS), StoC_piece_(StoC), taylor_number_(taylor),
          less_key_mode_(less_key_mode)
    {
        validate();
    }

    // Validation Function Implementation
    void BootstrappingConfig::validate()
    {
        if (CtoS_piece_ < 2 || CtoS_piece_ > 5)
        {
            throw std::out_of_range("CtoS_piece must be in range [2, 5]");
        }
        if (StoC_piece_ < 2 || StoC_piece_ > 5)
        {
            throw std::out_of_range("StoC_piece must be in range [2, 5]");
        }
        if (taylor_number_ < 6 || taylor_number_ > 15)
        {
            throw std::out_of_range("taylor_number must be in range [6, 15]");
        }
    }

} // namespace heongpu