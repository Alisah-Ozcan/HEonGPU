﻿// Copyright 2024 Alişah Özcan
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
// Developer: Alişah Özcan

#include "util.cuh"

namespace heongpu
{
    Data extendedGCD(Data a, Data b, Data& x, Data& y)
    {
        if (a == 0)
        {
            x = 0;
            y = 1;
            return b;
        }

        Data x1, y1;
        Data gcd = extendedGCD(b % a, a, x1, y1);

        x = y1 - (b / a) * x1;
        y = x1;

        return gcd;
    }

    Data modInverse(Data a, Data m)
    {
        Data x, y;
        Data gcd = extendedGCD(a, m, x, y);

        if (gcd != 1)
        {
            // Modular inverse does not exist
            return 0;
        }
        else
        {
            // Ensure the result is positive
            Data result = (x % m + m) % m;
            return result;
        }
    }

    int countBits(Data input)
    {
        return (int) log2(input) + 1;
    }

    bool is_power_of_two(size_t number)
    {
        return (number > 0) && ((number & (number - 1)) == 0);
    }

    int calculate_bit_count(Data number)
    {
        return log2(number) + 1;
    }

    int calculate_big_integer_bit_count(Data* number, int word_count)
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

    bool miller_rabin(const Data& value, size_t num_rounds)
    {
        Modulus modulus_in(value);

        Data d = value - 1;
        Data r = 0;
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
        std::uniform_int_distribution<Data> dist(3, value - 1);
        for (size_t i = 0; i < num_rounds; i++)
        {
            Data a = i ? dist(rand) : 2;
            Data x = VALUE::exp(a, d, modulus_in);
            if (x == 1 || x == value - 1)
            {
                continue;
            }
            Data count = 0;
            do
            {
                x = VALUE::mult(x, x, modulus_in);
                count++;
            } while (x != value - 1 && count < r - 1);
            if (x != value - 1)
            {
                return false;
            }
        }
        return true;
    }

    bool is_prime(const Data& value)
    {
        size_t num_rounds = 11;

        // First check the prime under 1000.
        std::vector<Data> low_primes = {
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

    std::vector<Data> generate_proper_primes(Data factor, int bit_size,
                                             size_t count)
    {
        std::vector<Data> destination;

        // Start with (2^bit_size - 1) / factor * factor + 1
        Data value = ((Data(0x1) << bit_size) - 1) / factor * factor + 1;

        Data lower_bound = Data(0x1) << (bit_size - 1);
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

    std::vector<Modulus> generate_primes(size_t poly_modulus_degree,
                                         const std::vector<int> prime_bit_sizes)
    {
        std::vector<Modulus> prime_vector_;
        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data>> prime_table;
        for (int size : prime_bit_sizes)
        {
            if ((size > MAX_USER_DEFINED_MOD_BIT_COUNT) ||
                (size < MIN_USER_DEFINED_MOD_BIT_COUNT))
            {
                throw std::logic_error("invalid modulus bit size");
            }

            ++count_table[size];
        }

        Data factor = Data(2) * Data(poly_modulus_degree);
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

    std::vector<Modulus> generate_internal_primes(size_t poly_modulus_degree,
                                                  const int prime_count)
    {
        std::vector<Modulus> all_primes;

        std::vector<int> prime_bit_sizes;
        for (int i = 0; i < prime_count; i++)
        {
            prime_bit_sizes.push_back(MAX_MOD_BIT_COUNT);
        }

        std::unordered_map<int, size_t> count_table;
        std::unordered_map<int, std::vector<Data>> prime_table;
        for (int size : prime_bit_sizes)
        {
            ++count_table[size];
        }

        Data factor = Data(2) * Data(poly_modulus_degree);
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

    bool is_primitive_root(Data root, size_t degree, Modulus& modulus)
    {
        // root^(degree/2) = modulus - 1 .
        Data degree_over2 = degree >> 1;

        return VALUE::exp(root, degree_over2, modulus) == (modulus.value - 1);
    }

    bool find_primitive_root(size_t degree, Modulus& modulus, Data& destination)
    {
        Data size_entire_group = modulus.value - 1;

        Data size_quotient_group = size_entire_group / degree;

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

            Data random_num =
                (static_cast<Data>(rd()) << 32) | static_cast<Data>(rd());
            // destination = VALUE::reduce(random_num, modulus);
            destination = random_num % modulus.value;

            // Raise the random number to power the size of the quotient
            // to get rid of irrelevant part
            destination = VALUE::exp(destination, size_quotient_group, modulus);
        } while (!is_primitive_root(destination, degree, modulus) &&
                 (attempt_counter < attempt_counter_max));

        return is_primitive_root(destination, degree, modulus);
    }

    Data find_minimal_primitive_root(size_t degree, Modulus& modulus)
    {
        Data root;
        if (!find_primitive_root(degree, modulus, root))
        {
            throw std::logic_error("no sufficient root unity");
        }

        Data generator_sq = VALUE::mult(root, root, modulus);

        Data current_generator = root;

        for (size_t i = 0; i < degree; i += 2)
        {
            if (current_generator < root)
            {
                root = current_generator;
            }

            current_generator =
                VALUE::mult(current_generator, generator_sq, modulus);
        }

        return root;
    }

    std::vector<Data>
    generate_primitive_root_of_unity(size_t poly_modulus_degree,
                                     std::vector<Modulus> primes)
    {
        std::vector<Data> root_of_unity;

        // 2nth root of unity
        for (int i = 0; i < primes.size(); i++)
        {
            root_of_unity.push_back(find_minimal_primitive_root(
                2 * poly_modulus_degree, primes[i]));
        }

        return root_of_unity;
    }

    std::vector<Root> generate_ntt_table(std::vector<Data> psi,
                                         std::vector<Modulus> primes,
                                         int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root> forward_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root> table;
            table.push_back(1);

            for (int j = 1; j < n_; j++)
            {
                Data exp = VALUE::mult(table[(j - 1)], psi[i], primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                forward_table.push_back(table[bitreverse(j, n_power)]);
            }
        }

        return forward_table;
    }

    std::vector<Root> generate_intt_table(std::vector<Data> psi,
                                          std::vector<Modulus> primes,
                                          int n_power)
    {
        int n_ = 1 << n_power;
        std::vector<Root> inverse_table; // bit reverse order
        for (int i = 0; i < primes.size(); i++)
        {
            std::vector<Root> table;
            table.push_back(1);

            Data inv_root = VALUE::modinv(psi[i], primes[i]);
            for (int j = 1; j < n_; j++)
            {
                Data exp = VALUE::mult(table[(j - 1)], inv_root, primes[i]);
                table.push_back(exp);
            }

            for (int j = 0; j < n_; j++) // take bit reverse order
            {
                inverse_table.push_back(table[bitreverse(j, n_power)]);
            }
        }

        return inverse_table;
    }

    std::vector<Ninverse> generate_n_inverse(size_t poly_modulus_degree,
                                             std::vector<Modulus> primes)
    {
        Data n_ = poly_modulus_degree;
        std::vector<Ninverse> n_inverse;
        for (int i = 0; i < primes.size(); i++)
        {
            n_inverse.push_back(VALUE::modinv(n_, primes[i]));
        }

        return n_inverse;
    }

    __global__ void unsigned_signed_convert(Data* input, Data* output,
                                            Modulus* modulus)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x; // ring size

        int64_t threshold = (modulus[0].value + 1) >> 1;
        int64_t input_reg = static_cast<int64_t>(input[idx]);
        input_reg = (input_reg > threshold)
                        ? input_reg - static_cast<int64_t>(modulus[0].value)
                        : input_reg;

        output[idx] = static_cast<Data>(input_reg);
    }
} // namespace heongpu