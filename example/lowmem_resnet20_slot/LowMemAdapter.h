#ifndef LOWMEM_RESNET20_ADAPTER_H
#define LOWMEM_RESNET20_ADAPTER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <heongpu/heongpu.hpp>
#include <heongpu/host/ckks/chebyshev_interpolation.cuh>
#include <gpufft/complex.cuh>
#include <cuda_runtime.h>

#include "Utils.h"

namespace lowmem {

constexpr auto Scheme = heongpu::Scheme::CKKS;
using Ctxt = heongpu::Ciphertext<Scheme>;
using Ptxt = heongpu::Plaintext<Scheme>;

struct HEConfig {
    size_t poly_modulus_degree = 65536;
    std::vector<int> coeff_modulus_bits = {
        60, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
        50, 50, 50, 50, 50, 50, 50, 50, 50};
    std::vector<int> special_primes_bits = {60, 60, 60};
    double scale = std::pow(2.0, 50);
    int relu_degree = 119;
    int ctos_piece = 3;
    int stoc_piece = 3;
    int taylor_number = 11;
    bool less_key_mode = true;
    heongpu::keyswitching_type keyswitch =
        heongpu::keyswitching_type::KEYSWITCHING_METHOD_II;
    heongpu::sec_level_type sec_level = heongpu::sec_level_type::none;
};

class FHEController {
  public:
    int circuit_depth = 0;
    int num_slots = 0;
    int relu_degree = 119;
    std::string weights_dir;
    bool debug_cuda = false;
    std::string debug_label;

    size_t mul_count = 0;
    size_t rot_count = 0;
    size_t rescale_count = 0;
    size_t relin_count = 0;
    size_t boot_count = 0;

    FHEController()
        : context_(heongpu::keyswitching_type::KEYSWITCHING_METHOD_II,
                   heongpu::sec_level_type::none) {}

    void initialize(const HEConfig& cfg)
    {
        context_ = heongpu::HEContext<Scheme>(cfg.keyswitch, cfg.sec_level);
        context_.set_poly_modulus_degree(cfg.poly_modulus_degree);
        context_.set_coeff_modulus_bit_sizes(cfg.coeff_modulus_bits,
                                             cfg.special_primes_bits);
        context_.generate();

        default_scale_ = cfg.scale;
        relu_degree = cfg.relu_degree;

        num_slots = static_cast<int>(context_.get_poly_modulus_degree() / 2);
        circuit_depth = static_cast<int>(cfg.coeff_modulus_bits.size()) - 1;

        keygen_ = std::make_unique<heongpu::HEKeyGenerator<Scheme>>(context_);
        secret_key_ = std::make_unique<heongpu::Secretkey<Scheme>>(context_);
        keygen_->generate_secret_key(*secret_key_);

        public_key_ = std::make_unique<heongpu::Publickey<Scheme>>(context_);
        keygen_->generate_public_key(*public_key_, *secret_key_);

        relin_key_ = std::make_unique<heongpu::Relinkey<Scheme>>(context_);
        keygen_->generate_relin_key(*relin_key_, *secret_key_);

        encoder_ = std::make_unique<heongpu::HEEncoder<Scheme>>(context_);
        encryptor_ =
            std::make_unique<heongpu::HEEncryptor<Scheme>>(context_,
                                                           *public_key_);
        decryptor_ =
            std::make_unique<heongpu::HEDecryptor<Scheme>>(context_,
                                                           *secret_key_);
        operators_ = std::make_unique<heongpu::HEArithmeticOperator<Scheme>>(
            context_, *encoder_);

        heongpu::BootstrappingConfig boot_cfg(cfg.ctos_piece, cfg.stoc_piece,
                                              cfg.taylor_number,
                                              cfg.less_key_mode);
        operators_->generate_bootstrapping_params(
            default_scale_, boot_cfg,
            heongpu::arithmetic_bootstrapping_type::REGULAR_BOOTSTRAPPING);

        std::vector<int> boot_shifts = operators_->bootstrapping_key_indexs();
        std::vector<int> shifts = collect_required_shifts();
        shifts.insert(shifts.end(), boot_shifts.begin(), boot_shifts.end());
        shifts = unique_sorted(shifts);

        galois_key_ = std::make_unique<heongpu::Galoiskey<Scheme>>(context_,
                                                                    shifts);
        keygen_->generate_galois_key(*galois_key_, *secret_key_);
    }

    double default_scale() const { return default_scale_; }

    Ptxt encode(const std::vector<double>& vec, int target_depth,
                int plaintext_num_slots)
    {
        Ptxt plain = encode_full_with_scale(vec, default_scale_,
                                            plaintext_num_slots);
        drop_plain_to_depth(plain, target_depth);
        check_cuda("encode_plain");
        return plain;
    }

    Ptxt encode(double val, int target_depth, int plaintext_num_slots)
    {
        if (plaintext_num_slots <= 0) {
            plaintext_num_slots = num_slots;
        }
        std::vector<double> vec(static_cast<size_t>(plaintext_num_slots), val);
        return encode(vec, target_depth, plaintext_num_slots);
    }

    Ctxt encrypt(const std::vector<double>& vec, int target_depth = 0,
                 int plaintext_num_slots = 0)
    {
        Ptxt p = encode_full(vec, plaintext_num_slots);
        Ctxt c(context_);
        encryptor_->encrypt(c, p);
        check_cuda("encrypt");
        if (target_depth > 0) {
            drop_to_depth(c, target_depth);
        }
        check_cuda("drop_to_depth");
        return c;
    }

    Ctxt encrypt_ptxt(Ptxt& p)
    {
        Ctxt c(context_);
        encryptor_->encrypt(c, p);
        return c;
    }

    Ptxt decrypt(const Ctxt& c)
    {
        Ptxt p(context_);
        decryptor_->decrypt(p, const_cast<Ctxt&>(c));
        return p;
    }

    std::vector<double> decrypt_tovector(const Ctxt& c, int slots)
    {
        Ptxt p(context_);
        decryptor_->decrypt(p, const_cast<Ctxt&>(c));
        std::vector<double> vec;
        encoder_->decode(vec, p);
        if (slots > 0 && static_cast<int>(vec.size()) > slots) {
            vec.resize(static_cast<size_t>(slots));
        }
        return vec;
    }

    Ctxt add(const Ctxt& c1, const Ctxt& c2)
    {
        Ctxt out(context_);
        operators_->add(const_cast<Ctxt&>(c1), const_cast<Ctxt&>(c2), out);
        check_cuda("add");
        return out;
    }

    Ctxt add_plain(const Ctxt& c, const Ptxt& p)
    {
        if (debug_cuda && p.depth() != c.depth()) {
            std::cerr << "add_plain depth mismatch c=" << c.depth()
                      << " p=" << p.depth();
            if (!debug_label.empty()) {
                std::cerr << " [" << debug_label << "]";
            }
            std::cerr << std::endl;
        }
        Ctxt out(context_);
        operators_->add_plain(const_cast<Ctxt&>(c), const_cast<Ptxt&>(p), out);
        check_cuda("add_plain");
        return out;
    }

    Ctxt mult(const Ctxt& c, double d)
    {
        Ctxt out(context_);
        operators_->multiply_plain(const_cast<Ctxt&>(c), d, out,
                                   c.scale());
        operators_->rescale_inplace(out);
        mul_count++;
        rescale_count++;
        check_cuda("mult_const");
        return out;
    }

    Ctxt mult(const Ctxt& c, const Ptxt& p)
    {
        Ctxt out(context_);
        operators_->multiply_plain(const_cast<Ctxt&>(c),
                                   const_cast<Ptxt&>(p), out);
        operators_->rescale_inplace(out);
        mul_count++;
        rescale_count++;
        check_cuda("mult_plain");
        return out;
    }

    Ctxt mult_mask(const Ctxt& c, const Ptxt& p)
    {
        Ctxt out = c;
        operators_->multiply_plain_mask_inplace(out, const_cast<Ptxt&>(p));
        mul_count++;
        check_cuda("mult_mask");
        return out;
    }

    Ctxt rotate_vector(const Ctxt& c, int steps)
    {
        Ctxt out(context_);
        operators_->rotate_rows(const_cast<Ctxt&>(c), out, *galois_key_, steps);
        rot_count++;
        check_cuda("rotate");
        return out;
    }

    Ctxt bootstrap(const Ctxt& c, bool timing = false)
    {
        auto start = utils::start_time();
        Ctxt out = operators_->regular_bootstrapping(
            const_cast<Ctxt&>(c), *galois_key_, *relin_key_);
        boot_count++;
        check_cuda("bootstrap");
        if (timing) {
            utils::print_duration(start, "Bootstrapping");
        }
        return out;
    }

    Ctxt relu(const Ctxt& c, double scale, bool timing = false)
    {
        auto start = utils::start_time();
        std::vector<double> coeffs = relu_coefficients(scale, relu_degree);
        Ctxt tmp = const_cast<Ctxt&>(c);
        Ctxt out = operators_->evaluate_poly_monomial(tmp, c.scale(), coeffs,
                                                      *relin_key_);
        check_cuda("relu_poly");
        if (timing) {
            utils::print_duration(start,
                                  "ReLU d = " + std::to_string(relu_degree));
        }
        return out;
    }

    void print(const Ctxt& c, int slots, const std::string& prefix)
    {
        std::vector<double> v = decrypt_tovector(c, slots);
        std::cout << prefix << "[ ";
        std::cout << std::fixed << std::setprecision(3);
        for (int i = 0; i < slots; i++) {
            double val = v[static_cast<size_t>(i)];
            std::string sign = val >= 0 ? " " : "-";
            if (val < 0) {
                val = -val;
            }
            if (i == slots - 1) {
                std::cout << sign << val << " ]";
            } else {
                if (std::abs(val) < 1e-8) {
                    std::cout << " 0.0000000000" << ", ";
                } else {
                    std::cout << sign << val << ", ";
                }
            }
        }
        std::cout << std::endl;
    }

    void print_min_max(const Ctxt& c)
    {
        std::vector<double> v = decrypt_tovector(c, num_slots);
        auto minmax = std::minmax_element(v.begin(), v.end());
        std::cout << "min: " << *minmax.first << ", max: " << *minmax.second
                  << std::endl;
    }

    double scale(const Ctxt& c) const { return c.scale(); }

    int level(const Ctxt& c) const { return c.level(); }

    void print_level_scale(const Ctxt& c, const std::string& label) const
    {
        std::cout << label << " level=" << c.level() << " scale=" << c.scale()
                  << std::endl;
    }

    // CNN functions
    Ctxt convbn_initial(const Ctxt& in, double scale = 0.5,
                        bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 32;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        if (debug_cuda) {
            debug_label = "convbn_initial bias encode";
        }
        std::vector<double> bias_values = utils::read_values_from_file(
            weights_dir + "/conv1bn1-bias.bin", scale);

        Ctxt finalsum(context_);
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<Ctxt> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                if (debug_cuda) {
                    debug_label = "convbn_initial ch=" + std::to_string(j) +
                                  " k=" + std::to_string(k + 1) + " encode";
                }
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/conv1bn1-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                Ptxt encoded = encode(values, in.depth(), 16384);
                if (debug_cuda) {
                    debug_label = "convbn_initial ch=" + std::to_string(j) +
                                  " k=" + std::to_string(k + 1) + " mult";
                }
                k_rows.push_back(mult(c_rotations[k], encoded));
            }

            Ctxt sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            Ctxt res = add(sum, rotate_vector(sum, 1024));
            res = add(res, rotate_vector(rotate_vector(sum, 1024), 1024));
            res = mult_mask(res, mask_from_to(0, 1024, res.depth()));

            if (!init) {
                finalsum = rotate_vector(res, 1024);
                init = true;
            } else {
                finalsum = add(finalsum, res);
                finalsum = rotate_vector(finalsum, 1024);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn_initial bias encode";
        }
        Ptxt bias = encode(bias_values, finalsum.depth(), 16384);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Initial layer");
        }

        return finalsum;
    }

    Ctxt convbn(const Ctxt& in, int layer, int n, double scale = 0.5,
                bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 32;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        std::vector<double> bias_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        Ctxt finalsum(context_);
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<Ctxt> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                Ptxt encoded = encode(values, in.depth(), 16384);
                k_rows.push_back(mult(c_rotations[k], encoded));
            }

            Ctxt sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            if (!init) {
                finalsum = rotate_vector(sum, -1024);
                init = true;
            } else {
                finalsum = add(finalsum, sum);
                finalsum = rotate_vector(finalsum, -1024);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn bias encode";
        }
        Ptxt bias = encode(bias_values, finalsum.depth(), 16384);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    Ctxt convbn2(const Ctxt& in, int layer, int n, double scale = 0.5,
                 bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 16;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        std::vector<double> bias_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        Ctxt finalsum(context_);
        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<Ctxt> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                Ptxt encoded = encode(values, in.depth(), 8192);
                k_rows.push_back(mult(c_rotations[k], encoded));
            }

            Ctxt sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            if (!init) {
                finalsum = rotate_vector(sum, -256);
                init = true;
            } else {
                finalsum = add(finalsum, sum);
                finalsum = rotate_vector(finalsum, -256);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn2 bias encode";
        }
        Ptxt bias = encode(bias_values, finalsum.depth(), 8192);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    Ctxt convbn3(const Ctxt& in, int layer, int n, double scale = 0.5,
                 bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 8;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), -img_width));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -padding), img_width));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, padding), img_width));

        std::vector<double> bias_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias.bin",
            scale);

        Ctxt finalsum(context_);
        bool init = false;

        for (int j = 0; j < 64; j++) {
            std::vector<Ctxt> k_rows;
            k_rows.reserve(9);
            for (int k = 0; k < 9; k++) {
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                Ptxt encoded = encode(values, in.depth(), 4096);
                k_rows.push_back(mult(c_rotations[k], encoded));
            }

            Ctxt sum = k_rows[0];
            for (size_t i = 1; i < k_rows.size(); i++) {
                sum = add(sum, k_rows[i]);
            }

            if (!init) {
                finalsum = rotate_vector(sum, -64);
                init = true;
            } else {
                finalsum = add(finalsum, sum);
                finalsum = rotate_vector(finalsum, -64);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn3 bias encode";
        }
        Ptxt bias = encode(bias_values, finalsum.depth(), 4096);
        finalsum = add_plain(finalsum, bias);

        if (timing) {
            utils::print_duration(start, "Block" + std::to_string(layer) +
                                             " - convbn" +
                                             std::to_string(n));
        }
        return finalsum;
    }

    std::vector<Ctxt> convbn1632sx(const Ctxt& in, int layer, int n,
                                   double scale = 0.5, bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 32;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -(img_width)), -padding));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -(img_width)), padding));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, (img_width)), -padding));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, (img_width)), padding));

        std::vector<double> bias1_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        std::vector<double> bias2_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        Ctxt finalSum016(context_);
        Ctxt finalSum1632(context_);
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<Ctxt> k_rows016;
            std::vector<Ctxt> k_rows1632;
            k_rows016.reserve(9);
            k_rows1632.reserve(9);

            for (int k = 0; k < 9; k++) {
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                k_rows016.push_back(
                    mult(c_rotations[k], encode(values, in.depth(), 16384)));

                values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 16) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                k_rows1632.push_back(
                    mult(c_rotations[k], encode(values, in.depth(), 16384)));
            }

            Ctxt sum016 = k_rows016[0];
            Ctxt sum1632 = k_rows1632[0];
            for (size_t i = 1; i < k_rows016.size(); i++) {
                sum016 = add(sum016, k_rows016[i]);
                sum1632 = add(sum1632, k_rows1632[i]);
            }

            if (!init) {
                finalSum016 = rotate_vector(sum016, -1024);
                finalSum1632 = rotate_vector(sum1632, -1024);
                init = true;
            } else {
                finalSum016 = add(finalSum016, sum016);
                finalSum016 = rotate_vector(finalSum016, -1024);
                finalSum1632 = add(finalSum1632, sum1632);
                finalSum1632 = rotate_vector(finalSum1632, -1024);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn1632sx bias encode";
        }
        Ptxt bias1 = encode(bias1_values, finalSum016.depth(), 16384);
        Ptxt bias2 = encode(bias2_values, finalSum1632.depth(), 16384);
        finalSum016 = add_plain(finalSum016, bias1);
        finalSum1632 = add_plain(finalSum1632, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnSx" +
                                             std::to_string(n));
        }
        return {finalSum016, finalSum1632};
    }

    std::vector<Ctxt> convbn1632dx(const Ctxt& in, int layer, int n,
                                   double scale = 0.5, bool timing = false)
    {
        auto start = utils::start_time();

        std::vector<double> bias1_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        std::vector<double> bias2_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        Ctxt finalSum016(context_);
        Ctxt finalSum1632(context_);
        bool init = false;

        for (int j = 0; j < 16; j++) {
            std::vector<double> values = utils::read_values_from_file(
                weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                    std::to_string(n) + "bn" + std::to_string(n) + "-ch" +
                    std::to_string(j) + "-k" + std::to_string(1) + ".bin",
                scale);
            Ctxt sum016 = mult(in, encode(values, in.depth(), num_slots));

            values = utils::read_values_from_file(
                weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                    std::to_string(n) + "bn" + std::to_string(n) + "-ch" +
                    std::to_string(j + 16) + "-k" + std::to_string(1) + ".bin",
                scale);
            Ctxt sum1632 = mult(in, encode(values, in.depth(), num_slots));

            if (!init) {
                finalSum016 = rotate_vector(sum016, -1024);
                finalSum1632 = rotate_vector(sum1632, -1024);
                init = true;
            } else {
                finalSum016 = add(finalSum016, sum016);
                finalSum016 = rotate_vector(finalSum016, -1024);
                finalSum1632 = add(finalSum1632, sum1632);
                finalSum1632 = rotate_vector(finalSum1632, -1024);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn1632dx bias encode";
        }
        Ptxt bias1 = encode(bias1_values, finalSum016.depth(), 16384);
        Ptxt bias2 = encode(bias2_values, finalSum1632.depth(), 16384);
        finalSum016 = add_plain(finalSum016, bias1);
        finalSum1632 = add_plain(finalSum1632, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnDx" +
                                             std::to_string(n));
        }

        return {finalSum016, finalSum1632};
    }

    std::vector<Ctxt> convbn3264sx(const Ctxt& in, int layer, int n,
                                   double scale = 0.5, bool timing = false)
    {
        auto start = utils::start_time();

        int img_width = 16;
        int padding = 1;

        std::vector<Ctxt> c_rotations;
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -(img_width)), -padding));
        c_rotations.push_back(rotate_vector(in, -img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, -(img_width)), padding));
        c_rotations.push_back(rotate_vector(in, -padding));
        c_rotations.push_back(in);
        c_rotations.push_back(rotate_vector(in, padding));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, (img_width)), -padding));
        c_rotations.push_back(rotate_vector(in, img_width));
        c_rotations.push_back(
            rotate_vector(rotate_vector(in, (img_width)), padding));

        std::vector<double> bias1_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        std::vector<double> bias2_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        Ctxt finalSum032(context_);
        Ctxt finalSum3264(context_);
        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<Ctxt> k_rows032;
            std::vector<Ctxt> k_rows3264;
            k_rows032.reserve(9);
            k_rows3264.reserve(9);

            for (int k = 0; k < 9; k++) {
                std::vector<double> values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                k_rows032.push_back(
                    mult(c_rotations[k], encode(values, in.depth(), 8192)));

                values = utils::read_values_from_file(
                    weights_dir + "/layer" + std::to_string(layer) + "-conv" +
                        std::to_string(n) + "bn" + std::to_string(n) +
                        "-ch" + std::to_string(j + 32) + "-k" +
                        std::to_string(k + 1) + ".bin",
                    scale);
                k_rows3264.push_back(
                    mult(c_rotations[k], encode(values, in.depth(), 8192)));
            }

            Ctxt sum032 = k_rows032[0];
            Ctxt sum3264 = k_rows3264[0];
            for (size_t i = 1; i < k_rows032.size(); i++) {
                sum032 = add(sum032, k_rows032[i]);
                sum3264 = add(sum3264, k_rows3264[i]);
            }

            if (!init) {
                finalSum032 = rotate_vector(sum032, -256);
                finalSum3264 = rotate_vector(sum3264, -256);
                init = true;
            } else {
                finalSum032 = add(finalSum032, sum032);
                finalSum032 = rotate_vector(finalSum032, -256);
                finalSum3264 = add(finalSum3264, sum3264);
                finalSum3264 = rotate_vector(finalSum3264, -256);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn3264sx bias encode";
        }
        Ptxt bias1 = encode(bias1_values, finalSum032.depth(), 8192);
        Ptxt bias2 = encode(bias2_values, finalSum3264.depth(), 8192);
        finalSum032 = add_plain(finalSum032, bias1);
        finalSum3264 = add_plain(finalSum3264, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnSx" +
                                             std::to_string(n));
        }

        return {finalSum032, finalSum3264};
    }

    std::vector<Ctxt> convbn3264dx(const Ctxt& in, int layer, int n,
                                   double scale = 0.5, bool timing = false)
    {
        auto start = utils::start_time();

        std::vector<double> bias1_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias1.bin",
            scale);
        std::vector<double> bias2_values = utils::read_values_from_file(
            weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                std::to_string(n) + "bn" + std::to_string(n) + "-bias2.bin",
            scale);

        Ctxt finalSum032(context_);
        Ctxt finalSum3264(context_);
        bool init = false;

        for (int j = 0; j < 32; j++) {
            std::vector<double> values = utils::read_values_from_file(
                weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                    std::to_string(n) + "bn" + std::to_string(n) + "-ch" +
                    std::to_string(j) + "-k" + std::to_string(1) + ".bin",
                scale);
            Ctxt sum032 = mult(in, encode(values, in.depth(), 8192));

            values = utils::read_values_from_file(
                weights_dir + "/layer" + std::to_string(layer) + "dx-conv" +
                    std::to_string(n) + "bn" + std::to_string(n) + "-ch" +
                    std::to_string(j + 32) + "-k" + std::to_string(1) + ".bin",
                scale);
            Ctxt sum3264 = mult(in, encode(values, in.depth(), 8192));

            if (!init) {
                finalSum032 = rotate_vector(sum032, -256);
                finalSum3264 = rotate_vector(sum3264, -256);
                init = true;
            } else {
                finalSum032 = add(finalSum032, sum032);
                finalSum032 = rotate_vector(finalSum032, -256);
                finalSum3264 = add(finalSum3264, sum3264);
                finalSum3264 = rotate_vector(finalSum3264, -256);
            }
        }

        if (debug_cuda) {
            debug_label = "convbn3264dx bias encode";
        }
        Ptxt bias1 = encode(bias1_values, finalSum032.depth(), 8192);
        Ptxt bias2 = encode(bias2_values, finalSum3264.depth(), 8192);
        finalSum032 = add_plain(finalSum032, bias1);
        finalSum3264 = add_plain(finalSum3264, bias2);

        if (timing) {
            utils::print_duration(start, "Block " + std::to_string(layer) +
                                             " - convbnDx" +
                                             std::to_string(n));
        }

        return {finalSum032, finalSum3264};
    }

    Ctxt downsample1024to256(const Ctxt& c1, const Ctxt& c2)
    {
        num_slots = 16384 * 2;

        Ctxt fullpack = add(mult_mask(c1, mask_first_n(16384, c1.depth())),
                            mult_mask(c2, mask_second_n(16384, c2.depth())));

        fullpack = mult_mask(
            add(fullpack, rotate_vector(fullpack, 1)),
            gen_mask(2, fullpack.depth()));
        fullpack = mult_mask(
            add(fullpack, rotate_vector(rotate_vector(fullpack, 1), 1)),
            gen_mask(4, fullpack.depth()));
        fullpack = mult_mask(add(fullpack, rotate_vector(fullpack, 4)),
                             gen_mask(8, fullpack.depth()));
        fullpack = add(fullpack, rotate_vector(fullpack, 8));

        Ctxt downsampledrows = encrypt({0}, c1.depth());

        for (int i = 0; i < 16; i++) {
            Ctxt masked = mult_mask(fullpack,
                                    mask_first_n_mod(16, 1024, i,
                                                     fullpack.depth()));
            downsampledrows = add(downsampledrows, masked);
            if (i < 15) {
                fullpack = rotate_vector(fullpack, 64 - 16);
            }
        }

        Ctxt downsampledchannels = encrypt({0}, c1.depth());
        for (int i = 0; i < 32; i++) {
            Ctxt masked =
                mult_mask(downsampledrows,
                          mask_channel(i, downsampledrows.depth()));
            downsampledchannels = add(downsampledchannels, masked);
            downsampledchannels =
                rotate_vector(downsampledchannels, -(1024 - 256));
        }

        downsampledchannels =
            rotate_vector(downsampledchannels, (1024 - 256) * 32);
        downsampledchannels =
            add(downsampledchannels, rotate_vector(downsampledchannels, -8192));
        downsampledchannels = add(
            downsampledchannels,
            rotate_vector(rotate_vector(downsampledchannels, -8192), -8192));

        num_slots = 8192;
        return downsampledchannels;
    }

    Ctxt downsample256to64(const Ctxt& c1, const Ctxt& c2)
    {
        num_slots = 8192 * 2;
        Ctxt fullpack = add(mult_mask(c1, mask_first_n(8192, c1.depth())),
                            mult_mask(c2, mask_second_n(8192, c2.depth())));

        fullpack = mult_mask(
            add(fullpack, rotate_vector(fullpack, 1)),
            gen_mask(2, fullpack.depth()));
        fullpack = mult_mask(
            add(fullpack, rotate_vector(rotate_vector(fullpack, 1), 1)),
            gen_mask(4, fullpack.depth()));
        fullpack = add(fullpack, rotate_vector(fullpack, 4));

        Ctxt downsampledrows = encrypt({0}, c1.depth());

        for (int i = 0; i < 32; i++) {
            Ctxt masked = mult_mask(fullpack,
                                    mask_first_n_mod2(8, 256, i,
                                                      fullpack.depth()));
            downsampledrows = add(downsampledrows, masked);
            if (i < 31) {
                fullpack = rotate_vector(fullpack, 32 - 8);
            }
        }

        Ctxt downsampledchannels = encrypt({0}, c1.depth());
        for (int i = 0; i < 64; i++) {
            Ctxt masked =
                mult_mask(downsampledrows,
                          mask_channel_2(i, downsampledrows.depth()));
            downsampledchannels = add(downsampledchannels, masked);
            downsampledchannels =
                rotate_vector(downsampledchannels, -(256 - 64));
        }

        downsampledchannels =
            rotate_vector(downsampledchannels, (256 - 64) * 64);
        downsampledchannels =
            add(downsampledchannels, rotate_vector(downsampledchannels, -4096));
        downsampledchannels = add(
            downsampledchannels,
            rotate_vector(rotate_vector(downsampledchannels, -4096), -4096));

        num_slots = 4096;
        return downsampledchannels;
    }

    Ctxt rotsum(const Ctxt& in, int slots)
    {
        Ctxt result = in;
        for (int i = 0; i < static_cast<int>(std::log2(slots)); i++) {
            result = add(result, rotate_vector(result, 1 << i));
        }
        return result;
    }

    Ctxt rotsum_padded(const Ctxt& in, int slots)
    {
        Ctxt result = in;
        for (int i = 0; i < static_cast<int>(std::log2(slots)); i++) {
            result = add(result, rotate_vector(result, slots * (1 << i)));
        }
        return result;
    }

    Ctxt repeat(const Ctxt& in, int slots)
    {
        return rotate_vector(rotsum(in, slots), -slots + 1);
    }

    // Masks
    Ptxt gen_mask(int n, int target_depth)
    {
        std::vector<double> mask(num_slots, 0.0);
        int copy_interval = n;
        for (int i = 0; i < num_slots; i++) {
            if (copy_interval > 0) {
                mask[static_cast<size_t>(i)] = 1.0;
            }
            copy_interval--;
            if (copy_interval <= -n) {
                copy_interval = n;
            }
        }
        return encode_mask(mask, target_depth);
    }

    Ptxt mask_first_n(int n, int target_depth)
    {
        std::vector<double> mask(num_slots, 0.0);
        for (int i = 0; i < std::min(n, num_slots); i++) {
            mask[static_cast<size_t>(i)] = 1.0;
        }
        return encode_mask(mask, target_depth);
    }

    Ptxt mask_second_n(int n, int target_depth)
    {
        std::vector<double> mask(num_slots, 0.0);
        for (int i = n; i < num_slots; i++) {
            mask[static_cast<size_t>(i)] = 1.0;
        }
        return encode_mask(mask, target_depth);
    }

    Ptxt mask_first_n_mod(int n, int padding, int pos, int target_depth)
    {
        std::vector<double> mask;
        mask.reserve(16384 * 2);
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < (pos * n); j++) {
                mask.push_back(0);
            }
            for (int j = 0; j < n; j++) {
                mask.push_back(1);
            }
            for (int j = 0; j < (padding - n - (pos * n)); j++) {
                mask.push_back(0);
            }
        }
        return encode_mask(mask, target_depth);
    }

    Ptxt mask_first_n_mod2(int n, int padding, int pos, int target_depth)
    {
        std::vector<double> mask;
        mask.reserve(8192 * 2);
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < (pos * n); j++) {
                mask.push_back(0);
            }
            for (int j = 0; j < n; j++) {
                mask.push_back(1);
            }
            for (int j = 0; j < (padding - n - (pos * n)); j++) {
                mask.push_back(0);
            }
        }
        return encode_mask(mask, target_depth);
    }

    Ptxt mask_channel(int n, int target_depth)
    {
        std::vector<double> mask;
        mask.reserve(16384 * 2);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 1024; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 256; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 1024 - 256; i++) {
            mask.push_back(0);
        }

        for (int i = 0; i < 31 - n; i++) {
            for (int j = 0; j < 1024; j++) {
                mask.push_back(0);
            }
        }

        return encode_mask(mask, target_depth);
    }

    Ptxt mask_channel_2(int n, int target_depth)
    {
        std::vector<double> mask;
        mask.reserve(8192 * 2);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        for (int i = 0; i < 64; i++) {
            mask.push_back(1);
        }

        for (int i = 0; i < 256 - 64; i++) {
            mask.push_back(0);
        }

        for (int i = 0; i < 63 - n; i++) {
            for (int j = 0; j < 256; j++) {
                mask.push_back(0);
            }
        }

        return encode_mask(mask, target_depth);
    }

    Ptxt mask_mod(int n, int target_depth, double custom_val)
    {
        std::vector<double> vec(num_slots, 0.0);
        for (int i = 0; i < num_slots; i++) {
            if (i % n == 0) {
                vec[static_cast<size_t>(i)] = custom_val;
            }
        }
        return encode(vec, target_depth, num_slots);
    }

    Ptxt mask_from_to(int from, int to, int target_depth)
    {
        std::vector<double> vec(num_slots, 0.0);
        for (int i = from; i < std::min(to, num_slots); i++) {
            vec[static_cast<size_t>(i)] = 1.0;
        }
        return encode_mask(vec, target_depth);
    }

  private:
    heongpu::HEContext<Scheme> context_;
    std::unique_ptr<heongpu::HEKeyGenerator<Scheme>> keygen_;
    std::unique_ptr<heongpu::Secretkey<Scheme>> secret_key_;
    std::unique_ptr<heongpu::Publickey<Scheme>> public_key_;
    std::unique_ptr<heongpu::Relinkey<Scheme>> relin_key_;
    std::unique_ptr<heongpu::Galoiskey<Scheme>> galois_key_;
    std::unique_ptr<heongpu::HEEncoder<Scheme>> encoder_;
    std::unique_ptr<heongpu::HEEncryptor<Scheme>> encryptor_;
    std::unique_ptr<heongpu::HEDecryptor<Scheme>> decryptor_;
    std::unique_ptr<heongpu::HEArithmeticOperator<Scheme>> operators_;

    double default_scale_ = std::pow(2.0, 50);
    std::unordered_map<std::string, std::vector<double>> relu_cache_;

    static std::vector<int> unique_sorted(std::vector<int> values)
    {
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        return values;
    }

    static std::vector<double> chebyshev_to_monomial(
        const std::vector<Complex64>& cheb, double a, double b)
    {
        const int degree = static_cast<int>(cheb.size()) - 1;
        std::vector<double> poly_t(static_cast<size_t>(degree + 1), 0.0);

        std::vector<double> T_prev(1, 1.0);
        std::vector<double> T_curr(2, 0.0);
        T_curr[1] = 1.0;

        auto add_scaled = [&](const std::vector<double>& src, double scale) {
            if (src.empty()) {
                return;
            }
            if (poly_t.size() < src.size()) {
                poly_t.resize(src.size(), 0.0);
            }
            for (size_t i = 0; i < src.size(); i++) {
                poly_t[i] += src[i] * scale;
            }
        };

        add_scaled(T_prev, cheb[0].real());
        if (degree >= 1) {
            add_scaled(T_curr, cheb[1].real());
        }

        for (int k = 2; k <= degree; k++) {
            std::vector<double> T_next(T_curr.size() + 1, 0.0);
            for (size_t i = 0; i < T_curr.size(); i++) {
                T_next[i + 1] += 2.0 * T_curr[i];
            }
            for (size_t i = 0; i < T_prev.size(); i++) {
                T_next[i] -= T_prev[i];
            }
            add_scaled(T_next, cheb[static_cast<size_t>(k)].real());
            T_prev = T_curr;
            T_curr = std::move(T_next);
        }

        double alpha = 2.0 / (b - a);
        double beta = -(a + b) / (b - a);

        std::vector<double> poly_x(1, 0.0);
        std::vector<double> pow_poly(1, 1.0);

        for (int i = 0; i <= degree; i++) {
            if (poly_x.size() < pow_poly.size()) {
                poly_x.resize(pow_poly.size(), 0.0);
            }
            for (size_t j = 0; j < pow_poly.size(); j++) {
                poly_x[j] += poly_t[static_cast<size_t>(i)] * pow_poly[j];
            }

            std::vector<double> next_pow(pow_poly.size() + 1, 0.0);
            for (size_t j = 0; j < pow_poly.size(); j++) {
                next_pow[j] += pow_poly[j] * beta;
                next_pow[j + 1] += pow_poly[j] * alpha;
            }
            pow_poly = std::move(next_pow);
        }

        return poly_x;
    }

    std::vector<double> relu_coefficients(double scale, int degree)
    {
        std::string key = std::to_string(scale) + "|" + std::to_string(degree);
        auto it = relu_cache_.find(key);
        if (it != relu_cache_.end()) {
            return it->second;
        }

        auto func = [scale](Complex64 x) -> Complex64 {
            double real = x.real();
            if (real < 0) {
                return Complex64(0.0, 0.0);
            }
            return Complex64(real / scale, 0.0);
        };

        std::vector<Complex64> cheb =
            heongpu::approximate_function(func, -1.0, 1.0, degree);
        std::vector<double> mono = chebyshev_to_monomial(cheb, -1.0, 1.0);
        relu_cache_[key] = mono;
        return mono;
    }

    Ptxt encode_mask(const std::vector<double>& vec, int target_depth)
    {
        if (debug_cuda) {
            debug_label = "mask encode";
            std::cout << "encode_mask depth=" << target_depth
                      << " slots=" << vec.size() << std::endl;
        }
        Ptxt plain = encode_full_with_scale(vec, 1.0, vec.size());
        drop_plain_to_depth(plain, target_depth);
        check_cuda("encode_mask");
        return plain;
    }

    std::vector<int> collect_required_shifts() const
    {
        std::vector<int> shifts = {
            1, -1, 2, -2, 4, -4, 7, -7, 8, -8, 9, -9, 15, -15, 16, -16, 17,
            -17, 24, -24, 31, -31, 32, -32, 33, -33, 48, -48, 63, -63, 64,
            -64, 128, -128, 192, -192, 256, -256, 512, -512, 768, -768, 1024,
            -1024, 2048, -2048, 3072, -3072, 4096, -4096, 8192, -8192,
            12288, -12288, 24576, -24576
        };

        return shifts;
    }

    Ptxt encode_full(const std::vector<double>& vec, int plaintext_num_slots)
    {
        return encode_full_with_scale(vec, default_scale_, plaintext_num_slots);
    }

    void drop_to_depth(Ctxt& c, int target_depth)
    {
        while (c.depth() < target_depth) {
            operators_->mod_drop_inplace(c);
        }
    }

    void drop_plain_to_depth(Ptxt& p, int target_depth)
    {
        while (p.depth() < target_depth) {
            operators_->mod_drop_inplace(p);
        }
    }

    Ptxt encode_full_with_scale(const std::vector<double>& vec, double scale,
                                int plaintext_num_slots)
    {
        if (plaintext_num_slots <= 0) {
            plaintext_num_slots = num_slots;
        }
        std::vector<double> msg = vec;
        if (static_cast<int>(msg.size()) < plaintext_num_slots) {
            msg.resize(static_cast<size_t>(plaintext_num_slots), 0.0);
        }
        if (debug_cuda) {
            std::cout << "encode_full scale=" << scale
                      << " slots=" << plaintext_num_slots
                      << " msg_size=" << msg.size() << std::endl;
        }
        Ptxt plain(context_);
        encoder_->encode(plain, msg, scale);
        check_cuda("encode_full");
        return plain;
    }

    void check_cuda(const char* label)
    {
        if (!debug_cuda) {
            return;
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after " << label;
            if (!debug_label.empty()) {
                std::cerr << " [" << debug_label << "]";
            }
            std::cerr << ": "
                      << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("CUDA failure");
        }
        cudaGetLastError();
    }
};

} // namespace lowmem

#endif // LOWMEM_RESNET20_ADAPTER_H
