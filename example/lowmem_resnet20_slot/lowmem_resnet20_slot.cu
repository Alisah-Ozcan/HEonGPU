#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "LowMemAdapter.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GREEN_TEXT ""

using lowmem::Ctxt;
using lowmem::FHEController;
using lowmem::HEConfig;

static FHEController controller;

static std::string data_dir;
static std::string weights_dir;
static std::string input_filename = "luis.png";
static int verbose = 0;
static bool plain = false;

static void parse_args(int argc, char* argv[])
{
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data_dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--weights_dir" && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_filename = argv[++i];
        } else if (arg == "--verbose" && i + 1 < argc) {
            verbose = std::atoi(argv[++i]);
        } else if (arg == "--plain") {
            plain = true;
        }
    }
}

static std::filesystem::path resolve_path(const std::filesystem::path& base,
                                          const std::string& path)
{
    if (path.empty()) {
        return {};
    }
    std::filesystem::path p(path);
    if (p.is_absolute()) {
        return p;
    }
    return base / p;
}

static std::vector<double> read_image(const std::string& filename)
{
    int width = 32;
    int height = 32;
    int channels = 3;
    unsigned char* image_data =
        stbi_load(filename.c_str(), &width, &height, &channels, 0);

    if (!image_data) {
        std::cerr << "Could not load the image in " << filename << std::endl;
        return {};
    }

    std::vector<double> imageVector;
    imageVector.reserve(width * height * channels);

    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0);
    }
    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(
            static_cast<double>(image_data[1 + 3 * i]) / 255.0);
    }
    for (int i = 0; i < width * height; ++i) {
        imageVector.push_back(
            static_cast<double>(image_data[2 + 3 * i]) / 255.0);
    }

    stbi_image_free(image_data);
    return imageVector;
}

static Ctxt initial_layer(const Ctxt& in)
{
    double scale = 0.90;
    Ctxt res = controller.convbn_initial(in, scale, verbose > 1);
    res = controller.relu(res, scale, verbose > 1);
    return res;
}

static Ctxt layer1(const Ctxt& in)
{
    bool timing = verbose > 1;
    double scale = 1.00;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt res1 = controller.convbn(in, 1, 1, scale, timing);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);

    scale = 0.52;

    res1 = controller.convbn(res1, 1, 2, scale, timing);
    res1 = controller.add(res1, controller.mult(in, scale));
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 1---" << std::endl;

    scale = 0.55;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn(res1, 2, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.36;

    res2 = controller.convbn(res2, 2, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 2---" << std::endl;

    scale = 0.63;

    if (verbose > 1) std::cout << "---Start: Layer1 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn(res2, 3, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.42;

    res3 = controller.convbn(res3, 3, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer1 - Block 3---" << std::endl;

    return res3;
}

static Ctxt layer2(const Ctxt& in)
{
    double scaleSx = 0.57;
    double scaleDx = 0.40;

    bool timing = verbose > 1;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    std::vector<Ctxt> res1sx =
        controller.convbn1632sx(boot_in, 4, 1, scaleSx, timing);
    std::vector<Ctxt> res1dx =
        controller.convbn1632dx(boot_in, 4, 1, scaleDx, timing);

    Ctxt fullpackSx = controller.downsample1024to256(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample1024to256(res1dx[0], res1dx[1]);

    controller.num_slots = 8192;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = controller.convbn2(fullpackSx, 4, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 1---" << std::endl;

    double scale = 0.76;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn2(res1, 5, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.37;

    res2 = controller.convbn2(res2, 5, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 2---" << std::endl;

    scale = 0.63;

    if (verbose > 1) std::cout << "---Start: Layer2 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn2(res2, 6, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.25;

    res3 = controller.convbn2(res3, 6, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer2 - Block 3---" << std::endl;

    return res3;
}

static Ctxt layer3(const Ctxt& in)
{
    double scaleSx = 0.63;
    double scaleDx = 0.40;

    bool timing = verbose > 1;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 1---" << std::endl;
    auto start = utils::start_time();
    Ctxt boot_in = controller.bootstrap(in, timing);

    std::vector<Ctxt> res1sx =
        controller.convbn3264sx(boot_in, 7, 1, scaleSx, timing);
    std::vector<Ctxt> res1dx =
        controller.convbn3264dx(boot_in, 7, 1, scaleDx, timing);

    Ctxt fullpackSx = controller.downsample256to64(res1sx[0], res1sx[1]);
    Ctxt fullpackDx = controller.downsample256to64(res1dx[0], res1dx[1]);

    controller.num_slots = 4096;
    fullpackSx = controller.bootstrap(fullpackSx, timing);

    fullpackSx = controller.relu(fullpackSx, scaleSx, timing);
    fullpackSx = controller.convbn3(fullpackSx, 7, 2, scaleDx, timing);
    Ctxt res1 = controller.add(fullpackSx, fullpackDx);
    res1 = controller.bootstrap(res1, timing);
    res1 = controller.relu(res1, scaleDx, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 1---" << std::endl;

    double scale = 0.57;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 2---" << std::endl;
    start = utils::start_time();
    Ctxt res2 = controller.convbn3(res1, 8, 1, scale, timing);
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);

    scale = 0.33;

    res2 = controller.convbn3(res2, 8, 2, scale, timing);
    res2 = controller.add(res2, controller.mult(res1, scale));
    res2 = controller.bootstrap(res2, timing);
    res2 = controller.relu(res2, scale, timing);
    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 2---" << std::endl;

    scale = 0.69;

    if (verbose > 1) std::cout << "---Start: Layer3 - Block 3---" << std::endl;
    start = utils::start_time();
    Ctxt res3 = controller.convbn3(res2, 9, 1, scale, timing);
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);

    scale = 0.10;

    res3 = controller.convbn3(res3, 9, 2, scale, timing);
    res3 = controller.add(res3, controller.mult(res2, scale));
    res3 = controller.bootstrap(res3, timing);
    res3 = controller.relu(res3, scale, timing);
    res3 = controller.bootstrap(res3, timing);

    if (verbose > 1) utils::print_duration(start, "Total");
    if (verbose > 1) std::cout << "---End  : Layer3 - Block 3---" << std::endl;

    return res3;
}

static Ctxt final_layer(const Ctxt& in)
{
    controller.num_slots = 4096;

    std::vector<double> fc_weight =
        utils::read_fc_weight(weights_dir + "/fc.bin");
    lowmem::Ptxt weight =
        controller.encode(fc_weight, in.depth(), controller.num_slots);

    Ctxt res = controller.rotsum(in, 64);
    res = controller.mult(res,
                          controller.mask_mod(64, res.depth(), 1.0 / 64.0));

    res = controller.repeat(res, 16);
    res = controller.mult(res, weight);
    res = controller.rotsum_padded(res, 64);

    if (verbose >= 0) {
        std::cout << "Decrypting the output..." << std::endl;
        controller.print(res, 10, "Output: ");
    }

    std::vector<double> clear_result = controller.decrypt_tovector(res, 10);
    auto max_it = std::max_element(clear_result.begin(), clear_result.end());
    int index_max = static_cast<int>(std::distance(clear_result.begin(), max_it));

    if (verbose >= 0) {
        std::cout << "The input image is classified as " << YELLOW_TEXT
                  << utils::get_class(index_max) << RESET_COLOR << std::endl;
        std::cout << "The index of max element is " << YELLOW_TEXT << index_max
                  << RESET_COLOR << std::endl;
    }

    if (plain) {
        std::string command =
            "python3 ../LowMemoryFHEResNet20/src/plain/script.py \"" +
            (data_dir + "/" + input_filename) + "\"";
        int return_sys = system(command.c_str());
        if (return_sys == 1) {
            std::cout << "Error launching plain script." << std::endl;
        }
    }

    return res;
}

static void execute_resnet20()
{
    if (verbose >= 0)
        std::cout << "Encrypted ResNet20 classification started." << std::endl;

    std::string input_path = data_dir + "/" + input_filename;
    if (verbose >= 0) {
        std::cout << "Encrypting and classifying " << GREEN_TEXT << input_path
                  << RESET_COLOR << std::endl;
    }

    std::vector<double> input_image = read_image(input_path);
    if (input_image.empty()) {
        std::cerr << "Input image load failed. Check --data_dir and --input."
                  << std::endl;
        return;
    }
    Ctxt in = controller.encrypt(
        input_image,
        controller.circuit_depth - 4 - utils::get_relu_depth(controller.relu_degree));

    auto start = utils::start_time();

    controller.print_level_scale(in, "Input");
    Ctxt firstLayer = initial_layer(in);
    controller.print_level_scale(firstLayer, "Initial layer out");

    auto startLayer = utils::start_time();
    controller.print_level_scale(firstLayer, "Layer 1 in");
    Ctxt resLayer1 = layer1(firstLayer);
    controller.print_level_scale(resLayer1, "Layer 1 out");
    utils::print_duration(startLayer, "Layer 1 took:");

    startLayer = utils::start_time();
    controller.print_level_scale(resLayer1, "Layer 2 in");
    Ctxt resLayer2 = layer2(resLayer1);
    controller.print_level_scale(resLayer2, "Layer 2 out");
    utils::print_duration(startLayer, "Layer 2 took:");

    startLayer = utils::start_time();
    controller.print_level_scale(resLayer2, "Layer 3 in");
    Ctxt resLayer3 = layer3(resLayer2);
    controller.print_level_scale(resLayer3, "Layer 3 out");
    utils::print_duration(startLayer, "Layer 3 took:");

    controller.print_level_scale(resLayer3, "Final layer in");
    Ctxt finalRes = final_layer(resLayer3);
    controller.print_level_scale(finalRes, "Final layer out");

    utils::print_duration_yellow(start,
                                 "The evaluation of the whole circuit took: ");

    std::cout << "Total mul: " << controller.mul_count
              << " rot: " << controller.rot_count
              << " rescale: " << controller.rescale_count
              << " boot: " << controller.boot_count << std::endl;
}

int main(int argc, char* argv[])
{
    cudaSetDevice(0);

    parse_args(argc, argv);

    std::filesystem::path exe_path = std::filesystem::absolute(argv[0]);
    std::filesystem::path repo_root =
        exe_path.parent_path() / ".." / ".." / "..";
    repo_root = std::filesystem::weakly_canonical(repo_root);

    if (data_dir.empty()) {
        data_dir =
            (repo_root / "example/lowmem_resnet20_slot/assets/inputs").string();
    } else {
        data_dir = resolve_path(repo_root, data_dir).string();
    }

    if (weights_dir.empty()) {
        weights_dir =
            (repo_root / "example/lowmem_resnet20_slot/assets/weights").string();
    } else {
        weights_dir = resolve_path(repo_root, weights_dir).string();
    }

    HEConfig cfg;
    cfg.relu_degree = controller.relu_degree;
    controller.weights_dir = weights_dir;
    controller.initialize(cfg);

    execute_resnet20();

    return 0;
}
