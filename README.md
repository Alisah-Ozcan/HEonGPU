# 🚀 **HEonGPU** - A GPU Based Homomorphic Encryption Library

HEonGPU is a high-performance library designed to optimize Fully Homomorphic Encryption (FHE) operations on GPUs. By leveraging the parallel processing power of GPUs, it significantly reduces the computational load of FHE through concurrent execution of complex operations. Its multi-stream architecture enables efficient parallel processing and minimizes the overhead of data transfers between the CPU and GPU. These features make HEonGPU ideal for large-scale encrypted computations, offering reduced latency and improved performance.

The goal of HEonGPU is to provide:
- A high-performance framework for executing FHE schemes, specifically `BFV`, `CKKS` and `TFHE`, by leveraging the parallel processing capabilities of CUDA.
- A user-friendly C++ interface that requires no prior knowledge of GPU programming, with all CUDA kernels encapsulated in easy-to-use classes.
- An optimized multi-stream architecture that ensures efficient memory management and concurrent execution of encrypted computations on the GPU.

For more information about HEonGPU: 
- https://eprint.iacr.org/2024/1543
- https://heongpu.readthedocs.io/

### Current HEonGPU Capabilities and Schemes

<div align="center">

| Capability / Scheme            | HEonGPU   |
|:------------------------------:|:---------:|
| BFV                            | ✓         |
| CKKS                           | ✓         |
| BGV                            | Very Soon |
| TFHE                           | ✓         |
| CKKS Regular Bootstrapping     | ✓         |
| CKKS Slim Bootstrapping        | ✓         |
| CKKS Bit Bootstrapping         | ✓         |
| CKKS Gate Bootstrapping        | ✓         |
| TFHE Gate Bootstrapping        | ✓         |
| Multiparty Computation (MPC)   | ✓         |
| Collective Bootstrapping (MPC) | ✓         |

</div>

## News

### 🚨 **New Feature: [Collective(Distributed) Bootstrapping](example/mpc/)**

The HEonGPU library now delivers `Collective Bootstrapping` for both BFV and CKKS, drawing on the designs introduced by [Mouchet et al.](https://eprint.iacr.org/2020/304.pdf)  and [Balle et al.](https://arxiv.org/pdf/2009.00349) A streamlined CUDA path merges share creation and re-encryption into a single launch, allowing deep multi-party workloads to keep running entirely on the GPU without pausing to reset noise.


### 🚨 **New Scheme: [TFHE (Torus Fully Homomorphic Encryption)](example/basic/12_basic_tfhe.cu)**

The HEonGPU library now includes support for the `TFHE` (Torus Fully Homomorphic Encryption) scheme with GPU acceleration. This enables efficient evaluation of Boolean circuits using fast gate bootstrapping and low-latency parallel execution on modern CUDA-enabled GPUs.

Currently, the implementation supports a fixed parameter set targeting 128-bit security. In upcoming releases, we plan to:

- Make the parameters fully configurable,
- Provide default parameter sets for **128-bit**, **192-bit**, and **256-bit** security levels.
- Further optimize and accelerate the TFHE implementation with improved CUDA kernels and parallelism strategies,
- Introduce native support for **homomorphic unsigned arithmetic** via new types:  
  `huint8`, `huint16`, `huint32`, `huint64`, `huint128`, and `huint256`.


<div align="center">

|           | uint8 | uint16 | uint32 | uint64 | uint128 | uint256 |
|-----------|-------|--------|--------|--------|---------|---------|
| [TFHE-rs](https://github.com/zama-ai/tfhe-rs)   | 31.53 | 31.54  | 31.55  | 32.03  | 33.74   | 58.32   |
| [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931)  | 18.63 | 18.61  | 18.87  | 24.23  | 29.97   | 58.30   |
| **HEonGPU**   | **12.72** | **12.75**  | **13.60**  | **15.88**  | **23.10**   | **38.24**   |
| Speedup (vs [TFHE-rs](https://github.com/zama-ai/tfhe-rs))  | 2.48× | 2.47×  | 2.32×  | 2.02×  | 1.46×   | 1.52×   |
| Speedup (vs [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931)) | 1.46× | 1.46×  | 1.39×  | 1.53×  | 1.30×   | 1.52×   |

</div>

> **Table**: Latency (ms) comparison between [TFHE-rs](https://github.com/zama-ai/tfhe-rs), [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931), and HEonGPU for different bit sizes (on GPU), based on the STD128 parameter set.  
> All benchmarks were performed on an **NVIDIA RTX 4090 GPU**.  
> The last two rows show the speedup of HEonGPU with respect to [TFHE-rs](https://github.com/zama-ai/tfhe-rs) and [Literature](https://tches.iacr.org/index.php/TCHES/article/view/11931).

### 🚨 **New Feature: Serialization Module**

The new serializer module has been successfully integrated into HEonGPU. It provides high-performance serialization and deserialization of homomorphic encryption objects (Context, Secretkey, Publickey, Relinkey, Galoiskey, Plaintext, Ciphertext, etc.) in raw binary or optional [Zlib-compressed](https://zlib.net/) formats. This enhancement enables blazing-fast disk I/O, seamless client-server transfers, and up to a 60% reduction in storage and bandwidth.

### 🚨 **New Feature: Integration [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU)**

[RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU) has been successfully integrated into HEonGPU. [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU) features a secure Deterministic Random Bit Generator (DRBG) designed according to NIST [Recommendation for Random Number Generation Using Deterministic Random Bit Generators](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-90Ar1.pdf). This integration enhances GPU-based random number generation, ensuring both high performance and robust security.

### 🚨 **New Application: Private Information Retrieval on GPU ([PIRonGPU](https://github.com/Alisah-Ozcan/PIRonGPU))**

[PIRonGPU](https://github.com/Alisah-Ozcan/PIRonGPU) is a high-performance library that enhances secure data retrieval through Private Information Retrieval (PIR) on GPUs. By modifying the  [SealPIR](https://github.com/microsoft/SealPIR) protocol with **HEonGPU**, it achieves rapid, confidential querying, offering an efficient and scalable solution for privacy-sensitive applications.

### 🚨 **New Feature: Logic Operation and [3 More CKKS Bootstrapping Types](example/bootstrapping/README.md)**

HEonGPU now provides comprehensive support for logic operations across both the `BFV` and `CKKS` encryption schemes. In addition, the latest update introduces three new `CKKS` Bootstrapping types; two of which leverage `Bit` Bootstrapping and `Gate` Bootstrapping techniques, while the third employs `Slim` Bootstrapping, a method that is significantly more efficient than `Regular` Bootstrapping.  These enhancements not only broaden HEonGPU’s functionality but also significantly improve its performance in managing noise and enabling efficient, secure computations on GPU platforms.

The Logic Operations supported: 
- NOT, AND, NAND, OR, NOR, XOR, XNOR

3 More CKKS Bootstrapping Types:
- `Slim` Bootstrapping (supports only Real Numbers)
- `Bit` Bootstrapping (supports only Binary Numbers)
- `Gate` Bootstrapping (supports only Binary Numbers)
  - AND, NAND, OR, NOR, XOR, XNOR

<div align="center">

### Execution times of the HEonGPU Bootstrapping Operations (on RTX 4090)

| Bootstrapping Type   | N    | Slot Count| LKM | Remaining Level | Total Time | Amortized Time |
|:--------------------:|:----:|:----:|:---:|:---------------:|:----------:|:--------------:|
| **Slim Bootstrapping**    | 2^16 | 2^15 | ON  | 0 Level        | 99.12 ms   | 3.02 µs        |
|                      | 2^16 | 2^15 | ON  | 2 Level        | 114.13 ms  | 3.48 µs        |
|                      | 2^16 | 2^15 | ON  | 4 Level        | 164.20 ms  | 5.01 µs        |
| **Bit Bootstrapping**     | 2^15 | 2^14 | OFF | 0 Level        | 33.74 ms   | 2.06 µs        |
|                      | 2^15 | 2^14 | OFF | 2 Level        | 39.36 ms   | 2.40 µs        |
|                      | 2^15 | 2^14 | OFF | 4 Level        | 46.54 ms   | 2.84 µs        |
|                      | 2^15 | 2^14 | OFF | 6 Level        | 55.66 ms   | 3.40 µs        |
|                      | 2^16 | 2^15 | OFF | 0 Level        | 86.69 ms   | 2.73 µs        |
|                      | 2^16 | 2^15 | OFF | 2 Level        | 100.72 ms  | 3.07 µs        |
|                      | 2^16 | 2^15 | OFF | 4 Level        | 115.88 ms  | 3.53 µs        |
| **Gate Bootstrapping***    | 2^15 | 2^14 | OFF | 0 Level        | 27.03 ms   | 1.64 µs        |
|                      | 2^16 | 2^15 | OFF | 0 Level        | 70.73 ms   | 2.16 µs        |
</div>
LKM: Less Key Mode is a bootstrapping optimization in HEonGPU. Its purpose is to reduce the required amount of Galois keys by 30% while sacrificing 15–20% performance. This is useful in cases where GPU memory is insufficient.

*: For all gates



### 🚨 **New Feature: [CKKS Regular Bootstrapping](example/bootstrapping/README.md)**

HEonGPU now includes support for `CKKS Regular` Bootstrapping, enabling efficient evaluation of deep computational circuits with high precision and security. On an NVIDIA RTX 4090, it performs `CKKS` Regular Bootstrapping for 
N=65536 in under 170 ms.

### 🚨 **New Feature: [Multiparty Computation (MPC) Support](example/mpc/README.md)**

HEonGPU now includes support for **Multiparty Computation (MPC)** protocols, providing a secure and collaborative framework for encrypted computations. By incorporating *Multiparty Homomorphic Encryption (MHE)* capabilities, the library enables distributed computations with threshold encryption models such as `N-out-of-N`. The implementation is fully optimized for GPU environments, delivering minimal latency and maximum performance in collaborative settings.

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.26.4
- [GCC](https://gcc.gnu.org/)
- [GMP](https://gmplib.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4
- [OpenSSL](https://www.openssl.org/) >= 1.1.0
- [ZLIB](https://zlib.net/)

### Third-Party Dependencies
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
- [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT)
- [RNGonGPU](https://github.com/Alisah-Ozcan/RNGonGPU)
- [RMM](https://github.com/rapidsai/rmm)
- [Thrust](https://github.com/NVIDIA/thrust)
- [GoogleTest](https://github.com/google/googletest)

HEonGPU automatically handle third-party dependencies like GPU-NTT, GPU-FFT, RMM, Thrust, GoogleTest.

### Build & Install

To build and install HEonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

<div align="center">

| GPU Architecture | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value) |
|:----------------:|:---------------------------------------------------:|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada	 | 89, 90 |

</div>

```bash
$ cmake -S . -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/
$ sudo cmake --install build
```

## Testing & Benchmarking

To run tests:

```bash
$ cmake -S . -D HEonGPU_BUILD_TESTS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/test/<...>
$ Example: ./build/bin/test/bfv_addition_testcases
```
Or:
```bash
$ cmake -S . -D HEonGPU_BUILD_TESTS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/
$ cmake --build build --target test
```

To run benchmarks:

```bash
$ cmake -S . -D HEonGPU_BUILD_BENCHMARKS=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/benchmark/<...>
$ Example: ./build/bin/benchmark/bfv_benchmark
```

## Examples

To run examples:

```bash
$ cmake -S . -D HEonGPU_BUILD_EXAMPLES=ON -D CMAKE_CUDA_ARCHITECTURES=89 -B build
$ cmake --build ./build/

$ ./build/bin/examples/<...>
$ Example: ./build/bin/examples/1_basic_bfv
```

### Toy Example

```c++
#include "heongpu.cuh"

int main() {
    cudaSetDevice(0); // Use it for memory pool
    
    heongpu::HEContext<heongpu::Scheme::BFV> context(
            heongpu::keyswitching_type::KEYSWITCHING_METHOD_I);

    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_coeff_modulus_default_values(1);
    int plain_modulus = 1032193;
    context.set_plain_modulus(plain_modulus);
    context.generate();

    heongpu::HEKeyGenerator<heongpu::Scheme::BFV> keygen(context);
    heongpu::Secretkey<heongpu::Scheme::BFV> secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey<heongpu::Scheme::BFV> public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder<heongpu::Scheme::BFV> encoder(context);
    heongpu::HEEncryptor<heongpu::Scheme::BFV> encryptor(context, public_key);
    heongpu::HEDecryptor<heongpu::Scheme::BFV> decryptor(context, secret_key);
    heongpu::HEArithmeticOperator<heongpu::Scheme::BFV> operators(context, encoder);

    std::vector<uint64_t> message(poly_modulus_degree, 8ULL);
    heongpu::Plaintext<heongpu::Scheme::BFV> P1(context);
    encoder.encode(P1, message);

    heongpu::Ciphertext<heongpu::Scheme::BFV> C1(context);
    encryptor.encrypt(C1, P1);

    operators.add_inplace(C1, C1);

    heongpu::Plaintext<heongpu::Scheme::BFV> P2(context);
    decryptor.decrypt(P2, C1);

    std::vector<uint64_t> result;
    encoder.decode(result, P2);

    return 0;
}
```

## Configuration Header ([define.h](src/heongpu/include/kernel/defines.h))

The [define.h](src/heongpu/include/kernel/defines.h) file is an essential configuration file for HEonGPU, containing key settings that define the library's limits and capabilities, including polynomial degrees, modulus bit-lengths, and memory pool sizes. 

Features in [define.h](src/heongpu/include/kernel/defines.h):
- **Polynomial Degree:** `MAX_POLY_DEGREE` (65536) and `MIN_POLY_DEGREE` (4096) define the range for polynomial degrees used in FHE.

- **Modulus Bit-Length:** `MAX_USER_DEFINED_MOD_BIT_COUNT` (60), `MIN_USER_DEFINED_MOD_BIT_COUNT` (30), `MAX_MOD_BIT_COUNT` (61), `MIN_MOD_BIT_COUNT` (30) specify valid bit-lengths for user-defined and general modulus values.

- **Maximum BFV Auxiliary Base Count:** `MAX_BSK_SIZE` (64) sets the maximum size for the BFV auxiliary base.

- **Galois Key Capability:** `MAX_SHIFT` (8) controls the maximum rotation capability for Galois keys. __Don't forget to change it if you need more rotation steps(for default galois key generation)__.

- **Device**:
  - Initial size (`0.5`): 50% of GPU memory.
  - Max size (`0.8`): 80% of GPU memory.
- **Host**:
  - Initial size (`0.1`): 10% of CPU memory.
  - Max size (`0.2`): 20% of CPU memory.

If your system allows, you can redefine the memory pool sizes to better suit your use case. 


## Storage Management

In HEonGPU, Fully Homomorphic Encryption (FHE) operations rely on an efficient storage management system that ensures data can seamlessly transition between host (CPU) and device (GPU) memory. The Storage Manager module enables this flexibility, allowing data to be stored and accessed as needed on either the host or device.

### How Storage Management Works

The `ExecutionOptions` struct in HEonGPU determines how input data is handled during and after GPU computations. Specifically:

- `set_storage_type`: Defines where the output data should reside after the operation (`DEVICE` or `HOST`).

- `set_initial_location`: Determines whether the input data should return to its original location after the computation.

```c++
// ...
ExecutionOptions options;
options.set_storage_type(storage_type::DEVICE)
       .set_initial_location(true);

operators.add(input1, input2, output, options);
// ...
```

- Simple case:
  - If `input1` and `input2` are initially on the `HOST`, they will be copied to the `DEVICE` for the computation.
  - After the operation, since `set_initial_location(true)` is specified, both `input1` and `input2` will return to the `HOST`.
  - The `output` will remain on the `DEVICE`, as specified by `set_storage_type(storage_type::DEVICE)`.

- All case:

<div align="center">

| `set_storage_type` | `set_initial_location` | Input Locations (Before)            | Input Locations (After)             | Output Location After Computation |
|:------------------:|:----------------------:|:-----------------------------------:|:-----------------------------------:|:---------------------------------:|
| `DEVICE`           | `true`                | `input1: HOST, input2: HOST`        | `input1: HOST, input2: HOST`        | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: HOST, input2: HOST`        | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `true`                | `input1: HOST, input2: DEVICE`      | `input1: HOST, input2: DEVICE`      | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: HOST, input2: DEVICE`      | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `true`                | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `DEVICE`           | `false`               | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `DEVICE`                             |
| `HOST`             | `true`                | `input1: HOST, input2: HOST`        | `input1: HOST, input2: HOST`        | `HOST`                               |
| `HOST`             | `false`               | `input1: HOST, input2: HOST`        | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `true`                | `input1: HOST, input2: DEVICE`      | `input1: HOST, input2: HOST`        | `HOST`                               |
| `HOST`             | `false`               | `input1: HOST, input2: DEVICE`      | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `true`                | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |
| `HOST`             | `false`               | `input1: DEVICE, input2: DEVICE`    | `input1: DEVICE, input2: DEVICE`    | `HOST`                               |

</div>

## Using HEonGPU in a downstream CMake project

Make sure HEonGPU is installed before integrating it into your project. The installed HEonGPU library provides a set of config files that make it easy to integrate HEonGPU into your own CMake project. In your CMakeLists.txt, simply add:

```cmake
project(<your-project> LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
# ...
find_package(HEonGPU)
# ...
target_link_libraries(<your-target> (PRIVATE|PUBLIC|INTERFACE) HEonGPU::heongpu CUDA::cudart)
# ...
set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# ...
```

## How to Cite HEonGPU

Please use the below BibTeX, to cite HEonGPU in academic papers.

```
@misc{cryptoeprint:2024/1543,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1543}
}
```

Please use the below BibTeX, to cite key-switching optimizations in academic papers.

```
@misc{cryptoeprint:2025/124,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{GPU} Implementations of Three Different Key-Switching Methods for Homomorphic Encryption Schemes},
      howpublished = {Cryptology {ePrint} Archive, Paper 2025/124},
      year = {2025},
      url = {https://eprint.iacr.org/2025/124}
}
```

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)