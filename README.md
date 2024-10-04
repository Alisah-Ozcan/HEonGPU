# 🚀 **HEonGPU** - A GPU Based Homomorphic Encryption Library

HEonGPU is a high-performance library designed to optimize Fully Homomorphic Encryption (FHE) operations on GPUs. By leveraging the parallel processing power of GPUs, it significantly reduces the computational load of FHE through concurrent execution of complex operations. Its multi-stream architecture enables efficient parallel processing and minimizes the overhead of data transfers between the CPU and GPU. These features make HEonGPU ideal for large-scale encrypted computations, offering reduced latency and improved performance.

The goal of HEonGPU is to provide:
- A high-performance framework for executing FHE schemes, specifically `BFV` and `CKKS`, by leveraging the parallel processing capabilities of CUDA.
- A user-friendly C++ interface that requires no prior knowledge of GPU programming, with all CUDA kernels encapsulated in easy-to-use classes.
- An optimized multi-stream architecture that ensures efficient memory management and concurrent execution of encrypted computations on the GPU.

For more information about HEonGPU: https://eprint.iacr.org/2024/1543

## Installation

### Requirements

- [CMake](https://cmake.org/download/) >=3.26.4
- [GCC](https://gcc.gnu.org/)
- [GMP](https://gmplib.org/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) >=11.4

### Third-Party Dependencies
- [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)
- [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT)
- [RMM](https://github.com/rapidsai/rmm)
- [Thrust](https://github.com/NVIDIA/thrust)
- [GoogleTest](https://github.com/google/googletest)

HEonGPU automatically handle third-party dependencies like GPU-NTT, GPU-FFT, RMM, Thrust, GoogleTest.

### Build & Install

To build and install HEonGPU, follow the steps below. This includes configuring the project using CMake, compiling the source code, and installing the library on your system.

| GPU Architecture        | Compute Capability (CMAKE_CUDA_ARCHITECTURES Value)                        |
|----------------|---------------------------------|
| Volta  | 70, 72 |
| Turing | 75 |
| Ampere | 80, 86 |
| Ada	 | 89, 90 |

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
    heongpu::Parameters context(heongpu::scheme_type::bfv,
            heongpu::keyswitching_type::KEYSWITHING_METHOD_I);

    size_t poly_modulus_degree = 8192;
    context.set_poly_modulus_degree(poly_modulus_degree);
    context.set_default_coeff_modulus(1);
    int plain_modulus = 1032193;
    context.set_plain_modulus(plain_modulus);
    context.generate();

    heongpu::HEKeyGenerator keygen(context);
    heongpu::Secretkey secret_key(context);
    keygen.generate_secret_key(secret_key);

    heongpu::Publickey public_key(context);
    keygen.generate_public_key(public_key, secret_key);

    heongpu::HEEncoder encoder(context);
    heongpu::HEEncryptor encryptor(context, public_key);
    heongpu::HEDecryptor decryptor(context, secret_key);
    heongpu::HEOperator operators(context);

    std::vector<uint64_t> message(poly_modulus_degree, 8ULL);
    heongpu::Plaintext P1(context);
    encoder.encode(P1, message);

    heongpu::Ciphertext C1(context);
    encryptor.encrypt(C1, P1);

    operators.add_inplace(C1, C1);

    heongpu::Plaintext P2(context);
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
add_compile_definitions(BARRETT_64)
add_compile_definitions(FLOAT_64)
target_compile_definitions(<your-target> PRIVATE BARRETT_64)
target_compile_definitions(<your-target> PRIVATE FLOAT_64)
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

## License
This project is licensed under the [Apache License](LICENSE). For more details, please refer to the License file.

## Contact
If you have any questions or feedback, feel free to contact me: 
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)