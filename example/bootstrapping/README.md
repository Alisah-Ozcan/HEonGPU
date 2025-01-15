# Bootstrapping in HEonGPU

Homomorphic encryption allows computations to be performed on encrypted data without decryption. However, as computations progress, the noise in the ciphertext grows, eventually making further operations impossible. Bootstrapping is the process of "refreshing" a ciphertext to reduce this noise, enabling deeper computations and extending the lifespan of the ciphertext.

Currently, HEonGPU supports bootstrapping for the `CKKS` scheme. In the near future, support for the `TFHE` scheme will be added, including `TFHE` bootstrapping to enable efficient bit-level encrypted computations.

## CKKS Bootstrapping

The `CKKS` bootstrapping implementation in HEonGPU demonstrates the initialization and execution of the bootstrapping process entirely on the GPU.

The [1_ckks_bootstrapping.cu](1_ckks_bootstrapping.cu) provides a basic example of `CKKS` bootstrapping with a polynomial degree of 4096. While functional, these parameters are insecure for practical use, as `CKKS` bootstrapping typically requires a modulus of 65536 for sufficient noise budget and security. This example is intended for demonstration purposes only and should not be used in a production environment.

### Implementation Details

`CKKS` bootstrapping primarily consists of four main stages: `Mod Raise`, `Coeff to Slot`, `Approximate Modular Reduction`, and `Slot to Coeff`. Among these, `Coeff to Slot` is the most time-consuming part of the bootstrapping. `Coeff to Slot` and `Slot to Coeff` stages primarily involve matrix multiplication, which can be accelerated using the BSGS method. However, HEonGPU employs the most optimized approach described in the [Faster Homomorphic Discrete Fourier Transforms and Improved FHE Bootstrapping](https://eprint.iacr.org/2018/1073.pdf) paper, significantly minimizing both the number of multiplications and rotations. Moreover, HEonGPU does not use sparse packing as described in the [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153.pdf) paper; instead, it utilizes all slots of CKKS.

Homomorphic operation number and depth consume for homomorphic DFT(from [paper](https://eprint.iacr.org/2018/1073.pdf)):
|                        | Naive       | BSGS        | Ours with radix r   | Hybrid with radix r   |
|------------------------|-------------|-------------|---------------------|-----------------------|
| # Hadamard Mult        | \(O(n)\)    | \(O(n)\)    | \(O(r \log_r n)\)   | \(O(r \log_r n)\)     |
| # Slot Shifting        | \(O(n)\)    | \(O(\sqrt{n})\) | \(O(r \log_r n)\) | \(O(\sqrt{r} \log_r n)\) |
| Depth                  | 1           | 1           | \(O(\log_r n)\)     | \(O(\log_r n)\)       |

Although this method results in higher depth consumption in stages `Coeff to Slot` and `Slot to Coeff` compared to Naive and BSGS approaches, it provides a significant performance improvement. The HEonGPU `BootstrappingConfig` class allows selecting the level loss for operations in stages `Coeff to Slot` and `Slot to Coeff` (currently limited to 2, 3, 4, or 5).

Additionally, the `BootstrappingConfig` class allows specifying the number of galois keys to be used for rotations in stages `Coeff to Slot` and `Slot to Coeff`. If the `less_key_mode` mode is false, a unique galois key is generated for each rotation index, leading to significant memory usage. When `less_key_mode` mode is true, this memory overhead can be reduced at the cost of some performance. To further reduce this, `KEYSWITCHING_METHOD_II` is utilized. There is also an option to store all galois keys on the CPU, but this would significantly slow down the bootstrapping process. However, the key size still considerably large, so it is important to keep this in mind while perfoming bootstrapping!

Currently, taylor approximation is used for `Approximate Modular Reduction`; however, chebyshev implementation will be added in the near future. Additionally, the `BootstrappingConfig` class allows configuring the precision of the taylor approximation.

The vandermonde matrix is automatically generated on GPU based on the `BootstrappingConfig` class and stored on the GPU in a ready-to-use and encoded format for processing.

> **⚠ Warning:** Bootstrapping was tested on a system with 128 GB RAM and an NVIDIA RTX 4090(24 GB). If you are working with a system with lower specifications, please refer to [define.h](../../src/heongpu/include/kernel/defines.h) and adjust the memory pool settings according to your system.


### The papers used while implementing HEonGPU CKKS bootstrapping

- [Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/153.pdf)
- [Improved Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2018/1043.pdf)
- [Faster Homomorphic Discrete Fourier Transforms and Improved FHE Bootstrapping](https://eprint.iacr.org/2018/1073.pdf)
- [Better Bootstrapping for Approximate Homomorphic Encryption](https://eprint.iacr.org/2019/688.pdf)
- [Efficient Bootstrapping for Approximate Homomorphic Encryption with Non-Sparse Keys](https://eprint.iacr.org/2020/1203.pdf)
- [Over 100x Faster Bootstrapping in Fully Homomorphic Encryption through Memory-centric Optimization with GPUs](https://eprint.iacr.org/2021/508.pdf) (added soon)
- [BLEACH: Cleaning Errors in Discrete Computations over CKKS](https://eprint.iacr.org/2022/1298.pdf) (added soon)
- [Bootstrapping Bits with CKKS](https://eprint.iacr.org/2024/767.pdf) (added soon)

## How to Cite HEonGPU CKKS Bootstrapping

Please keep using the below BibTeX to cite HEonGPU in academic papers. We are currently working on a new paper specifically for GPU implementation of CKKS Bootstrapping, and until its release, please continue to use this citation.

```
@misc{cryptoeprint:2024/1543,
      author = {Ali Şah Özcan and Erkay Savaş},
      title = {{HEonGPU}: a {GPU}-based Fully Homomorphic Encryption Library 1.0},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1543},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1543}
}
```

## Acknowledgments

I would like to appreciate `Efe İzbudak` for his mathematical support and the great teamwork during `CKKS` bootstrapping implementation.

#

If you have further questions or require additional information, feel free to reach out at:
- Email: alisah@sabanciuniv.edu
- LinkedIn: [Profile](https://www.linkedin.com/in/ali%C5%9Fah-%C3%B6zcan-472382305/)