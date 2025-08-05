.. _introduction:

Introduction to HEonGPU
=======================

HEonGPU is a high-performance, open-source C++ library designed to optimize and accelerate Fully Homomorphic Encryption (FHE) operations on Graphics Processing Units (GPUs). FHE is a cryptographic method that enables meaningful arithmetic and logic calculations to be conducted directly on encrypted data without relying on the secret key. While this capability is transformative for privacy-preserving applications in untrusted environments like cloud computing, its practical adoption has been hindered by significant computational overhead.

HEonGPU directly addresses this performance bottleneck by leveraging the massive parallel processing capacity of modern GPUs to execute complex operations concurrently. The library's architecture is built around a multi-stream model that not only allows for parallel processing of tasks to improve throughput but also minimizes the overhead of data transfers between the host (CPU) and the device (GPU)—a common bottleneck in hybrid computation.

Project Goals and Philosophy
----------------------------

The HEonGPU project is guided by several core objectives:

* **High-Performance Framework**: To provide an optimized environment for executing the ``BFV``, ``CKKS``, and ``TFHE`` homomorphic encryption schemes by harnessing the full power of the CUDA platform.
* **User-Friendly Interface**: To offer a high-level C++ API that requires no prior knowledge of GPU programming. All complex CUDA kernels are encapsulated within easy-to-use, object-oriented classes. The interface is intentionally designed to be straightforward, drawing inspiration from the user-friendly design of Microsoft SEAL.
* **Optimized Architecture**: To ensure efficient memory management and concurrent execution of encrypted computations on the GPU through a fine-tuned, multi-stream design and the use of the RAPIDS Memory Manager (RMM) for an efficient memory pool.

Key Features and Capabilities
-----------------------------

HEonGPU is distinguished by a comprehensive set of features designed for performance, security, and usability.

.. list-table:: HEonGPU Capabilities at a Glance
   :widths: 30 15
   :header-rows: 1

   * - Capability / Scheme
     - Status
   * - BFV Scheme
     - ✓
   * - CKKS Scheme
     - ✓
   * - TFHE Scheme
     - ✓
   * - BGV Scheme
     - Planned
   * - CKKS Regular Bootstrapping
     - ✓
   * - CKKS Slim Bootstrapping
     - ✓
   * - CKKS Bit Bootstrapping
     - ✓
   * - CKKS Gate Bootstrapping
     - ✓
   * - TFHE Gate Bootstrapping
     - ✓
   * - Multi-Party Computation (MPC)
     - ✓
   * - Collective Bootstrapping (MPC)
     - ✓

**Full GPU Execution**
    A core design principle is that all FHE operations—from basic arithmetic to the most complex procedures like bootstrapping and key-switching—are executed entirely on the GPU. This architectural decision is critical for performance, as it eliminates the latency associated with frequent data transfers between the host and device.

**Advanced Bootstrapping Support**
    Bootstrapping, the process of refreshing a ciphertext to reduce its noise, is a cornerstone feature that enables computations of arbitrary depth. HEonGPU provides extensive, highly optimized bootstrapping support:

    * **CKKS**: 

        Four distinct variants are implemented: **Regular** (for complex numbers), **Slim** (an efficient version for real numbers), **Bit** (for binary data), and **Gate** (which embeds a logic gate evaluation within the bootstrap).
    
    * **TFHE**:

        **Gate bootstrapping** is supported, enabling the efficient evaluation of boolean circuits.

**Multi-Party Computation (MPC)**
    The library includes support for secure multi-party computation protocols. This allows multiple parties to collaboratively compute on encrypted data using threshold encryption models (e.g., N-out-of-N). This capability is enhanced by **Collective (Distributed) Bootstrapping**, which allows deep multi-party workloads to run entirely on the GPU without pausing to reset noise.

**Broad Scheme Support**
    HEonGPU provides high-performance implementations for several of the most widely used FHE schemes:

    * **BFV**: For performing exact computations on encrypted integers, using RNS variants for efficiency.
    * **CKKS**: For performing approximate arithmetic on encrypted real or complex numbers, making it particularly suitable for privacy-preserving machine learning.
    * **TFHE**: For high-speed boolean circuit evaluation and programmable bootstrapping. It includes native support for homomorphic unsigned integer types from ``huint8`` to ``huint256``.

**Logic and Arithmetic Operations**
    The library provides comprehensive support for standard arithmetic as well as bit-wise logic operations (``NOT``, ``AND``, ``NAND``, ``OR``, ``NOR``, ``XOR``, ``XNOR``) across both the BFV and CKKS schemes.

**High-Performance Serialization**
    A dedicated serialization module provides fast serialization and deserialization of all core homomorphic encryption objects (Context, keys, plaintexts, ciphertexts). It supports both raw binary and an optional `Zlib-compressed <https://zlib.net/>`_ format, which can reduce storage and bandwidth requirements by up to 60%.

The HEonGPU Ecosystem
---------------------

HEonGPU is not a standalone project but the cornerstone of a broader suite of GPU-accelerated cryptographic tools developed by the same author. This ecosystem approach demonstrates the library's maturity and interface stability. Related projects include:

* **PIRonGPU**: A high-performance library for Private Information Retrieval built on top of HEonGPU.
* **GPU-NTT** and **GPU-FFT**: Specialized libraries for the core polynomial arithmetic primitives (Number Theoretic Transform and Fast Fourier Transform) that power HEonGPU.
* **RNGonGPU**: A secure Deterministic Random Bit Generator (DRBG) designed according to NIST SP 800-90A recommendations. It is fully integrated into HEonGPU to provide high-performance, cryptographically secure random number generation directly on the GPU, replacing the non-secure default CURAND generators for security-critical applications.
