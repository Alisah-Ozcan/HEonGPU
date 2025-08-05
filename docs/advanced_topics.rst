.. _advanced_topics:

Advanced Topics
===============

This section covers advanced usage patterns, configuration options, and integration strategies for developers looking to leverage the full capabilities of the HEonGPU library.

Configuration (``define.h``)
--------------------------

The ``src/heongpu/include/kernel/defines.h`` file is an essential configuration header that contains key settings defining the library's limits and default behaviors. Advanced users can modify these values to tune the library for specific hardware or demanding use cases.

Key settings include:

* **Polynomial Degree**: ``MAX_POLY_DEGREE`` (default: 65536) and ``MIN_POLY_DEGREE`` (default: 4096) define the supported range for polynomial degrees.
* **Modulus Bit-Length**: These constants specify the valid bit-lengths for user-defined and general modulus values, ensuring cryptographic security.
* **Galois Key Capability**: ``MAX_SHIFT`` (default: 8) controls the maximum rotation capability for default Galois key generation. If your application requires more rotation steps, this value must be increased.
* **Memory Pool Sizes**: The initial and maximum sizes for the device (GPU) and host (CPU) memory pools are defined as percentages of available system memory. By default, the GPU pool is initialized to 50% of VRAM and can grow to 80%, while the pinned host memory pool is initialized to 10% of RAM and can grow to 20%. These values can be adjusted for systems with different memory capacities or for applications with particularly large memory footprints.

Multiparty Computation (MPC)
----------------------------

HEonGPU includes support for secure **Multi-Party Computation (MPC)** protocols, enabling multiple parties to collaboratively compute on encrypted data. This is achieved through Multiparty Homomorphic Encryption (MHE) capabilities, providing a framework for threshold encryption models such as `N-out-of-N`.

A key feature of the MPC implementation is **Collective (Distributed) Bootstrapping** for both the BFV and CKKS schemes. This powerful technique, based on designs by Mouchet et al. and Balle et al., allows a group of parties to jointly refresh a ciphertext, resetting its noise level without a trusted dealer. The implementation merges the share creation and re-encryption steps into a single, efficient CUDA kernel launch. This allows deep multi-party workloads to continue running entirely on the GPU without pausing to reset noise, which is critical for complex, collaborative privacy-preserving applications.

Using HEonGPU in a Downstream CMake Project
-------------------------------------------

To integrate HEonGPU into your own CMake project, first ensure the library is installed on your system. The installed library provides the necessary config files to make integration seamless. In your project's ``CMakeLists.txt``, you can use ``find_package`` to locate and link against the library.

.. code-block:: cmake

    # Set the project language to include CUDA
    project(<your-project> LANGUAGES CXX CUDA)

    # Find the CUDA Toolkit, which is a dependency
    find_package(CUDAToolkit REQUIRED)
    
    # ... your other project configurations ...

    # Find the HEonGPU package
    find_package(HEonGPU REQUIRED)

    # ... define your executable target ...
    add_executable(<your-target> main.cpp)

    # Link your application against the HEonGPU library and the CUDA runtime
    target_link_libraries(<your-target> PRIVATE HEonGPU::heongpu CUDA::cudart)

    # Enable separable compilation for CUDA, which is often required
    set_target_properties(<your-target> PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

Project Roadmap
---------------

HEonGPU is an actively developing project with a clear vision for the future. The roadmap indicates a strategic expansion from accelerating core cryptographic primitives to enabling complex, end-to-end privacy-preserving systems.

Planned enhancements include:

* **Support for Additional Schemes**: The library plans to add support for the **BGV** scheme to expand its cryptographic capabilities.
* **Python Wrapper**: To make the library more accessible to the data science and machine learning communities, a Python wrapper is on the roadmap.
* **Multi-GPU Support**: The architecture is planned to be extended to support multi-GPU configurations, including a multi-GPU memory pool structure, to facilitate the execution of even larger-scale applications.
