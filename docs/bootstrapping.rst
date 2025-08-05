.. _bootstrapping:

Bootstrapping in HEonGPU
========================

Homomorphic encryption allows computations to be performed on encrypted data. However, as these computations progress, the noise inherent in the ciphertext grows, eventually corrupting the underlying message and making further operations impossible. **Bootstrapping** is the process of "refreshing" a ciphertext to reduce this noise, effectively enabling an unlimited number of computations and making the scheme fully homomorphic.

HEonGPU provides extensive, highly optimized support for bootstrapping, with all operations executed entirely on the GPU. Currently, the library supports four different bootstrapping types for the CKKS scheme and gate bootstrapping for the TFHE scheme.

CKKS Bootstrapping
------------------

The CKKS bootstrapping implementation in HEonGPU is a complex procedure consisting of four main stages: `Mod Raise`, `Coeff to Slot`, `Approximate Modular Reduction`, and `Slot to Coeff`. The most time-consuming part of this process is the `Coeff to Slot` stage, which involves homomorphic Discrete Fourier Transforms (DFTs).

Implementation and Optimization Details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The implementation is heavily optimized for GPU execution and is based on techniques described in several key academic papers.

* **Efficient DFT**: Instead of naive or BSGS methods for the homomorphic DFT, HEonGPU employs a more optimized approach described in the paper "Faster Homomorphic Discrete Fourier Transforms and Improved FHE Bootstrapping" (`ePrint 2018/1073 <https://eprint.iacr.org/2018/1073.pdf>`_). While this method consumes more circuit depth, it significantly minimizes the number of multiplications and rotations, leading to a substantial performance improvement.
* **Full Slot Utilization**: The library utilizes all available slots in the CKKS ciphertext, avoiding the sparse packing method described in some earlier works.
* **Configurable Precision**: The `Approximate Modular Reduction` stage currently uses a Taylor approximation. The `BootstrappingConfig` class allows the user to configure the precision of this approximation.
* **Memory Management**: Bootstrapping is extremely memory-intensive, particularly due to the large size of the Galois keys required for rotations. The `BootstrappingConfig` class provides a ``less_key_mode``. When enabled, this mode reduces the number of required Galois keys by 30% at the cost of a 15-20% performance decrease, making it a valuable option for systems with limited GPU memory.

CKKS Bootstrapping Variants
^^^^^^^^^^^^^^^^^^^^^^^^^^^

HEonGPU supports four different types of bootstrapping for the CKKS scheme, each tailored to a specific use case.

1. Regular Bootstrapping
""""""""""""""""""""""""

This is the standard bootstrapping procedure for CKKS, applied when the encrypted message contains **complex numbers**.

.. mermaid::

    graph LR
      A["m(x)"] -->|ModRaise| B["m(x) + qI(x)"]
      B["m(x) + qI(x)"] -->|CoeffToSlot| C["Encode(<b>m</b> + q<b>I</b>)"]
      C["Encode(<b>m</b> + q<b>I</b>)"] -->|EvalMod| D["Encode(<b>m</b>)"]
      D["Encode(<b>m</b>)"] -->|SlotToCoeff| E["m(x)"]

2. Slim Bootstrapping
"""""""""""""""""""""

This is a more efficient variant designed for messages in the **real number domain**. The process begins with `SlotToCoeff` rather than `ModRaise`, ensuring that the modular reduction is applied exclusively to the real part of the message.

.. mermaid::

    graph LR
      A["Encode(<b>z</b>)"] -->|SlotToCoeff| B["z(x)"]
      B["z(x)"] -->|ModRaise| C["z(x) + qI(x)"]
      C["z(x) + qI(x)"] -->|CoeffToSlot| D["Encode(<b>z</b> + q<b>I</b>)"]
      D["Encode(<b>z</b> + q<b>I</b>)"] -->|EvalMod| E["Encode(<b>z</b>)"]

3. Bit Bootstrapping
""""""""""""""""""""

This variant is highly optimized for messages in the **binary domain**. It replaces the standard `EvalMod` function with a more efficient `EvalBinboot` function, which requires a lower multiplication depth. For this method to function correctly, the last modulus in the coefficient chain (:math:`q_L`) must be exactly twice the value of the CKKS scaling factor (:math:`\Delta`).

4. Gate Bootstrapping
"""""""""""""""""""""

This is the most specialized variant, also designed for **binary messages**. It is similar to Bit Bootstrapping but uses an `EvalGateboot` function. This function applies a logic gate (e.g., AND, OR, XOR) *directly during the bootstrapping process*, similar to how gate bootstrapping is performed in TFHE. This is extremely efficient as it does not require an extra level to continue operations after bootstrapping. For this method to function correctly, the last modulus (:math:`q_L`) must be exactly three times the value of the scaling factor (:math:`\Delta`).

TFHE Bootstrapping
------------------

In addition to CKKS, HEonGPU supports **Gate Bootstrapping** for the TFHE scheme. This enables the efficient evaluation of boolean circuits on encrypted data and is the core operation that makes TFHE a powerful tool for bit-level encrypted computations.

.. warning::
    Bootstrapping was tested on a system with 128 GB of RAM and an NVIDIA RTX 4090 with 24 GB of VRAM. If you are working with a system with lower specifications, please refer to the configuration header (`define.h`) and adjust the memory pool settings according to your system's capabilities.
