.. _examples:

Library Examples
================

The HEonGPU repository contains a rich set of sample programs designed to help users understand and experiment with the library's functionalities. This guide provides a detailed walkthrough of each example, explaining its purpose and the key concepts demonstrated.

Basic Operations
----------------

These examples cover the fundamental workflows for the BFV, CKKS, and TFHE schemes.

1. Basic BFV Arithmetic (``1_basic_bfv.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example demonstrates the core workflow for performing exact integer arithmetic with the BFV scheme.

    * **Workflow**: It shows how to set up the ``HEContext`` for BFV, generate keys, encode a vector of integers into a plaintext, encrypt it, perform homomorphic multiplication and relinearization, and finally decrypt and decode the result.

    * **Key Concept**: **Noise Budget Management**. The example explicitly prints the noise budget remaining in the ciphertext before and after operations using ``decryptor.remainder_noise_budget()``.

2. Basic CKKS Arithmetic (``2_basic_ckks.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example introduces the CKKS scheme for approximate arithmetic on real numbers.

    * **Workflow**: It covers setting up the CKKS context, defining a ``scale`` for precision, encoding a vector of doubles, encrypting, performing homomorphic multiplication, and then applying the essential ``relinearize`` and ``rescale`` operations.
    * **Key Concept**: **Scaling and Rescaling**. It demonstrates the critical need to call ``operators.rescale_inplace()`` after a multiplication to manage the scale of the ciphertext and control the precision of the approximate numbers.

3. BFV Logic Operations (``8_basic_bfv_logic.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example shows how to perform bitwise logic operations on encrypted binary data (0s and 1s) using the BFV scheme.

    * **Workflow**: It encrypts vectors of binary values and then uses the ``HELogicOperator`` class to perform homomorphic ``AND`` and ``XNOR`` operations.
    * **Key Concept**: The ``HELogicOperator`` provides a dedicated interface for boolean computations, which are fundamental in many privacy-preserving applications.

4. CKKS Logic Operations (``9_basic_ckks_logic.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Similar to the BFV logic example, this demonstrates performing logic gates on binary data that has been encoded within the CKKS scheme.

    * **Workflow**: It encrypts vectors of 0.0s and 1.0s and uses the ``HELogicOperator`` to perform ``AND`` and ``XNOR`` operations.
    * **Key Concept**: This showcases the versatility of the CKKS scheme, which can handle both approximate arithmetic and boolean logic.

5. Basic TFHE Operations (``12_basic_tfhe.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   This example provides a complete demonstration of the TFHE scheme for boolean circuit evaluation.

    * **Workflow**: It initializes a TFHE context, generates a secret key and a bootstrapping key, encrypts several boolean vectors, and then evaluates a series of logic gates (``NAND``, ``AND``, ``OR``, ``XOR``, ``NOT``, ``MUX``) on the ciphertexts.

    * **Key Concept**: **Gate Bootstrapping**. Every binary gate operation (except for NOT) in TFHE involves a bootstrapping procedure to refresh the ciphertext, allowing for the evaluation of arbitrarily complex boolean circuits. This is demonstrated by passing the ``boot_key`` to each logic function.

Advanced Features
-----------------

These examples demonstrate more advanced capabilities of the library, such as different key-switching methods and parallel execution with CUDA streams.

6. Key-Switching Methods in BFV (``3_switchkey_methods_bfv.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example explores the various key-switching operations available in the BFV scheme.

    * **Workflow**: After a standard multiplication and relinearization, it demonstrates how to perform ``rotate_rows`` and ``rotate_columns`` using Galois keys. It also shows how to perform a ``keyswitch`` operation to change the secret key under which a ciphertext is encrypted.
    * **Key Concept**: **Galois Keys and Rotation**. It illustrates how to generate ``Galoiskey`` objects and use them to perform SIMD rotations on the encrypted vectors.

7. Key-Switching Methods in CKKS (``4_switchkey_methods_ckks.cu``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This example is the CKKS counterpart to the previous one, showcasing rotations and key-switching for approximate numbers.

    * **Workflow**: It demonstrates the use of ``rotate_rows_inplace`` with Galois keys and the ``keyswitch`` operation, highlighting that the core concepts are similar across both BFV and CKKS.
    * **Key Concept**: **Key-Switching Variants**. The example is configured to use ``KEYSWITCHING_METHOD_II``, a more advanced hybrid method that can offer better performance than the default ``METHOD_I`` in certain scenarios.

8. Multi-Stream Parallel Execution (``5_``, ``6_``, ``7_`` examples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    This set of examples demonstrates how to leverage CUDA streams to execute multiple independent homomorphic operations concurrently, maximizing GPU utilization.

    * ``5_default_stream_usage.cu``: Serves as a baseline, performing a sequence of 64 operations serially in the default CUDA stream.
    * ``6_multi_stream_usage_way1.cu``: Uses **OpenMP** to parallelize a loop across multiple CPU threads. Each thread is assigned its own CUDA stream, and operations within that thread are submitted to its dedicated stream.
    * ``7_multi_stream_usage_way2.cu``: Shows an alternative approach where a single CPU thread manages multiple CUDA streams, assigning tasks to them in a round-robin fashion.
    * **Key Concept**: By passing a ``cudaStream_t`` object via ``ExecutionOptions``, users can direct HEonGPU to execute an operation on a specific stream, enabling powerful parallel execution patterns.

Serialization
-------------

9. Serialization in BFV and CKKS (``10_``, ``11_`` examples)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    These examples demonstrate how to serialize and deserialize all major HEonGPU objects, which is essential for saving state, persistence, and client-server communication.

    * ``10_bfv_serialization.cu`` and ``11_ckks_serialization.cu`` showcase the same workflow for their respective schemes.
    * **Workflow**: They demonstrate serializing and deserializing every object type: ``HEContext``, ``Secretkey``, ``Publickey``, ``Relinkey``, ``Galoiskey``, ``Plaintext``, and ``Ciphertext``.
    * **Key Concept**: The library provides two main ways to serialize:
        * Directly calling ``object.save(stream)`` and ``object.load(stream)`` for raw, uncompressed binary data.
        * Using the convenient ``heongpu::serializer`` helpers (e.g., ``save_to_file``, ``load_from_file``), which automatically apply Zlib compression to reduce object size by 50-60%.

Bootstrapping Examples
----------------------

These examples, located in the `example/bootstrapping` directory, demonstrate the use of the library's powerful bootstrapping capabilities.

* ``1_ckks_regular_bootstrapping.cu``: Shows how to refresh a CKKS ciphertext containing complex numbers.
* ``2_ckks_slim_bootstrapping.cu``: Demonstrates the more efficient slim bootstrapping variant for real numbers.
* ``3_ckks_bit_bootstrapping.cu``: Illustrates the specialized bootstrapping for binary data.
* ``4_ckks_gate_bootstrapping.cu``: Shows how to embed a logic gate within the bootstrapping process for maximum efficiency.

Multi-Party Computation Examples
--------------------------------

These examples, located in the `example/mpc` directory, cover the library's features for secure multi-party computation.

* ``1_multiparty_computation_bfv.cu`` and ``2_multiparty_computation_ckks.cu``: Demonstrate the full protocol for N-out-of-N threshold FHE. This includes each party generating key shares, a server aggregating them into a single public key, encryption with the collective key, and finally, each party generating a partial decryption which are then combined to get the final result.
* ``3_mpc_collective_bootstrapping_bfv.cu`` and ``4_mpc_collective_bootstrapping_ckks.cu``: Showcase the advanced collective bootstrapping protocol, where multiple parties can jointly refresh a ciphertext without any single party having access to the full secret key.
