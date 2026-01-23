# HEonGPU Basic Examples

The basic example directory in the HEonGPU repository contains sample programs that demonstrate how to utilize various features of the HEonGPU library. These examples are designed to help users understand and experiment with the library's functionalities, from basic encryption and decryption to more advanced homomorphic operations.

## Overview

- [1_basic_bfv.cu](1_basic_bfv.cu): Demonstrates basic arithmetic operations on encrypted integers using the BFV scheme, including encryption, evaluation, and decryption.

- [2_basic_ckks.cu](2_basic_ckks.cu): Introduces the CKKS scheme for approximate arithmetic on real and complex numbers, showing encoding, encryption, evaluation, and decryption.

- [3_basic_memorypool_config.cu](3_basic_memorypool_config.cu): Shows how to configure the memory pool (host/device size via percentages or bytes) before generating a context.

- [4_switchkey_methods_bfv.cu](4_switchkey_methods_bfv.cu): Explores different key switching methods in the BFV scheme, illustrating how to manage and switch between keys during encrypted computations.

- [5_switchkey_methods_ckks.cu](5_switchkey_methods_ckks.cu): Examines key switching methods within the CKKS scheme, demonstrating the process of changing keys in homomorphic operations.

- [6_default_stream_usage.cu](6_default_stream_usage.cu): Shows how to perform encrypted computations using the default CUDA stream, highlighting the integration of GPU processing in homomorphic encryption.

- [7_multi_stream_usage_way1.cu](7_multi_stream_usage_way1.cu): Uses OpenMP to assign a CUDA stream to each CPU thread, demonstrating a multi-stream approach for executing parallel encrypted operations efficiently.

- [8_multi_stream_usage_way2.cu](8_multi_stream_usage_way2.cu): Implements a basic encrypted operation using multiple CUDA streams, where each stream is invoked by a single CPU thread. This example highlights an alternative method to leverage multi-stream functionality for concurrent GPU processing.

- [9_basic_bfv_logic.cu](9_basic_bfv_logic.cu): Demonstrates basic logic operations on encrypted binarys using the BFV scheme, including encryption, evaluation, and decryption.

- [10_basic_ckks_logic.cu](10_basic_ckks_logic.cu): Demonstrates basic logic operations on encrypted binarys using the CKKS scheme, including encryption, evaluation, and decryption.

- [11_bfv_serialization.cu](11_bfv_serialization.cu): Demonstrates serialization and deserialization workflows for BFV context and keys.

- [12_ckks_serialization.cu](12_ckks_serialization.cu): Demonstrates serialization and deserialization workflows for CKKS context and keys.

- [13_basic_tfhe.cu](13_basic_tfhe.cu): Introduces the TFHE scheme basics and shows simple Boolean gate operations.
