# HEonGPU Basic Examples

The basic example directory in the HEonGPU repository contains sample programs that demonstrate how to utilize various features of the HEonGPU library. These examples are designed to help users understand and experiment with the library's functionalities, from basic encryption and decryption to more advanced homomorphic operations.

## Overview

- [1_basic_bfv.cu](1_basic_bfv.cu): Demonstrates basic arithmetic operations on encrypted integers using the BFV scheme, including encryption, evaluation, and decryption.

- [2_basic_ckks.cu](2_basic_ckks.cu): Introduces the CKKS scheme for approximate arithmetic on real and complex numbers, showing encoding, encryption, evaluation, and decryption.

- [3_switchkey_methods_bfv.cu](3_switchkey_methods_bfv.cu): Explores different key switching methods in the BFV scheme, illustrating how to manage and switch between keys during encrypted computations.

- [4_switchkey_methods_ckks.cu](4_switchkey_methods_ckks.cu): Examines key switching methods within the CKKS scheme, demonstrating the process of changing keys in homomorphic operations.

- [5_default_stream_usage.cu](5_default_stream_usage.cu): Shows how to perform encrypted computations using the default CUDA stream, highlighting the integration of GPU processing in homomorphic encryption.

- [6_multi_stream_usage_way1.cu](6_multi_stream_usage_way1.cu): Uses OpenMP to assign a CUDA stream to each CPU thread, demonstrating a multi-stream approach for executing parallel encrypted operations efficiently.

- [7_multi_stream_usage_way2.cu](7_multi_stream_usage_way2.cu): Implements a basic encrypted operation using multiple CUDA streams, where each stream is invoked by a single CPU thread. This example highlights an alternative method to leverage multi-stream functionality for concurrent GPU processing.

- [8_basic_bfv_logic.cu](8_basic_bfv_logic.cu): Demonstrates basic logic operations on encrypted binarys using the BFV scheme, including encryption, evaluation, and decryption.

- [9_basic_ckks_logic.cu](9_basic_ckks_logic.cu): Demonstrates basic logic operations on encrypted binarys using the CKKS scheme, including encryption, evaluation, and decryption.