# Copyright 2024-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples/basic)

set(EXECUTABLES
    1_basic_bfv 1_basic_bfv.cu
    2_basic_ckks 2_basic_ckks.cu
    3_basic_memorypool_config 3_basic_memorypool_config.cu
    4_switchkey_methods_bfv 4_switchkey_methods_bfv.cu
    5_switchkey_methods_ckks 5_switchkey_methods_ckks.cu
    6_default_stream_usage 6_default_stream_usage.cu
    7_multi_stream_usage_way1 7_multi_stream_usage_way1.cu
    8_multi_stream_usage_way2 8_multi_stream_usage_way2.cu
    9_basic_bfv_logic 9_basic_bfv_logic.cu
    10_basic_ckks_logic 10_basic_ckks_logic.cu
    11_bfv_serialization 11_bfv_serialization.cu
    12_ckks_serialization 12_ckks_serialization.cu
    13_basic_tfhe 13_basic_tfhe.cu
)

function(add_example exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
    target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE OFF
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endfunction()

list(LENGTH EXECUTABLES EXECUTABLES_LENGTH)
math(EXPR EXECUTABLES_COUNT "${EXECUTABLES_LENGTH} / 2")
math(EXPR EXECUTABLES_COUNT_LOOP "${EXECUTABLES_COUNT} - 1")

foreach(i RANGE 0 ${EXECUTABLES_COUNT_LOOP})
    math(EXPR index1 "${i} * 2")
    math(EXPR index2 "${i} * 2 + 1")
    list(GET EXECUTABLES ${index1} exe)
    list(GET EXECUTABLES ${index2} source)
    add_example(${exe} ${source})
endforeach()
