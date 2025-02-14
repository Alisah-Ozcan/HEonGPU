# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples/basic)

set(EXECUTABLES
    1_basic_bfv 1_basic_bfv.cu
    2_basic_ckks 2_basic_ckks.cu
    3_switchkey_methods_bfv 3_switchkey_methods_bfv.cu
    4_switchkey_methods_ckks 4_switchkey_methods_ckks.cu
    5_default_stream_usage 5_default_stream_usage.cu
    6_multi_stream_usage_way1 6_multi_stream_usage_way1.cu
    7_multi_stream_usage_way2 7_multi_stream_usage_way2.cu
    8_basic_bfv_logic 8_basic_bfv_logic.cu
    9_basic_ckks_logic 9_basic_ckks_logic.cu
)

function(add_example exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
    target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
    target_link_libraries(${exe} PRIVATE -lgomp)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
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