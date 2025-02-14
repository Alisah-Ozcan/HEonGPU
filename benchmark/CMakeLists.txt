# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/benchmark)

set(EXECUTABLES
    bfv_benchmark benchmark_bfv.cu
    ckks_benchmark benchmark_ckks.cu
)

function(add_benchmark exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart)
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
    add_benchmark(${exe} ${source})
endforeach()
