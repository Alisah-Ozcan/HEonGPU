# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.0
  SOURCE_DIR     "${CMAKE_BINARY_DIR}/googletest-src"
)

FetchContent_MakeAvailable(googletest)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/test)

set(EXECUTABLES
    bfv_addition_testcases test_bfv_addition.cu
    bfv_encoding_testcases test_bfv_encoding.cu
    bfv_encryption_testcases test_bfv_encryption.cu
    bfv_multiplication_testcases test_bfv_multiplication.cu
    bfv_relinearization_testcases test_bfv_relinearization.cu
    bfv_rotation_method_1_testcases test_bfv_rotation_method_1.cu
    bfv_rotation_method_2_testcases test_bfv_rotation_method_2.cu

    ckks_addition_testcases test_ckks_addition.cu
    ckks_encoding_testcases test_ckks_encoding.cu
    ckks_encryption_testcases test_ckks_encryption.cu
    ckks_multiplication_testcases test_ckks_multiplication.cu
    ckks_relinearization_testcases test_ckks_relinearization.cu
    ckks_rotation_method_1_testcases test_ckks_rotation_method_1.cu
    ckks_rotation_method_2_testcases test_ckks_rotation_method_2.cu
)

function(add_test exe source)
    add_executable(${exe} ${source})
    target_link_libraries(${exe} PRIVATE heongpu CUDA::cudart gtest_main)
    set_target_properties(${exe} PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        POSITION_INDEPENDENT_CODE ON
        CUDA_RUNTIME_LIBRARY Static
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
    add_test(${exe} ${source})
endforeach()
