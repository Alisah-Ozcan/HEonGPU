# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

execute_process(
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE BUILD_SCRIPT_RESULT
    OUTPUT_VARIABLE BUILD_SCRIPT_OUTPUT
    ERROR_VARIABLE BUILD_SCRIPT_ERROR
)

if(NOT BUILD_SCRIPT_RESULT EQUAL 0)
    message(FATAL_ERROR "Thirdparty build script failed:\n${BUILD_SCRIPT_ERROR}")
endif()

message(STATUS "Thirdparty build completed successfully")

include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        branch-24.08
)

set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(rmm)

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)
add_subdirectory(RNGonGPU)

