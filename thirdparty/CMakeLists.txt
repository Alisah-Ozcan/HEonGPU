# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

execute_process(COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/build.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)

FetchContent_Declare(
    rmm
    GIT_REPOSITORY https://github.com/rapidsai/rmm
    GIT_TAG        branch-24.08
)

option(BUILD_TESTS OFF)
FetchContent_MakeAvailable(rmm)

add_subdirectory(GPU-FFT)
add_subdirectory(GPU-NTT)


