# Copyright 2024-2025 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(HEonGPU VERSION 1.1 LANGUAGES C CXX CUDA ASM)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Debug or Release)" )
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug)
message(STATUS "Build HEonGPU version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(BINARY_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++ Standard is: ${CMAKE_CXX_STANDARD}")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")

set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG   "${CMAKE_CUDA_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

find_package(CUDAToolkit REQUIRED)
find_package(Thrust REQUIRED)
find_package(OpenMP REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "CUDA GPU architectures" FORCE)
endif()
message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES} for HEonGPU")

##################
# Export Library #
##################

set(RUNTIME_DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
set(LIBRARY_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(ARCHIVE_DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
set(INCLUDES_DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
set(INCLUDES_INSTALL_DIR ${INCLUDES_DESTINATION}/HEonGPU-${PROJECT_VERSION})

set(HEonGPU_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(HEonGPU_VERSION_CONFIG "${HEonGPU_GENERATED_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(HEonGPU_PROJECT_CONFIG "${HEonGPU_GENERATED_DIR}/${PROJECT_NAME}Config.cmake")
set(HEonGPU_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(HEonGPU_CONFIG_INSTALL_DIR "lib/cmake/${PROJECT_NAME}-${PROJECT_VERSION}")
set(HEonGPU_NAMESPACE "${PROJECT_NAME}::")
set(HEonGPU_VERSION 1.1)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${HEonGPU_VERSION_CONFIG}" VERSION ${HEonGPU_VERSION} COMPATIBILITY SameMajorVersion
)
configure_file("${HEonGPU_SOURCE_DIR}/cmake/Config.cmake.in" "${HEonGPU_PROJECT_CONFIG}" @ONLY)

install(
    FILES "${HEonGPU_PROJECT_CONFIG}" "${HEonGPU_VERSION_CONFIG}"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

install(
    FILES "${HEonGPU_SOURCE_DIR}/cmake/FindCCCL.cmake"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

install(
    EXPORT "${HEonGPU_TARGETS_EXPORT_NAME}"
    NAMESPACE "${HEonGPU_NAMESPACE}"
    DESTINATION "${HEonGPU_CONFIG_INSTALL_DIR}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

set(RNGonGPU_USE_INTERNAL_GPUNTT
    OFF CACHE BOOL
    "Force using global GPU-NTT (thirdparty/GPU-NTT) instead of RNGonGPU's internal" FORCE)

add_subdirectory(thirdparty)
add_subdirectory(src)

#########
# Tests #
#########

option(HEonGPU_BUILD_TESTS "Build HEonGPU Tests" OFF)
message(STATUS "HEonGPU_BUILD_TESTS: ${HEonGPU_BUILD_TESTS}")
if(HEonGPU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

############
# Examples #
############

option(HEonGPU_BUILD_EXAMPLES "Build HEonGPU Examples" OFF)
message(STATUS "HEonGPU_BUILD_EXAMPLES: ${HEonGPU_BUILD_EXAMPLES}")
if(HEonGPU_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()

##############
# BENCHMARKS #
##############

option(HEonGPU_BUILD_BENCHMARKS "Build HEonGPU Benchmarks" OFF)
message(STATUS "HEonGPU_BUILD_BENCHMARKS: ${HEonGPU_BUILD_BENCHMARKS}")
if(HEonGPU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
