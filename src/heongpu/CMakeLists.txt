# Copyright 2024 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

file(GLOB_RECURSE LIB_SOURCE_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    lib/host/*.cu
    lib/kernel/*.cu
    lib/util/*.cu
)

file(GLOB_RECURSE LIB_HEADER_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    include/host/*.cuh
    include/kernel/*.cuh
    include/util/*.cuh
    include/host/*.h
    include/kernel/*.h
    include/util/*.h
)

add_library(heongpu STATIC ${LIB_SOURCE_FILES})

set_target_properties(heongpu PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  OUTPUT_NAME "heongpu-${PROJECT_VERSION}"
)

target_include_directories(
    heongpu
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/host>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/kernel>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/util>
    $<INSTALL_INTERFACE:include/HEonGPU-${PROJECT_VERSION}>
)


target_link_libraries(heongpu PUBLIC 
    ntt
    fft
    gmp
    rmm
    CUDA::curand
)

set_target_properties(heongpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
        CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_RUNTIME_LIBRARY Static
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

install(TARGETS heongpu
  EXPORT ${HEonGPU_TARGETS_EXPORT_NAME}
  RUNTIME DESTINATION ${RUNTIME_DESTINATION}
  LIBRARY DESTINATION ${LIBRARY_DESTINATION}
  ARCHIVE DESTINATION ${ARCHIVE_DESTINATION}
)

install(FILES ${LIB_HEADER_FILES} DESTINATION ${INCLUDES_INSTALL_DIR})
