# Locate Thrust library and set THRUST_INCLUDE_DIR

find_path(THRUST_INCLUDE_DIR
  NAMES thrust/version.h
  HINTS
    /usr/include/cuda
    /usr/local/include
    /usr/local/cuda/include
    ${CUDA_INCLUDE_DIRS}
    ${CUDA_TOOLKIT_ROOT_DIR}
    ${CUDA_SDK_ROOT_DIR}
)

if (THRUST_INCLUDE_DIR)
  list(REMOVE_DUPLICATES THRUST_INCLUDE_DIR)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Thrust REQUIRED_VARS THRUST_INCLUDE_DIR)

if(Thrust_FOUND AND NOT TARGET Thrust)
  add_library(Thrust INTERFACE)
  target_include_directories(Thrust INTERFACE ${THRUST_INCLUDE_DIR})
endif()

mark_as_advanced(THRUST_INCLUDE_DIR)
