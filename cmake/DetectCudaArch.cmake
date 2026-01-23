function(heongpu_detect_cuda_arch out_var)
    set(${out_var} "" PARENT_SCOPE)

    # Try nvidia-smi first (most reliable when available)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND "${NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _smi_output
            ERROR_VARIABLE _smi_err
            RESULT_VARIABLE _smi_result
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    else()
        set(_smi_result 1)
        set(_smi_output "")
    endif()

    if(_smi_result EQUAL 0 AND NOT _smi_output STREQUAL "")
        string(REPLACE "\n" ";" _smi_list "${_smi_output}")
        list(GET _smi_list 0 _smi_cap)
        string(STRIP "${_smi_cap}" _smi_cap)
        string(REGEX REPLACE "[^0-9.]" "" _smi_cap "${_smi_cap}")
        if(_smi_cap MATCHES "^[0-9]+\\.[0-9]+$")
            string(REPLACE "." "" _smi_arch "${_smi_cap}")
            set(${out_var} "${_smi_arch}" PARENT_SCOPE)
            return()
        endif()
        if(_smi_cap MATCHES "^[0-9]+$")
            set(${out_var} "${_smi_cap}" PARENT_SCOPE)
            return()
        endif()
    endif()

    if(NOT CMAKE_CUDA_COMPILER)
        return()
    endif()

    set(_src_dir "${CMAKE_BINARY_DIR}/heongpu_detect_cuda_arch")
    file(MAKE_DIRECTORY "${_src_dir}")
    set(_src_file "${_src_dir}/detect_cuda_arch.cu")

    file(WRITE "${_src_file}" [=[
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) return 1;
  int dev = 0;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 2;
  std::printf("%d", prop.major * 10 + prop.minor);
  return 0;
}
]=])

    try_run(
        _run_result _compile_result
        "${_src_dir}"
        "${_src_file}"
        CMAKE_FLAGS "-DCMAKE_CUDA_STANDARD=17" "-DCMAKE_CUDA_STANDARD_REQUIRED=ON"
        RUN_OUTPUT_VARIABLE _run_output
    )

    if(_compile_result AND _run_result EQUAL 0)
        string(STRIP "${_run_output}" _arch)
        if(_arch MATCHES "^[0-9]+$")
            set(${out_var} "${_arch}" PARENT_SCOPE)
        endif()
    endif()
endfunction()
