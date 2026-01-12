set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_LOCATION}/examples)

add_executable(lowmem_resnet20_slot lowmem_resnet20_slot.cu)

target_link_libraries(lowmem_resnet20_slot PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
target_compile_options(lowmem_resnet20_slot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
set_target_properties(lowmem_resnet20_slot PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE OFF
    CUDA_RUNTIME_LIBRARY Static
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
