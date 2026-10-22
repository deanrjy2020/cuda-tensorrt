cmake_minimum_required(VERSION 3.24)

project(2.3-matmul-basic LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.3-matmul-basic
    main.cpp
    matmul_cpu.cpp
    matmul_gpu_basic.cu
    utils.cpp
)

target_compile_features(2.3-matmul-basic PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(2.3-matmul-basic PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2.3-matmul-basic PRIVATE CUDA::cudart)

if(MSVC)
    # Keep the teaching source's standard C functions available with MSVC.
    target_compile_definitions(2.3-matmul-basic PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.3-matmul-basic PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
