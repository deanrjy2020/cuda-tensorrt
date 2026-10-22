cmake_minimum_required(VERSION 3.24)

project(2.8-bank-conflict LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.8-bank-conflict
    main.cpp
    matmul_gpu_bank_conflict_pad.cu
    matmul_gpu_bank_conflict.cu
    matmul_gpu_basic.cu
    matmul_gpu_shared.cu
    timer.cpp
    utils.cpp
)

target_compile_features(2.8-bank-conflict PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(2.8-bank-conflict PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2.8-bank-conflict PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.8-bank-conflict PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.8-bank-conflict PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
