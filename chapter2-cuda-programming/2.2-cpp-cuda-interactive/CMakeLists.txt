cmake_minimum_required(VERSION 3.24)

project(2.2-cpp-cuda-interactive LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.2-cpp-cuda-interactive
    main.cpp
    print_index.cu
)

target_compile_features(2.2-cpp-cuda-interactive PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(2.2-cpp-cuda-interactive PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2.2-cpp-cuda-interactive PRIVATE CUDA::cudart)

if(MSVC)
    # Read the teaching source and headers as UTF-8 in both compilers.
    target_compile_options(2.2-cpp-cuda-interactive PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
