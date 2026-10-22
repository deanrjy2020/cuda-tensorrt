cmake_minimum_required(VERSION 3.24)

project(2.1-dim-and-index LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.1-dim-and-index print_index.cu)

target_compile_features(2.1-dim-and-index PRIVATE cxx_std_17 cuda_std_17)

if(MSVC)
    # Read Chinese source comments as UTF-8.
    target_compile_options(2.1-dim-and-index PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
