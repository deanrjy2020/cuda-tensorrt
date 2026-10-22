cmake_minimum_required(VERSION 3.24)

project(2.5-device-info LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.5-device-info
    main.cpp
)

target_compile_features(2.5-device-info PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(2.5-device-info PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2.5-device-info PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_options(2.5-device-info PRIVATE /utf-8)
endif()
