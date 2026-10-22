cmake_minimum_required(VERSION 3.24)

project(2.9-stream-and-event LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(2.9-stream-and-event
    gelu.cu
    main.cpp
    stream.cu
    timer.cpp
    utils.cpp
)

target_compile_features(2.9-stream-and-event PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(2.9-stream-and-event PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2.9-stream-and-event PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.9-stream-and-event PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.9-stream-and-event PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
