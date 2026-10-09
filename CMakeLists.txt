cmake_minimum_required(VERSION 3.24)

# RTX 4070 / Ada. Override with -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA compute capabilities")
endif()

project(cuda_tensorrt LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Register demos individually as they are migrated from Makefile to CMake.
# Each demo target must exactly match its directory name, e.g. 2.1-dim-and-index.
add_subdirectory(chapter2-cuda-programming/2.1-dim-and-index)
add_subdirectory(chapter2-cuda-programming/2.2-cpp-cuda-interactive)
add_subdirectory(chapter2-cuda-programming/2.3-matmul-basic)
add_subdirectory(chapter2-cuda-programming/2.4-error-handler)
add_subdirectory(chapter2-cuda-programming/2.5-device-info)
add_subdirectory(chapter2-cuda-programming/2.6-nsight-system-and-compute)
add_subdirectory(chapter2-cuda-programming/2.7-matmul-shared-memory)
add_subdirectory(chapter2-cuda-programming/2.8-bank-conflict)
add_subdirectory(chapter2-cuda-programming/2.9-stream-and-event)
add_subdirectory(chapter2-cuda-programming/2.9.1-stream-and-event-more-tests)

# Discover the user's OpenCV package; -DOpenCV_DIR=... takes precedence.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs
    HINTS "$ENV{USERPROFILE}/bin/opencv/build" "$ENV{HOME}/bin/opencv/build"
)
add_subdirectory(chapter2-cuda-programming/2.10-bilinear-interpolation)
add_subdirectory(chapter2-cuda-programming/2.11-bilinear-interpolation-template)
