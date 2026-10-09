add_executable(2.2-cpp-cuda-interactive
    src/main.cpp
    src/print_index.cu
)
target_include_directories(2.2-cpp-cuda-interactive PRIVATE include)
target_link_libraries(2.2-cpp-cuda-interactive PRIVATE CUDA::cudart)

if(MSVC)
    # Read the teaching source and headers as UTF-8 in both compilers.
    target_compile_options(2.2-cpp-cuda-interactive PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
