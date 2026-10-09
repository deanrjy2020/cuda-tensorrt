add_executable(2.3-matmul-basic
    src/main.cpp
    src/matmul_cpu.cpp
    src/matmul_gpu_basic.cu
    src/utils.cpp
)
target_include_directories(2.3-matmul-basic PRIVATE include)
target_link_libraries(2.3-matmul-basic PRIVATE CUDA::cudart)

if(MSVC)
    # Keep the teaching source's standard C functions available with MSVC.
    target_compile_definitions(2.3-matmul-basic PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.3-matmul-basic PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
