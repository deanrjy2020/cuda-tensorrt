add_executable(2.7-matmul-shared-memory
    src/main.cpp
    src/matmul_gpu_basic.cu
    src/matmul_gpu_shared.cu
    src/timer.cpp
    src/utils.cpp
)
target_include_directories(2.7-matmul-shared-memory PRIVATE include)
target_link_libraries(2.7-matmul-shared-memory PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.7-matmul-shared-memory PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.7-matmul-shared-memory PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
