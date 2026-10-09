add_executable(2.6-nsight-system-and-compute
    src/main.cpp
    src/matmul_gpu_basic.cu
    src/utils.cpp
)
target_include_directories(2.6-nsight-system-and-compute PRIVATE include)
target_link_libraries(2.6-nsight-system-and-compute PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.6-nsight-system-and-compute PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.6-nsight-system-and-compute PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
