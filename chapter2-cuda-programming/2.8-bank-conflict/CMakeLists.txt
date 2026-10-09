add_executable(2.8-bank-conflict
    src/main.cpp
    src/matmul_gpu_bank_conflict_pad.cu
    src/matmul_gpu_bank_conflict.cu
    src/matmul_gpu_basic.cu
    src/matmul_gpu_shared.cu
    src/timer.cpp
    src/utils.cpp
)
target_include_directories(2.8-bank-conflict PRIVATE include)
target_link_libraries(2.8-bank-conflict PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.8-bank-conflict PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.8-bank-conflict PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
