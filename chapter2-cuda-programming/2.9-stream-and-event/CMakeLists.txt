add_executable(2.9-stream-and-event
    src/gelu.cu
    src/main.cpp
    src/stream.cu
    src/timer.cpp
    src/utils.cpp
)
target_include_directories(2.9-stream-and-event PRIVATE include)
target_link_libraries(2.9-stream-and-event PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_definitions(2.9-stream-and-event PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.9-stream-and-event PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
