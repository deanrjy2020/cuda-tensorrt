add_executable(2.10-bilinear-interpolation
    src/main.cpp
    src/preprocess.cpp
    src/preprocess.cu
    src/timer.cpp
    src/utils.cpp
)
target_include_directories(2.10-bilinear-interpolation PRIVATE include)
target_include_directories(2.10-bilinear-interpolation SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(2.10-bilinear-interpolation PRIVATE CUDA::cudart ${OpenCV_LIBS})

# Resolve image paths independently of the launch working directory.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/results")
target_compile_definitions(2.10-bilinear-interpolation PRIVATE
    DEMO_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data/"
    DEMO_RESULTS_DIR="${CMAKE_CURRENT_BINARY_DIR}/results/"
)
if(MSVC)
    target_compile_definitions(2.10-bilinear-interpolation PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(2.10-bilinear-interpolation PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()

if(WIN32)
    add_custom_command(TARGET 2.10-bilinear-interpolation POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:2.10-bilinear-interpolation> $<TARGET_FILE_DIR:2.10-bilinear-interpolation>
        COMMAND_EXPAND_LISTS
    )
endif()
