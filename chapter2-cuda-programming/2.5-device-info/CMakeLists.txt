add_executable(2.5-device-info src/main.cpp)
target_include_directories(2.5-device-info PRIVATE include)
target_link_libraries(2.5-device-info PRIVATE CUDA::cudart)

if(MSVC)
    target_compile_options(2.5-device-info PRIVATE /utf-8)
endif()
