add_executable(2.1-dim-and-index src/print_index.cu)

if(MSVC)
    # The teaching source contains Chinese comments; avoid the system code page.
    target_compile_options(2.1-dim-and-index PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
endif()
