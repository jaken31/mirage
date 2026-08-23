cmake_minimum_required(VERSION 3.20)

project(mirage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(mirage_sim main.cpp)

option(MIRAGE_ASAN "Build with AddressSanitizer" off)

set(MSVC_WARNINGS
    /W4 # Baseline reasonable warnings
    /w14242 # 'identifier': conversion from 'type1' to 'type2', possible loss of data
    /WX # Treats all warnings as errors
    /w14826 # Conversion from 'type1' to 'type2' is sign-extended. This may cause unexpected runtime behavior.
    /permissive- # standards conformance mode for MSVC compiler.
)
target_compile_options(mirage_sim PRIVATE /Zc:__cplusplus ${MSVC_WARNINGS})
target_link_options(mirage_sim PRIVATE /WX)

if(MIRAGE_ASAN)
    # /Zi is not optional: MSVC warns C5072 without it, and /WX makes that fatal.
    # Without debug info an ASan report has no symbols, which defeats the purpose.
    target_compile_options(mirage_sim PRIVATE /fsanitize=address /Zi)
    target_link_options(mirage_sim PRIVATE /INCREMENTAL:NO /DEBUG)
    # MSVC always links the ASan runtime dynamically, whatever the CRT setting,
    # and its DLL lives next to cl.exe - on PATH only inside a Developer prompt.
    # Copy it beside the binary so the sanitizer build runs from any shell.
    get_filename_component(MIRAGE_MSVC_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(TARGET mirage_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MIRAGE_MSVC_BIN}/clang_rt.asan_dynamic-x86_64.dll"
            "$<TARGET_FILE_DIR:mirage_sim>")
endif()

