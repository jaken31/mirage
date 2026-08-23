cmake_minimum_required(VERSION 3.20)

project(mirage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
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
    target_compile_options(mirage_sim PRIVATE /fsanitize=address)
    target_link_options(mirage_sim PRIVATE /INCREMENTAL:NO)
endif()

