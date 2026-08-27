cmake_minimum_required(VERSION 3.20)

project(mirage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    mujoco
    URL "https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-windows-x86_64.zip"
    URL_HASH SHA256=ffe071c2747dd9513a1c59e7d2428bb678d887f9edec9eb9674b4288a248a8e9
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Defines mujoco_SOURCE_DIR. Must run before the paths below are read.
FetchContent_MakeAvailable(mujoco)

add_library(mujoco::mujoco SHARED IMPORTED)

set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/bin/mujoco.dll"
    IMPORTED_IMPLIB "${mujoco_SOURCE_DIR}/lib/mujoco.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
    # Same path again on purpose: this property does not add a search dir, it
    # only marks one as system so /W4 /WX ignores warnings in MuJoCo's headers.
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw.git"
    GIT_TAG 3.5.1
    GIT_SHALLOW TRUE
)

# Plain set() is enough: GLFW 3.5.1 requires CMake >= 3.16, so CMP0077 is NEW
# and these override the option() calls inside GLFW's own CMakeLists.
set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_INSTALL OFF)
FetchContent_MakeAvailable(glfw)

find_package(OpenGL REQUIRED)

add_executable(mirage_sim main.cpp gl_context.cpp policy.cpp truth.cpp)
target_link_libraries(mirage_sim PRIVATE mujoco::mujoco glfw OpenGL::GL)

add_custom_command(TARGET mirage_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:mujoco::mujoco>"
        "$<TARGET_FILE_DIR:mirage_sim>")

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

