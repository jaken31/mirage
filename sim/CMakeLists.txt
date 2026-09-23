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
# Defines mujoco_SOURCE_DIR, which the paths below use, so it must come first.
FetchContent_MakeAvailable(mujoco)

add_library(mujoco::mujoco SHARED IMPORTED)

set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/bin/mujoco.dll"
    IMPORTED_IMPLIB "${mujoco_SOURCE_DIR}/lib/mujoco.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
    # Same path again on purpose: this does not add a search path, it marks it
    # as a system path so /W4 /WX ignore warnings inside MuJoCo's headers.
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw.git"
    GIT_TAG 3.5.1
    GIT_SHALLOW TRUE
)

# Plain set() is enough to override GLFW's own option() defaults: GLFW 3.5.1
# needs CMake >= 3.16, which turns on policy CMP0077 (option() respects an
# existing variable).
set(GLFW_BUILD_DOCS OFF)
set(GLFW_BUILD_TESTS OFF)
set(GLFW_BUILD_EXAMPLES OFF)
set(GLFW_INSTALL OFF)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
    nlohmann_json
    URL "https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz"
    # Computed locally from the downloaded asset, not copied from a page:
    # curl -sL <the URL above> | sha256sum
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# The release tarball, not the git repo: 115 KB instead of a clone with the
# whole test suite. set() overrides its option for the same reason as GLFW's.
set(JSON_BuildTests OFF)
FetchContent_MakeAvailable(nlohmann_json)

find_package(OpenGL REQUIRED)

add_executable(mirage_sim main.cpp gl_context.cpp policy.cpp truth.cpp shard_writer.cpp)
target_link_libraries(mirage_sim PRIVATE mujoco::mujoco glfw OpenGL::GL nlohmann_json::nlohmann_json)

add_custom_command(TARGET mirage_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:mujoco::mujoco>"
        "$<TARGET_FILE_DIR:mirage_sim>")

option(MIRAGE_ASAN "Build with AddressSanitizer" off)

set(MSVC_WARNINGS
    /W4 # Standard warning level
    /w14242 # Warn on conversions that may lose data
    /WX # Treat all warnings as errors
    /w14826 # Warn on sign-extending conversions, which can change values at runtime
    /permissive- # Strict standards conformance
)
target_compile_options(mirage_sim PRIVATE /Zc:__cplusplus ${MSVC_WARNINGS})
target_link_options(mirage_sim PRIVATE /WX)

if(MIRAGE_ASAN)
    # /Zi is required: without it MSVC warns C5072, which /WX makes fatal. And an
    # AddressSanitizer report with no debug symbols is close to useless.
    target_compile_options(mirage_sim PRIVATE /fsanitize=address /Zi)
    target_link_options(mirage_sim PRIVATE /INCREMENTAL:NO /DEBUG)
    # MSVC always links the sanitizer runtime as a DLL, and that DLL sits next
    # to cl.exe, which is only on PATH inside a Developer prompt. Copy it next
    # to the binary so the sanitizer build runs from any shell.
    get_filename_component(MIRAGE_MSVC_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
    add_custom_command(TARGET mirage_sim POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${MIRAGE_MSVC_BIN}/clang_rt.asan_dynamic-x86_64.dll"
            "$<TARGET_FILE_DIR:mirage_sim>")
endif()

