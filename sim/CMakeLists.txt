cmake_minimum_required(VERSION 3.20)

project(mirage LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Two supported targets, each with the one compiler its flags below are written
# for. Anything else fails here, by name, rather than halfway through the build
# on a flag the compiler does not know.
set(MIRAGE_LINUX FALSE)
if(WIN32)
    if(NOT MSVC)
        message(FATAL_ERROR "On Windows this build needs MSVC; got ${CMAKE_CXX_COMPILER_ID}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
        message(FATAL_ERROR "On Linux this build needs GCC or Clang; got ${CMAKE_CXX_COMPILER_ID}")
    endif()
    set(MIRAGE_LINUX TRUE)
else()
    message(FATAL_ERROR "Unsupported platform ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}: "
                        "this build knows Windows x64 (MSVC) and Linux x86_64 (GCC or Clang)")
endif()

if(WIN32)
    FetchContent_Declare(
        mujoco
        URL "https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-windows-x86_64.zip"
        URL_HASH SHA256=ffe071c2747dd9513a1c59e7d2428bb678d887f9edec9eb9674b4288a248a8e9
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
else()
    FetchContent_Declare(
        mujoco
        URL "https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-linux-x86_64.tar.gz"
        # Computed locally from the downloaded asset, not copied from a page:
        # curl -sL <the URL above> | sha256sum
        URL_HASH SHA256=a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
endif()
# Defines mujoco_SOURCE_DIR, which the paths below use, so it must come first.
FetchContent_MakeAvailable(mujoco)

add_library(mujoco::mujoco SHARED IMPORTED)

if(WIN32)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/bin/mujoco.dll"
        IMPORTED_IMPLIB "${mujoco_SOURCE_DIR}/lib/mujoco.lib"
    )
else()
    # The real file, not the libmujoco.so symlink beside it, so the POST_BUILD
    # copy below lands a file named after the SONAME the binary asks for.
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${mujoco_SOURCE_DIR}/lib/libmujoco.so.3.12.0"
        IMPORTED_SONAME "libmujoco.so.3.12.0"
    )
endif()
set_target_properties(mujoco::mujoco PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${mujoco_SOURCE_DIR}/include"
    # Same path again on purpose: this does not add a search path, it marks it
    # as a system path so warnings-as-errors ignore warnings inside MuJoCo's
    # headers.
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

if(MIRAGE_LINUX)
    # The copy above puts libmujoco.so.3.12.0 beside the binary, as on Windows,
    # and $ORIGIN makes the loader look there - and only there, so the binary
    # never picks up some other MuJoCo from the system. It keeps working if
    # the build directory's _deps/ is deleted.
    set_target_properties(mirage_sim PROPERTIES
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "$ORIGIN"
    )
endif()

option(MIRAGE_ASAN "Build with AddressSanitizer (and UndefinedBehaviorSanitizer on Linux)" off)

if(MSVC)
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
else()
    # The same intent as the MSVC set above, spelled for GCC and Clang.
    set(GCC_WARNINGS
        -Wall -Wextra # Standard warning level (/W4)
        -Wshadow # A name hiding another one, part of /W4 (C4456-C4459)
        -Wconversion # Warn on conversions that may lose data (C4242)
        -Wsign-conversion # Signed/unsigned conversions, a superset of C4826's sign extension
        -Wpedantic # Strict standards conformance (/permissive-)
        -Werror # Treat all warnings as errors (/WX)
    )
    target_compile_options(mirage_sim PRIVATE ${GCC_WARNINGS})
    target_link_options(mirage_sim PRIVATE LINKER:--fatal-warnings)

    if(MIRAGE_ASAN)
        # MuJoCo 3.12.0's mjsan.h, which mujoco.h includes, turns on when it sees
        # AddressSanitizer and then puts __attribute__((always_inline)) after a
        # function definition's declarator. Clang accepts that; GCC 16 rejects it
        # as a hard error ("attributes are not allowed on a function-definition"),
        # which no warning flag or system include can silence.
        if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            message(FATAL_ERROR "MIRAGE_ASAN on Linux needs Clang: MuJoCo 3.12.0's mjsan.h "
                                "does not compile under GCC with -fsanitize=address. "
                                "Configure with -DCMAKE_CXX_COMPILER=clang++")
        endif()
        # UndefinedBehaviorSanitizer too, which MSVC does not have. By default it
        # prints a report and carries on, so a run with undefined behaviour in it
        # would still exit 0; -fno-sanitize-recover makes the first report fatal,
        # the way an AddressSanitizer report already is. -g for the same reason
        # as /Zi above.
        set(MIRAGE_SANITIZERS -fsanitize=address,undefined -fno-sanitize-recover=undefined)
        target_compile_options(mirage_sim PRIVATE ${MIRAGE_SANITIZERS} -fno-omit-frame-pointer -g)
        target_link_options(mirage_sim PRIVATE ${MIRAGE_SANITIZERS})
    endif()
endif()
