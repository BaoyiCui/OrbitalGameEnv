cmake_minimum_required(VERSION 3.28)

project(OrbitalGameEnv VERSION 0.1.0 LANGUAGES CXX)

# Build the C++ shared library
option(BUILD_CPP_LIB "Build C++ Interface" ON)

# Build the native Python bindings using pybind11
option(BUILD_PYTHON_LIB "Build Python Interface" ON)

# Discover VCPKG default triplet
if (DEFINED ENV{VCPKG_DEFAULT_TRIPLET} AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET
            "$ENV{VCPKG_DEFAULT_TRIPLET}"
            CACHE STRING "")
endif ()

# Discover VCPKG toolchain
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    # VCPKG_ROOT is what Microsoft recommends,
    # VCPKG_INSTALLATION_ROOT is what's defined on Azure
    if (DEFINED ENV{VCPKG_ROOT})
        set(CMAKE_TOOLCHAIN_FILE
                "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
                CACHE STRING "")
    elseif (DEFINED ENV{VCPKG_INSTALLATION_ROOT})
        set(CMAKE_TOOLCHAIN_FILE
                "$ENV{VCPKG_INSTALLATION_ROOT}/scripts/buildsystems/vcpkg.cmake"
                CACHE STRING "")
    endif ()
endif ()

if (DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Successfully configured vcpkg: ${CMAKE_TOOLCHAIN_FILE}")
else ()
    message(WARNING "vcpkg not found!")
endif ()

if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR [=[
    Source and build directories cannot be the same.
    You should probably also remove CMakeFiles/ and CMakeCache.txt.
  ]=])
endif ()

add_subdirectory(src/oge)

