cmake_minimum_required(VERSION 3.28)


# Build the C++ shared library
option(BUILD_CPP_LIB "Build C++ Interface" ON)

# Build the native Python bindings using pybind11
option(BUILD_PYTHON_LIB "Build Python Interface" ON)

# Build python vector interface
option(BUILD_VECTOR_LIB "Build Vector Interface" OFF)


# Set cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Overlay VCPKG custom triplets
if (NOT DEFINED VCPKG_OVERLAY_TRIPLETS)
    set(VCPKG_OVERLAY_TRIPLETS
            "${CMAKE_MODULE_PATH}/custom-triplets"
            CACHE STRING "")
endif ()

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

include(ParseVersion)
parse_version("version.txt" PREFIX OGE)
project(oge
        VERSION ${OGE_DEFAULT_VERSION}
        LANGUAGES CXX
)


add_subdirectory(src/oge)

