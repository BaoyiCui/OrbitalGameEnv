cmake_minimum_required(VERSION 3.5)
project(lambert_solver)

set(CMAKE_CXX_STANDARD 11)

# find_package(pybind11) is the recommended way to use pybind11 with CMake.
# It automatically finds the necessary include directories and linker flags.
find_package(pybind11 REQUIRED)

# pybind11_add_module is a helper function provided by pybind11.
# It simplifies the creation of a Python extension module.
# It's a replacement for add_library() and target_link_libraries().
pybind11_add_module(lambert_solver lambert.cpp)