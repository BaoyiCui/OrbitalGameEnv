cmake_minimum_required(VERSION 3.5)
project(lambert_solver)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

add_library(lambert_solver MODULE lambert.cpp)
target_link_libraries(lambert_solver PRIVATE pybind11::module)