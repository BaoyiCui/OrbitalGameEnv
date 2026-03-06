# Include src/ and cmake binary directory
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Compiler options
add_compile_options(
        "$<$<CONFIG:RELEASE>:-O3>"
        "$<$<CONFIG:DEBUG>:-O0>"
        "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>"
        "$<$<CXX_COMPILER_ID:MSVC>:/W4>"
)

# oge object containing source files
add_library(oge OBJECT
        simcore/propagator.h
        simcore/propagator.cpp
        simcore/utils.h
        simcore/utils.cpp
        common/constants.h
        environment/orbital_game_environment.h
        environment/orbital_game_environment.cpp
        common/constants.cpp
        environment/oge_wrapper.h
        environment/oge_wrapper.cpp
)

find_package(Threads REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


target_link_libraries(oge PRIVATE
        Threads::Threads
        Eigen3::Eigen
)

# Add submodules
add_subdirectory(common)
add_subdirectory(environment)

# C++ Library
if (BUILD_CPP_LIB OR BUILD_PYTHON_LIB)
    add_library(oge-lib oge_interface.cpp)
    set_target_properties(oge-lib PROPERTIES OUTPUT_NAME oge)
    target_link_libraries(oge-lib PUBLIC oge)
endif ()

# Python Library
if (BUILD_PYTHON_LIB)
    add_subdirectory(python)
endif ()

