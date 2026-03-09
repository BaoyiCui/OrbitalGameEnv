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
        environment/oge_settings.h
        environment/oge_settings.cpp
)

find_package(Threads REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)


target_link_libraries(oge PRIVATE
        Threads::Threads
        Eigen3::Eigen
)

configure_file("version.h.in" "version.h")

# Add submodules
add_subdirectory(common)
add_subdirectory(environment)

# C++ Library
if (BUILD_CPP_LIB OR BUILD_PYTHON_LIB)
    add_library(oge-lib oge_interface.cpp)
    set_target_properties(oge-lib PROPERTIES OUTPUT_NAME oge)
    target_link_libraries(oge-lib PUBLIC oge Eigen3::Eigen)
endif ()

# Python Library
if (BUILD_PYTHON_LIB)
    add_subdirectory(python)
endif ()

# Install cpp-lib and CMake config & pkg-config
if (UNIX AND BUILD_CPP_LIB)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS oge oge-lib
            EXPORT oge-export
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    # Install Header files
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
            FILES_MATCHING
            REGEX "\.h((pp)?|(xx?))$"
            REGEX "os_dependent\/.*(Win32)\.(h|h(xx)?|h(pp)?|c(xx)?|c(pp)?)$" EXCLUDE
    )

    # Install configured version file
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/version.h
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
    )

    # Export cmake targets
    install(EXPORT oge-export
            FILE ${PROJECT_NAME}-targets.cmake
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    )

    # Write config-version
    write_basic_package_version_file(
            ${PROJECT_NAME}-config-version.cmake
            VERSION ${PACKAGE_VERSION}
            COMPATIBILITY AnyNewerVersion)

    # Configure installable cmake config
    configure_package_config_file(
            ${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in
            ${PROJECT_NAME}-config.cmake
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

    # Install config-version and config
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
endif ()

