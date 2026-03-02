find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind (installed via pip in build-system.requires)
execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Use nanobind_add_module for proper scikit-build-core integration
nanobind_add_module(_oge_py MODULE oge_python_interface.cpp)
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_XLA_LIB)

elseif (BUILD_VECTOR_LIB)
    target_sources(_oge_py PRIVATE oge_vector_python_interface.cpp)
    target_compile_definitions(_oge_py PRIVATE BUILD_VECTOR_LIB)
endif ()

target_link_libraries(_oge_py PUBLIC oge oge-lib)

# Install the compiled modules
install(TARGETS _oge_py LIBRARY DESTINATION oge_py)

# Install Python source files
install(FILES
        __init__.py
        env.py
        registration.py
        vector_env.py
        py.typed
        DESTINATION oge_py)

# Install stub files if they exist
install(FILES
        __init__.pyi
        DESTINATION oge_py
        OPTIONAL)