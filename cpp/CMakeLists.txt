cmake_minimum_required(VERSION 3.14)
project(santa2025_collision_cpp)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    # Try to install pybind11 via pip
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install pybind11
        RESULT_VARIABLE PIP_RESULT
    )
    if(PIP_RESULT EQUAL 0)
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found and could not be installed")
    endif()
endif()

# Add the module
pybind11_add_module(santa2025_collision_cpp src/collision.cpp)

# Set C++ standard
target_compile_features(santa2025_collision_cpp PRIVATE cxx_std_17)

# Enable optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(santa2025_collision_cpp PRIVATE -O3 -march=native)
endif()
