cmake_minimum_required(VERSION 3.14)
project(portfolio_scorer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

# Standalone executable
add_executable(scorer_standalone scorer.cpp)
target_link_libraries(scorer_standalone PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(scorer_standalone PRIVATE -O3)

# Python module (optional, requires pybind11)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(scorer scorer.cpp)
    target_compile_definitions(scorer PRIVATE USE_PYBIND11)
    target_link_libraries(scorer PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(scorer PRIVATE -O3)
    message(STATUS "Building Python bindings with pybind11")
else()
    message(STATUS "pybind11 not found, skipping Python bindings")
endif()
