cmake_minimum_required(VERSION 3.18)
project(npaged_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(npaged_core kernel.cpp)

target_compile_options(npaged_core PRIVATE -O3)
