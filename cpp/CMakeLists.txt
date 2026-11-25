cmake_minimum_required(VERSION 3.18)

project(RoboticsBackend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)

find_package(Python REQUIRED
    COMPONENTS Interpreter Development.Module
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    robotics_cpp
    bindings/bindings.cpp
    src/dataset_processor.cpp
)

target_include_directories(
    robotics_cpp PRIVATE include
)