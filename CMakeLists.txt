cmake_minimum_required(VERSION 3.20)
project(MachineLearning LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core)

add_subdirectory(Matrix)
add_subdirectory(MatrixTest)
add_subdirectory(MachineLearning)