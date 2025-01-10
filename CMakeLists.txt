cmake_minimum_required(VERSION 3.10)
project(MeanShiftClusterOpenMp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
# Add include directories for OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target
add_executable(MeanShiftClusterOpenMP main.cpp)
target_link_libraries(MeanShiftClusterOpenMP OpenMP::OpenMP_CXX)
# Link OpenCV libraries
target_link_libraries(MeanShiftClusterOpenMP ${OpenCV_LIBS})
