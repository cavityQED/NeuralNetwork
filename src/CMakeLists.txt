cmake_minimum_required( VERSION 3.0 )
set( CMAKE_CXX_COMPILER /usr/bin/g++)
set( CMAKE_CXX_STANDARD 14)
project(NeuralNetwork)

include_directories(/usr/lib/x86_64-linux-gnu)
include_directories(/usr/local/lib)
include_directories(/home/mike/Desktop/Mesh/boost/include)
include_directories(/home/mike/Desktop/Mesh/boost/include/boost/numeric/ublas)
include_directories(/home/mike/Desktop/neuralNetwork/src)
include_directories(/home/mike/Desktop/neuralNetwork/src/layers)
include_directories(/home/mike/Desktop/neuralNetwork/src/utilities)
include_directories(/usr/include/mpi)

add_subdirectory(/home/mike/Desktop/neuralNetwork/src/layers /home/mike/Desktop/neuralNetwork/build/layers)
add_subdirectory(/home/mike/Desktop/neuralNetwork/src/utilities /home/mike/Desktop/neuralNetwork/build/utilities)

find_package(MPI REQUIRED)

add_library(networkLib ImageReader.cpp ParallelNet.cpp NeuralNet.cpp)
target_link_libraries(networkLib mpi Layers Utilities)

add_executable(parallel_net parallel_net.cpp)
target_link_libraries(parallel_net networkLib)
