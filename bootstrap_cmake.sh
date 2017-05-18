#!/bin/sh
# This bash script prepares cmake 
# for action and setups an eclipse project

#Create a build directory Create a build directory in your project folder as a sibling to the source directory (/src):
mkdir ./build

# CMake Go to build directory
cd build

cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug ../src/
