cmake_minimum_required(VERSION 3.8)
project(untitled1)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES hw1_main.cpp normal_random.cpp hw1.h matrix.cpp normal_random.h matrix.h nn.cpp nn.h activation.cpp activation.h hw2.h main.cpp)
add_executable(untitled1 ${SOURCE_FILES})