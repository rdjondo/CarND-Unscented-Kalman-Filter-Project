project(UnscentedKF)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
    ukf.cpp
    main.cpp
    tools.cpp
    test.cpp)

add_executable(UnscentedKF ${sources})
