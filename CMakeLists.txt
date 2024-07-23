project(Samples)
cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -I/usr/include/habanalabs \
        -I${SPDLOG_ROOT} -Wall -g -L/usr/lib/habanalabs/ -lSynapse -ldl")

file(GLOB_RECURSE SOURCE_FILES *.cpp)
foreach(A_CPP_FILE ${SOURCE_FILES})
    get_filename_component(EXECUTABLE_NAME ${A_CPP_FILE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${A_CPP_FILE})
endforeach(A_CPP_FILE ${SOURCE_FILES})

