cmake_minimum_required(VERSION 3.18)
project(SuperResolutionApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.5 REQUIRED COMPONENTS core imgproc highgui imgcodecs dnn dnn_superres photo)

add_executable(super_resolution src/main.cpp)
target_link_libraries(super_resolution ${OpenCV_LIBS})

if(NOT MSVC)
    target_compile_options(super_resolution PRIVATE -O3 -march=native -mtune=native)
endif()
