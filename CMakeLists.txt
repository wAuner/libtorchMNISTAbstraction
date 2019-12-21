cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Capstone)


find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(Capstone src/main.cpp src/Architecture.cpp include/Architecture.h src/Model.cpp include/Model.h)
target_link_libraries(Capstone "${TORCH_LIBRARIES}")
set_property(TARGET Capstone PROPERTY CXX_STANDARD 11)
