cmake_minimum_required(VERSION 3.5)
project(all1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_executable(all1 src/main.cpp src/LIME.h)


find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
target_link_libraries(all1 Threads::Threads ${OpenCV_LIBS} -lpthread -O3)
