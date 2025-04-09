cmake_minimum_required(VERSION 3.10)
project(project)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")


set(CMAKE_PREFIX_PATH "/your/path/libtorch" CACHE PATH "Torch")
find_package(Torch REQUIRED)

add_executable(main main.cpp)


target_link_libraries(main PRIVATE "${TORCH_LIBRARIES}"  stdc++)

set_property(TARGET main PROPERTY CXX_STANDARD 17)
