cmake_minimum_required(VERSION 3.10)
project(project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_compile_options(-O3 -march=native)


include_directories(include)


set(HEADERS
    include/Index/enum.h
    include/Utils/deal_data_functions.h
    include/Utils/space.h
    include/Utils/utils.h
    include/def.h
    include/Model/dataset.h
    include/Model/loss_function.h
    include/Model/model.h
    include/Model/train.h
) 


set(SOURCES
    src/main.cpp
    src/Index/enum.cpp
    src/Utils/space.cpp
    src/Utils/deal_data_functions.cpp
    src/Utils/utils.cpp
    src/Model/dataset.cpp
    src/Model/loss_function.cpp
    src/Model/model.cpp
    src/Model/train.cpp
)
set(CMAKE_PREFIX_PATH "/your/path/libtorch" CACHE PATH "Torch")
find_package(Torch REQUIRED)

add_executable(main src/main.cpp ${SOURCES} ${HEADERS})
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)

