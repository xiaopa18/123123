cmake_minimum_required(VERSION 3.10)
project(project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

add_compile_options(-O3 -march=native)


include_directories(include)


set(HEADERS
    include/Clustering/clustering_params.h
    include/Clustering/kmeans.h
    include/Index/enum.h
    include/Index/ivf.h
    include/Utils/deal_data_functions.h
    include/Utils/space.h
    include/Utils/utils.h
    include/Utils/monitoring.hpp
    include/def.h
    include/Model/dataset.h
    include/Model/model.h
)


set(COMMON_SOURCES
    src/Clustering/clustering_params.cpp
    src/Clustering/kmeans.cpp
    src/Index/enum.cpp
    src/Index/ivf.cpp
    src/Utils/space.cpp
    src/Utils/deal_data_functions.cpp
    src/Utils/utils.cpp
    src/Model/dataset.cpp
    src/Model/model.cpp
)

set(CMAKE_PREFIX_PATH "/your/path/libtorch" CACHE PATH "Torch")
find_package(Torch REQUIRED)

add_executable(main src/main.cpp ${COMMON_SOURCES})
target_link_libraries(main "${TORCH_LIBRARIES}")
