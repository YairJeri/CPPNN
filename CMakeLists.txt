cmake_minimum_required(VERSION 3.15)
project(CPPNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

add_subdirectory(thirdparty/vulkan)
add_subdirectory(thirdparty/imgui)
add_subdirectory(thirdparty/implot)

include_directories(
    include
    thirdparty/vulkan/include
    thirdparty/imgui
    thirdparty/implot
)

set(SOURCES
    src/main.cpp
    src/TensorBase.cpp
    src/TensorCPU.cpp
)

add_executable(CPPNN ${SOURCES})
target_link_libraries(CPPNN PRIVATE OpenMP::OpenMP_CXX)
