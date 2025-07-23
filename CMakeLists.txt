cmake_minimum_required(VERSION 3.10)

project(CPPNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(CPPNN ${SOURCES} ${HEADERS})

target_include_directories(CPPNN PRIVATE ${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)

target_link_libraries(CPPNN Vulkan::Vulkan OpenMP::OpenMP_CXX)

find_program(GLSL_VALIDATOR glslangValidator REQUIRED)

file (GLOB_RECURSE GLSL_COMP_SHADERS "${CMAKE_SOURCE_DIR}/shaders/*.comp")

foreach(SHADER ${GLSL_COMP_SHADERS})
    message(STATUS "Compiling shader ${SHADER}")
    get_filename_component(SHADER_NAME ${SHADER} NAME)

    set(SPRIV "${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.spv")
    message(STATUS "SPRIV: ${SPRIV}")
    add_custom_command(
        OUTPUT ${SPRIV}
        COMMAND ${GLSL_VALIDATOR} -V ${SHADER} -o ${SPRIV}
        DEPENDS ${SHADER}
    )
    list(APPEND SHADERS_SPV ${SPRIV})
endforeach( SHADER )

add_custom_target(shaders ALL DEPENDS ${SHADERS_SPV})
