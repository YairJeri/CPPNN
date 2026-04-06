cmake_minimum_required(VERSION 3.10)

project(CPPNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Vulkan REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

# ImGui & ImPlot sources
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/thirdparty/imgui")
set(IMPLOT_DIR "${CMAKE_SOURCE_DIR}/thirdparty/implot")

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot_demo.cpp
)

file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(CPPNN ${SOURCES} ${IMGUI_SOURCES})

target_include_directories(CPPNN PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
)

target_link_libraries(CPPNN 
    Vulkan::Vulkan 
    OpenMP::OpenMP_CXX 
    OpenGL::GL 
    glfw
)

find_program(GLSL_VALIDATOR glslangValidator REQUIRED)

file(GLOB_RECURSE GLSL_COMP_SHADERS "${CMAKE_SOURCE_DIR}/shaders/*.comp")

foreach(SHADER ${GLSL_COMP_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    set(SPRIV "${CMAKE_SOURCE_DIR}/shaders/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPRIV}
        COMMAND ${GLSL_VALIDATOR} -V ${SHADER} -o ${SPRIV}
        DEPENDS ${SHADER}
    )
    list(APPEND SHADERS_SPV ${SPRIV})
endforeach(SHADER)

add_custom_target(shaders ALL DEPENDS ${SHADERS_SPV})
add_dependencies(CPPNN shaders)
