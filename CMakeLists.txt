cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5OpenGL REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

add_subdirectory(volume)
add_subdirectory(painter)

file(GLOB CUDA_SOURCES *.cuda.cu)
file(GLOB CUDA_HEADERS *.cuda.h)

file(GLOB SOURCES *.cpp)
file(GLOB HEADERS *.h)
file(GLOB SHADERS *.vert *.frag)
file(GLOB RESOURCE_FILES *.qrc)

cuda_add_library(volrencuda ${CUDA_SOURCES} ${CUDA_HEADERS})
target_link_libraries(volrencuda ${CUDA_LIBRARIES})

qt5_add_resources(RESOURCES ${RESOURCE_FILES})
add_library(volren ${SOURCES} ${HEADERS} ${RESOURCES} ${SHADERS})
target_include_directories(volren PUBLIC volume painter)
target_link_libraries(volren ${OPENGL_LIBRARIES} volume painter volrencuda)
target_compile_features(volren PRIVATE cxx_range_for)
qt5_use_modules(volren Core OpenGL)
