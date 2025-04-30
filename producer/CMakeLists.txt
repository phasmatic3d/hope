cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(hope_producer VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Draco subdirectory
add_subdirectory(deps/draco)

# Add the executable
add_executable(hope_producer src/producer.cpp)

# Add Draco's include directories (if needed)
target_include_directories(hope_producer PRIVATE
    ${CMAKE_BINARY_DIR}
    deps/draco/src
)
target_link_libraries(hope_producer PRIVATE draco)