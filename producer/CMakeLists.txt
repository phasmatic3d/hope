cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(hope_producer VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add Draco subdirectory
add_subdirectory(deps/draco)

# Add Realsense subdirectory
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable librealsense examples")
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "Disable librealsense examples")
add_subdirectory(deps/librealsense)

# Add the executable
add_executable(hope_producer src/producer.cpp)

# Add Draco's include directories (if needed)
target_include_directories(hope_producer PRIVATE
    ${CMAKE_BINARY_DIR}
    deps/draco/src
    deps/glm
    deps/OpenSSL/include
    deps/asio/asio/include
    deps/websocketpp
)

target_link_directories(hope_producer PRIVATE deps/OpenSSL/lib)

if (WIN32)
    target_link_libraries(hope_producer PRIVATE draco libcrypto libssl realsense2)
else()
    target_link_libraries(hope_producer PRIVATE draco_static crypto ssl realsense2)
endif()

if (WIN32)
    add_custom_command(TARGET hope_producer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/deps/OpenSSL/bin/libcrypto-3-x64.dll"
            "$<TARGET_FILE_DIR:hope_producer>/libcrypto-3-x64.dll"
    )
    add_custom_command(TARGET hope_producer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/deps/OpenSSL/bin/libssl-3-x64.dll"
            "$<TARGET_FILE_DIR:hope_producer>/libssl-3-x64.dll"
    )
    add_custom_command(TARGET hope_producer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/deps/draco/$<CONFIG>/draco.dll"
            "$<TARGET_FILE_DIR:hope_producer>/draco.dll"
    )
endif()

add_custom_command(TARGET hope_producer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/rsrc/certs/server.key"
        "${CMAKE_BINARY_DIR}/server.key"
)
add_custom_command(TARGET hope_producer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/rsrc/certs/server.crt"
        "${CMAKE_BINARY_DIR}/server.crt"
)
