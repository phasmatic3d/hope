cmake_minimum_required(VERSION 3.10)
# Tell CMake to let option() respect any pre-set CACHE or normal variables.
cmake_policy(SET CMP0077 NEW)

# Set the project name and version
project(hope_producer VERSION 1.0)


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenSSL REQUIRED)
find_package(realsense2 REQUIRED) 


#####################################################################
# Fetch and add Draco (encoder-decoder for 3D data)
#####################################################################

set(FETCHCONTENT_QUIET OFF)

include(FetchContent)

FetchContent_Declare(
  draco
  GIT_REPOSITORY https://github.com/google/draco.git
  GIT_TAG        1.5.7
)

set(DRACO_POINT_CLOUD_COMPRESSION   ON CACHE BOOL "Enable point‐cloud compression."         FORCE)
set(DRACO_MESH_COMPRESSION          ON  CACHE BOOL "Enable Mesh compression"                FORCE)


set(DRACO_STANDARD_EDGEBREAKER      ON  CACHE BOOL "" FORCE)
set(DRACO_PREDICTIVE_EDGEBREAKER    OFF CACHE BOOL "" FORCE)

set(DRACO_TRANSCODER_SUPPORTED      OFF CACHE BOOL "Disable the GLTF transcoder tool."    FORCE)
set(DRACO_JS_GLUE                   OFF CACHE BOOL "Disable JS/Emscripten targets."       FORCE)
set(DRACO_WASM                      OFF CACHE BOOL "Disable WASM support."                FORCE)
set(DRACO_ANIMATION_ENCODING        OFF CACHE BOOL "Disable animation encoding."          FORCE)
set(DRACO_GLTF_BITSTREAM            OFF CACHE BOOL "Disable GLTF bitstream–only mode."    FORCE)
set(DRACO_UNITY_PLUGIN              OFF CACHE BOOL "Disable Unity plugin."                FORCE)
set(DRACO_MAYA_PLUGIN               OFF CACHE BOOL "Disable Maya plugin."                 FORCE)
set(DRACO_TESTS                     OFF CACHE BOOL "Disable internal unit‐tests."         FORCE)
set(DRACO_DEBUG_COMPILER_WARNINGS   OFF CACHE BOOL "Turn off extra warnings."             FORCE) 

FetchContent_MakeAvailable(draco)

if (TARGET draco::draco)
    set(DRACO_LIB draco::draco)
else()
    set(DRACO_LIB draco_static)
endif()

#####################################################################
# Fetch and add spdlog (thread-safe logging library)
#####################################################################
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.11.0
)

FetchContent_GetProperties(spdlog)
if(NOT spdlog_POPULATED)
  FetchContent_Populate(spdlog)
  add_subdirectory(${spdlog_SOURCE_DIR} ${spdlog_BINARY_DIR})
endif()

#####################################################################
# Fetch and add WebSocket++(header only lib that implements RFC6455)
#####################################################################

FetchContent_Declare(
  websocketpp
  GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
  GIT_TAG        0.8.2
)

set(WEBSOCKETPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WEBSOCKETPP_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(websocketpp)

#####################################################################
# Fetch and add ASIO(used from websocketspp)
#####################################################################
FetchContent_Declare(
  asio
  GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
  GIT_TAG        asio-1-28-0
)

FetchContent_MakeAvailable(asio)

#####################################################################
# Fetch and add GLM (OpenGL Mathematics)
#####################################################################
FetchContent_Declare(
  glm
  GIT_REPOSITORY https://github.com/g-truc/glm.git
  GIT_TAG        1.0.1
)

FetchContent_MakeAvailable(glm)

# Add the executable
add_executable(hope_producer src/producer.cpp)

target_include_directories(hope_producer PRIVATE
  ${draco_SOURCE_DIR}/src
  ${CMAKE_BINARY_DIR}              # <-- so "draco/draco_features.h" is found
  ${spdlog_SOURCE_DIR}/include
  ${websocketpp_SOURCE_DIR}
  ${asio_SOURCE_DIR}/asio/include
  ${glm_SOURCE_DIR}
)


# Add Realsense subdirectory
#set(BUILD_EXAMPLES OFF CACHE BOOL "Disable librealsense examples")
#set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "Disable librealsense examples")
#add_subdirectory(deps/librealsense)
#
#
#
#target_link_directories(hope_producer PRIVATE deps/OpenSSL/lib)
#

if (WIN32)
    target_link_libraries(hope_producer PRIVATE draco libcrypto libssl realsense2)
else()
    target_link_libraries(hope_producer PRIVATE 
      ${DRACO_LIB}
      OpenSSL::Crypto 
      OpenSSL::SSL 
      realsense2::realsense2
    )
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