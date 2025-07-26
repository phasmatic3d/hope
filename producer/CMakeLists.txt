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
find_package(Python 3.11 EXACT COMPONENTS Interpreter Development)

message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${Python_LIBRARIES}")



#####################################################################
# Fetch and add Draco (encoder-decoder for 3D data)
#####################################################################

set(FETCHCONTENT_QUIET OFF)

#####################################################################
# Fetch and add nanobind (C++ ↔ Python binding library)
#####################################################################
include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.7.0     # or whatever release you want
)
FetchContent_MakeAvailable(nanobind)

add_subdirectory(${CMAKE_BINARY_DIR}/_deps/nanobind-src)

set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries" FORCE)

#####################################################################
# Fetch and add draco 
#####################################################################

FetchContent_Declare(
  draco
  GIT_REPOSITORY https://github.com/google/draco.git
  GIT_TAG        1.5.7
)

set(DRACO_POINT_CLOUD_COMPRESSION   ON CACHE BOOL "Enable point‐cloud compression."         FORCE)
set(DRACO_MESH_COMPRESSION          OFF  CACHE BOOL "Enable Mesh compression"                FORCE)


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

#####################################################################
# Bindings for encoder
#####################################################################

set(ENCODER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src")

nanobind_add_module(encoder
    src/bindings/draco_bindings.cpp
)

if (WIN32)
	set(ENCODER_SUFFIX ".pyd")
else()
  	set(ENCODER_SUFFIX ".so")
endif()

set_target_properties(encoder PROPERTIES
  	PREFIX ""
  	SUFFIX "${ENCODER_SUFFIX}"
  	RUNTIME_OUTPUT_DIRECTORY        "${ENCODER_OUTPUT_DIR}"
  	LIBRARY_OUTPUT_DIRECTORY        "${ENCODER_OUTPUT_DIR}"
  	ARCHIVE_OUTPUT_DIRECTORY        "${ENCODER_OUTPUT_DIR}"
)

target_include_directories(encoder PRIVATE
  	${draco_SOURCE_DIR}/src      
  	${CMAKE_BINARY_DIR}/draco      
  	${CMAKE_BINARY_DIR}        
)

target_link_libraries(encoder PRIVATE
  	draco::draco
)

message(STATUS "Draco source dir: ${draco_SOURCE_DIR}")
message(STATUS "Draco build dir:  ${draco_BINARY_DIR}")
message(STATUS "Cmake build dir:  ${CMAKE_BINARY_DIR}")
message(STATUS "Cmake source dir:  ${CMAKE_SOURCE_DIR}")

if (WIN32)
	add_custom_command(TARGET encoder POST_BUILD
	  COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    "$<TARGET_FILE:draco::draco>"
	    "${ENCODER_OUTPUT_DIR}/draco.dll"
	)
endif()

#####################################################################
# Bindings for broadcaster
#####################################################################

# Add the executable for the broadcaster
nanobind_add_module(broadcaster
    src/bindings/server_bindings.cpp
)

set(BROADCASTER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src")

if (WIN32)
	set(BROADCASTER_SUFFIX ".pyd")
else()
  	set(BROADCASTER_SUFFIX ".so")
endif()


set_target_properties(broadcaster PROPERTIES
  	PREFIX ""           
  	SUFFIX "${BROADCASTER_SUFFIX}"    
  	RUNTIME_OUTPUT_DIRECTORY        "${BROADCASTER_OUTPUT_DIR}"
  	LIBRARY_OUTPUT_DIRECTORY        "${BROADCASTER_OUTPUT_DIR}"
  	ARCHIVE_OUTPUT_DIRECTORY        "${BROADCASTER_OUTPUT_DIR}"
)

target_include_directories(broadcaster PRIVATE
  	${asio_SOURCE_DIR}/asio/include     
  	${websocketpp_SOURCE_DIR}       
)

target_compile_definitions(broadcaster PRIVATE
  	ASIO_STANDALONE
  	_WEBSOCKETPP_CPP11_TYPE_TRAITS_
)

#####################################################################
# Certificates
#####################################################################

#add_custom_command(TARGET encoder POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${CMAKE_SOURCE_DIR}/rsrc/certs/server.key"
#        "${CMAKE_BINARY_DIR}/server.key"
#)
#add_custom_command(TARGET encoder POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${CMAKE_SOURCE_DIR}/rsrc/certs/server.crt"
#        "${CMAKE_BINARY_DIR}/server.crt"
#)