cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(hope_producer VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate the CUDA toolkit (headers + libraries)
find_package(CUDAToolkit REQUIRED)
set(CUDNN_INCLUDE_DIR "D:/Util/Cuda SDK/include"
    CACHE PATH "Path to cudnn.h")
set(CUDNN_LIBRARY     "D:/Util/Cuda SDK/lib/x64/cudnn.lib"
    CACHE FILEPATH "Path to cudnn.lib")

# (Optional) print out where it was found
message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")

# Manually locate cuDNN in your CUDA SDK install
set(CUDA_SDK_ROOT "D:/Util/Cuda SDK" CACHE PATH "Root of CUDA Toolkit + cuDNN")
find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS "${CUDA_SDK_ROOT}/include"
)
find_library(CUDNN_LIBRARY
    NAMES cudnn cudnn64_9
    HINTS "${CUDA_SDK_ROOT}/lib/x64"
)
if (NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "cuDNN not found. Make sure:\n"
        " - cudnn.h is in ${CUDA_SDK_ROOT}/include\n"
        " - cudnn.lib or cudnn64_8.lib is in ${CUDA_SDK_ROOT}/lib/x64")
endif()

find_library(CUDART_LIBRARY
  NAMES cudart cudart_static
  HINTS "${CUDA_SDK_ROOT}/lib/x64"
)
if (NOT CUDART_LIBRARY)
  message(FATAL_ERROR "Could not find cudart.lib in ${CUDA_SDK_ROOT}/lib/x64")
endif()
message(STATUS "Found CUDA runtime library: ${CUDART_LIBRARY}")
message(STATUS "Found cuDNN include: ${CUDNN_INCLUDE_DIR}")
message(STATUS "Found cuDNN library: ${CUDNN_LIBRARY}")

# Add Draco subdirectory
add_subdirectory(deps/draco)

# Add Realsense subdirectory

set(BUILD_EXAMPLES OFF CACHE BOOL "Disable librealsense examples")
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "Disable librealsense examples")
add_subdirectory(deps/librealsense)

# Add OpenCV subdirectory

set(OPENCV_EXTRA_MODULES_PATH
    "${CMAKE_SOURCE_DIR}/deps/opencv_contrib/modules"
    CACHE STRING "Path to opencv_contrib modules" FORCE)

set(BUILD_SHARED_LIBS   ON  CACHE BOOL "Build OpenCV as shared libraries" FORCE)

set(BUILD_LIST          core imgproc highgui dnn cudaimgproc cudawarping cudev CACHE STRING "" FORCE)
set(BUILD_EXAMPLES    OFF CACHE BOOL "No OpenCV examples")
set(BUILD_DOCS        OFF CACHE BOOL "No OpenCV docs")
set(BUILD_TESTS       OFF CACHE BOOL "No OpenCV tests")
set(BUILD_PERF_TESTS  OFF CACHE BOOL "No OpenCV perf tests")
set(WITH_IPP    OFF CACHE BOOL "Disable Intel IPP support" FORCE)
set(WITH_IPP_IW OFF CACHE BOOL "Disable IPP IW optimizations" FORCE)
set(WITH_HAL    OFF CACHE BOOL "Disable the OpenCV HAL module"   FORCE)



# Enable CUDA and the DNN CUDA backend
set(WITH_CUDA           ON  CACHE BOOL "Build with CUDA" FORCE)
set(WITH_CUBLAS         ON  CACHE BOOL "Use cuBLAS" FORCE)
set(WITH_CUFFT          ON  CACHE BOOL "Use cuFFT" FORCE)
set(WITH_NVCUVID        ON  CACHE BOOL "Use NVIDIA video decoder" FORCE)
set(WITH_CUDNN          ON  CACHE BOOL "Use cuDNN" FORCE)
set(OPENCV_DNN_CUDA     ON  CACHE BOOL "Enable DNN CUDA backend" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "" FORCE)

add_subdirectory(deps/opencv)

set(OpenCV_DIR "deps/opencv")
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)

# Add the executable
add_executable(hope_producer
    src/Source/producer.cpp
    src/Source/Detector.cpp
    src/Headers/Detector.h
    src/Headers/Structs.h
)

# URL locations for model and classes
set(MODEL_URL   "https://huggingface.co/SpotLab/YOLOv8Detection/resolve/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx")
set(CLASSES_URL "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names")

# Local paths where we expect to find them
set(MODEL_PATH   "${CMAKE_SOURCE_DIR}/models/yolov8n.onnx")
set(CLASSES_PATH "${CMAKE_SOURCE_DIR}/models/coco.names")

# If the files aren’t already there, download them
if (NOT EXISTS "${MODEL_PATH}")
  message(STATUS "Downloading YOLOv8 model to ${MODEL_PATH} …")
  file(DOWNLOAD
    "${MODEL_URL}"
    "${MODEL_PATH}"
    SHOW_PROGRESS
    TLS_VERIFY ON
  )
endif()


if (NOT EXISTS "${CLASSES_PATH}")
  message(STATUS "Downloading COCO names to ${CLASSES_PATH} …")
  file(DOWNLOAD
    "${CLASSES_URL}"
    "${CLASSES_PATH}"
    SHOW_PROGRESS
    TLS_VERIFY ON
  )
endif()

# Add Draco's include directories (if needed)
target_include_directories(hope_producer PRIVATE
    ${CMAKE_BINARY_DIR}
    deps/draco/src
    deps/glm
    deps/OpenSSL/include
    deps/asio/asio/include
    deps/websocketpp
    deps/opencv/include          
    src/Headers
    src/cpp
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
)

target_link_directories(hope_producer PRIVATE deps/OpenSSL/lib)

if (WIN32)
    target_link_libraries(hope_producer PRIVATE
        draco
        libcrypto
        libssl
        realsense2
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_dnn
        ${CUDART_LIBRARY}
        ${CUDNN_LIBRARY}
    )
else()
    target_link_libraries(hope_producer PRIVATE
        draco_static
        crypto
        ssl
        realsense2
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_dnn
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
    
    add_custom_command(TARGET hope_producer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/$<CONFIG>/realsense2d.dll"
            "$<TARGET_FILE_DIR:hope_producer>/realsense2d.dll"
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
