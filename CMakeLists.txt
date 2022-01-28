cmake_minimum_required(VERSION 3.0.0)
project(tflitec_app)

# include packages
find_package(fmt CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

set(VENDOR_DIR ${CMAKE_SOURCE_DIR}/vendor)

set(tflitec_DIR ${VENDOR_DIR}/tflite_c)
set(imgui_docking_DIR ${VENDOR_DIR}/imgui_docking)
find_package(tflitec CONFIG REQUIRED)
find_package(imgui_docking CONFIG REQUIRED)

# define our target
set(CMAKE_CXX_STANDARD 17)

set(SRC_FILES
    src/main.cpp 
    src/model.cpp src/missing.c 
    src/App.cpp src/gui.cpp src/SoccerPlayer.cpp src/SoccerPlayerController.cpp
    vendor/util/MSS.cpp    
    vendor/util/KeyListener.cpp
    vendor/util/AutoGui.cpp)

add_executable(main ${SRC_FILES})
include_directories(main ${VENDOR_DIR})
target_link_libraries(main PRIVATE 
    tflitec imgui_docking 
    fmt::fmt
    spdlog::spdlog spdlog::spdlog_header_only
    "d3d11.lib" "dxgi.lib" "d3dcompiler.lib") 

# vcpkg.cmake has internal stuff that autogenerates this
# we have to do this manually
add_custom_command(TARGET main 
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
    ${tflitec_DIR}/bin/tensorflowlite_c.dll
    $<TARGET_FILE_DIR:main>)