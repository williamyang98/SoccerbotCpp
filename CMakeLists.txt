cmake_minimum_required(VERSION 3.0.0)
project(tflitec_app)

# include packages
# find_package(fmt CONFIG REQUIRED)
# find_package(spdlog CONFIG REQUIRED)

set(VENDOR_DIR ${CMAKE_SOURCE_DIR}/vendor)

add_subdirectory(${VENDOR_DIR}/fmt)

set(tflitec_DIR ${VENDOR_DIR}/tflite_c)
set(imgui_docking_DIR ${VENDOR_DIR}/imgui_docking)
find_package(tflitec CONFIG REQUIRED)
find_package(imgui_docking CONFIG REQUIRED)

# define our target
set(CMAKE_CXX_STANDARD 17)

# enable address sanitizer for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /fsanitize=address")

set(SRC_FILES
    src/main.cpp 
    # neural network
    src/model.cpp
    # soccer logic
    src/App.cpp src/gui.cpp 
    src/SoccerPlayer.cpp 
    src/SoccerPlayerController.cpp 
    src/Predictor.cpp
    # utility
    vendor/util/MSS.cpp    
    vendor/util/KeyListener.cpp
    vendor/util/AutoGui.cpp)

add_executable(main ${SRC_FILES})
include_directories(main ${VENDOR_DIR})
target_link_libraries(main PRIVATE 
    tflitec imgui_docking 
    fmt::fmt
    "d3d11.lib" "dxgi.lib" "d3dcompiler.lib") 
target_compile_options(main PRIVATE "/MP")

# vcpkg.cmake has internal stuff that autogenerates this
# we have to do this manually
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    add_custom_command(TARGET main 
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
        ${tflitec_DIR}/bin/x64/Release/tensorflowlite_c.dll
        $<TARGET_FILE_DIR:main>)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_custom_command(TARGET main 
        POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy 
        ${tflitec_DIR}/bin/x86/Release/tensorflowlite_c.dll
        $<TARGET_FILE_DIR:main>)
endif()
