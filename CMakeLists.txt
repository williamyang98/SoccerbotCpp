cmake_minimum_required(VERSION 3.0.0)
project(tflitec_app)

# add packages
set(VENDOR_DIR ${CMAKE_SOURCE_DIR}/vendor)
add_subdirectory(${VENDOR_DIR}/fmt)
add_subdirectory(${VENDOR_DIR}/onnxruntime-directml)
set(ARGPARSE_INSTALL OFF CACHE BOOL "Include an install target" FORCE)
add_subdirectory(${VENDOR_DIR}/argparse)
set(tflitec_DIR ${VENDOR_DIR}/tflite_c)
set(imgui_docking_DIR ${VENDOR_DIR}/imgui_docking)
find_package(tflitec CONFIG REQUIRED)
find_package(imgui_docking CONFIG REQUIRED)

# enable address sanitizer for debug builds in MSVC compiler
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /fsanitize=address")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /fsanitize=address")
    add_compile_options(/MP)
endif()

add_executable(main 
    ${CMAKE_SOURCE_DIR}/src/main.cpp 
    ${CMAKE_SOURCE_DIR}/src/gui.cpp 
    # neural network
    ${CMAKE_SOURCE_DIR}/src/TensorflowLiteModel.cpp
    ${CMAKE_SOURCE_DIR}/src/OnnxDirectMLModel.cpp
    # soccer logic
    ${CMAKE_SOURCE_DIR}/src/App.cpp 
    ${CMAKE_SOURCE_DIR}/src/SoccerPlayer.cpp 
    ${CMAKE_SOURCE_DIR}/src/SoccerPlayerController.cpp 
    ${CMAKE_SOURCE_DIR}/src/Predictor.cpp
    # utility
    ${VENDOR_DIR}/util/MSS.cpp    
    ${VENDOR_DIR}/util/KeyListener.cpp
    ${VENDOR_DIR}/util/AutoGui.cpp)

set_target_properties(main PROPERTIES CXX_STANDARD 17)
target_include_directories(main PRIVATE ${VENDOR_DIR})
target_link_libraries(main PRIVATE 
    tflitec onnxruntime
    argparse::argparse fmt::fmt
    imgui_docking 
    "d3d11.lib" "dxgi.lib" "d3dcompiler.lib") 

# install dlls for tensorflow-lite and onnxruntime-directml
add_custom_command(
    TARGET main 
    POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy 
    $<TARGET_RUNTIME_DLLS:main> 
    $<TARGET_FILE_DIR:main>
    COMMAND_EXPAND_LISTS
)
