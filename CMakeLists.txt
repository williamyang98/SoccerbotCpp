cmake_minimum_required(VERSION 3.0.0)
project(soccerbot)

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

add_executable(soccerbot 
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

set_target_properties(soccerbot PROPERTIES CXX_STANDARD 17)
target_include_directories(soccerbot PRIVATE ${VENDOR_DIR})
target_link_libraries(soccerbot PRIVATE 
    tflitec onnxruntime
    argparse::argparse fmt::fmt
    imgui_docking 
    "d3d11.lib" "dxgi.lib" "d3dcompiler.lib") 

# install dlls for tensorflow-lite and onnxruntime-directml
add_custom_command(
    TARGET soccerbot 
    POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_RUNTIME_DLLS:soccerbot> 
    $<TARGET_FILE_DIR:soccerbot>
    COMMAND_EXPAND_LISTS
)

# NOTE: Libraries that use onnxruntime must copy this dll 
#       There isn't a good way in cmake to add this dependency 
#       https://gitlab.kitware.com/cmake/cmake/-/issues/22993
add_custom_command(
    TARGET soccerbot 
    POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${VENDOR_DIR}/onnxruntime-directml/bin/DirectML.dll"
    $<TARGET_FILE_DIR:soccerbot>
)
