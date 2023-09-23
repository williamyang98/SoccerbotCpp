cmake_minimum_required(VERSION 3.0.0)
project(onnxruntime-directml)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
    IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/bin/onnxruntime.lib"
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/bin/onnxruntime.dll")

