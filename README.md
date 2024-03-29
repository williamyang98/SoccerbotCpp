[![x86-windows](https://github.com/williamyang98/SoccerbotCpp/actions/workflows/x86-windows.yml/badge.svg)](https://github.com/williamyang98/SoccerbotCpp/actions/workflows/x86-windows.yml)

# Introduction
A bot for playing facebook soccer 
- Uses windows api for mouse movement and screen grabbing
- Uses tensorflowlite c api for loading and parsing quantized model
- Refer to https://github.com/williamyang98/SoccerBot for original python implementation

# Preview
https://github.com/williamyang98/SoccerbotCpp/assets/21079869/9c4b8c97-8eb6-4dfd-8b0c-ecee0e84505d

# Build instructions
1. ```Setup MSVC C++ developer environment with clang, Ninja, and vcpkg```
2. ```CC=clang CXX=clang++ ./scripts/toolchains/cmake_configure.sh```
3. ```ninja -C build```

# Run instructions
| Command | Description |
| --- | --- |
| ```./soccerbot``` | Run with default settings |
| ```./soccerbot --model ./models/*.tflite --runtime tflite``` | Run tflite mode on CPU |
| ```./soccerbot --model ./models/*.onnx --runtime onnx --onnx-device cpu``` | Run onnx model on CPU |
| ```./soccerbot --model ./models/*.onnx --runtime onnx --onnx-device directml``` | Run onnx model on GPU using DirectML |

# Training and emulator
Refer to ```scripts/README.md``` for instructions to train models and run emulator.
