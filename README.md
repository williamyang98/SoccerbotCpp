# Introduction
A bot for playing facebook soccer 
- Uses windows api for mouse movement and screen grabbing
- Uses tensorflowlite c api for loading and parsing quantized model
- Refer to https://github.com/FiendChain/SoccerBot for original python implementation

# Preview
![Main window](docs/screenshot_v1.png)

# Build instructions
1. ```Setup MSVC C++ developer environment with clang, Ninja, and vcpkg```
2. ```CC=clang CXX=clang++ ./cmake_configure.sh```
3. ```ninja -C build```
4. ```./build/main.exe```

# Training and emulator
Refer to ```scripts/README.md``` for instructions to train models and run emulator.