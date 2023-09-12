# Instructions
Requires python 3.9. This can be downloaded from [here](https://www.python.org/downloads/release/python-390/).

## Setup virtual environment
1. ```py -3.9 -m venv venv```
2. ```source ./venv/Scripts/activate```

## Emulator
1. ```cd emulator```
2. ```pip install -r requirements.txt```
3. ```python run_emulator.py```

## Training
1. ```cd training```
2. ```pip install -r requirements.txt```
3. ```python run_training.py```
4. ```python run_quantize.py```
5. Copy ```*.tflite``` model over to desired location.
