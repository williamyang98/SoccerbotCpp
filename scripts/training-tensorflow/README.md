# Instructions
Requires python 3.9. This can be downloaded from [here](https://www.python.org/downloads/release/python-390/).

1. ```py -3.9 -m venv venv```
2. ```source ./venv/Scripts/activate```
3. ```pip install -r requirements.txt```
4. ```python run_training.py --model-type [model_type]```

## Create onnx or tflite model
1. ```python run_create_onnx.py --model-type [model_type]```
2. ```python run_create_tflite.py --model-type [model_type]```
3. Copy ```*.tflite``` or ```*.onnx``` model over to desired location.
