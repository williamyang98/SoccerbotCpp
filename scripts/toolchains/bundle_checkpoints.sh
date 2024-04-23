#!/bin/sh
pytorch_in_dir="./scripts/training-pytorch/data"
tflite_in_dir="./scripts/training-tensorflow/data"
pytorch_out_dir="./checkpoints/pytorch"
tflite_out_dir="./checkpoints/tflite"

echo "Copying pytorch checkpoints"
rm -rf ${pytorch_out_dir}
mkdir -p ${pytorch_out_dir}
cp ${pytorch_in_dir}/*.pt ${pytorch_out_dir}
cp ${pytorch_in_dir}/*.onnx ${pytorch_out_dir}

echo "Copying tensorflow checkpoints"
rm -rf ${tflite_out_dir}
mkdir -p ${tflite_out_dir}
cp ${tflite_in_dir}/checkpoint-* ${tflite_out_dir}
cp ${tflite_in_dir}/*.onnx ${tflite_out_dir}
cp ${tflite_in_dir}/*.tflite ${tflite_out_dir}
