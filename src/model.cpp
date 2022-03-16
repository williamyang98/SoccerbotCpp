#include <stdio.h>
#include <chrono>
#include <thread>
#include <stdexcept>

#include <fmt/core.h>

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

#include "model.h"

static void PrintTfLiteModelSummary(TfLiteInterpreter *interpreter);
static void PrintTfLiteTensorSummary(const TfLiteTensor *tensor);

Model::Model(const char *filepath, uint32_t num_threads)
{
    // load model
    m_model = TfLiteModelCreateFromFile(filepath);
    m_options = TfLiteInterpreterOptionsCreate();
    // default number of threads is same as core count
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    TfLiteInterpreterOptionsSetNumThreads(m_options, num_threads);
    // Create the interpreter.
    m_interp = TfLiteInterpreterCreate(m_model, m_options);
    // Allocate tensors and populate the input tensor data.
    TfLiteInterpreterAllocateTensors(m_interp);

    // verify input size matches
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(m_interp, 0);
    if (input_tensor->dims->size != 4) {
        throw std::runtime_error(fmt::format(
            "Model expected input tensor shape of dimension 4, got {}",
            input_tensor->dims->size
        ));
    }

    size_t input_size = 1; 
    {
        for (int i = 0; i < input_tensor->dims->size; i++) {
            input_size *= input_tensor->dims->data[i];
        }
    }

    // format of input tensor shape is: (1,height,width,channels)
    m_height = input_tensor->dims->data[1];
    m_width = input_tensor->dims->data[2];
    m_channels = input_tensor->dims->data[3];
    m_num_pixels = m_height * m_width;

    if (m_channels != 3) {
        throw std::runtime_error(fmt::format(
            "Model expected model with 3 channels, got ({},{},{})", 
            m_width, m_height, m_channels));
    }

    // verify output size matches
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(m_interp, 0);
    size_t output_size = 1; 
    {
        for (int i = 0; i < output_tensor->dims->size; i++) {
            output_size *= output_tensor->dims->data[i];
        }
    }
    if (output_size != 3) {
        throw std::runtime_error(fmt::format("Model expected 3 outputs (got {})", output_size));
    }

    // allocate buffer after all checks completed
    m_input_buffer = (RGB<float> *)malloc(sizeof(RGB<float>) * m_num_pixels);
}

void Model::Print() {
    PrintTfLiteModelSummary(m_interp);
}

bool Model::CopyDataToInput(const uint8_t *data, const int width, const int height) {
    // make sure image size matches
    if ((width != m_width) || (height != m_height)) {
        return false;
    }


    // copy the uint8_t image data to the buffer
    const RGBA<uint8_t> *rgba_data = reinterpret_cast<const RGBA<uint8_t>*>(data);
    for (size_t i = 0; i < m_num_pixels; i++) {
        size_t j = m_num_pixels-i-1;
        m_input_buffer[i].r = ((float)rgba_data[j].b) / 255.0f;
        m_input_buffer[i].g = ((float)rgba_data[j].g) / 255.0f;
        m_input_buffer[i].b = ((float)rgba_data[j].r) / 255.0f;
    }
    return true;
}

void Model::Parse() {
    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(m_interp, 0);
    const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(m_interp, 0);
    // Copy and run
    const int num_channels = 3;
    TfLiteTensorCopyFromBuffer(input_tensor, m_input_buffer, m_num_pixels*m_channels*sizeof(float));
    TfLiteInterpreterInvoke(m_interp);
    // Extract the output tensor data.
    TfLiteTensorCopyToBuffer(
        output_tensor, 
        &m_result, sizeof(m_result));
}

void Model::RunBenchmark(const int n) {
    // Time the average duration of an invoke
    float ms_avg_invoke = 0.0f;
    for (int i = 0; i < n; i++) {
        auto timer_start = std::chrono::high_resolution_clock::now();
        TfLiteInterpreterInvoke(m_interp);
        auto timer_end = std::chrono::high_resolution_clock::now();
        long long ms_start = std::chrono::time_point_cast<std::chrono::milliseconds>(timer_start).time_since_epoch().count();
        long long ms_end = std::chrono::time_point_cast<std::chrono::milliseconds>(timer_end).time_since_epoch().count();
        long long ms_dt = ms_end-ms_start;
        ms_avg_invoke += (float)(ms_dt);
    }
    ms_avg_invoke /= (float)(n);
    printf("%.2fms per invoke (N=%d)\n", ms_avg_invoke, n);
}

Model::~Model() {
    // Dispose of the model and interpreter objects.
    TfLiteInterpreterDelete(m_interp);
    TfLiteInterpreterOptionsDelete(m_options);
    TfLiteModelDelete(m_model);
    free(m_input_buffer);
}

void PrintTfLiteModelSummary(TfLiteInterpreter *interpreter) {
    int input_tensor_count = TfLiteInterpreterGetInputTensorCount(interpreter);
    int output_tensor_count = TfLiteInterpreterGetOutputTensorCount(interpreter);
    printf("inputs=%d, output=%d\n", input_tensor_count, output_tensor_count);

    for (int i = 0; i < input_tensor_count; i++) {
        const TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, i);
        printf("inp_tensor[%d]: ", i);
        PrintTfLiteTensorSummary(input_tensor);
    }

    for (int i = 0; i < output_tensor_count; i++) {
        const TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, i);
        printf("out_tensor[%d]: ", i);
        PrintTfLiteTensorSummary(output_tensor);
    }
}

void PrintTfLiteTensorSummary(const TfLiteTensor *tensor) {
    printf("%s (", tensor->name);
    // size of tensor
    for (int j = 0; j < tensor->dims->size; j++) {
        int dim = tensor->dims->data[j];
        if (j != tensor->dims->size-1) {
            printf("%d,", dim);
        } else {
            printf("%d) ", dim);
        }
    }
    // type of tensor
    TfLiteType t = TfLiteTensorType(tensor);
    printf("%s ", TfLiteTypeGetName(t));
    // quantisation?
    if (t == kTfLiteUInt8) {
        TfLiteQuantizationParams qparams = TfLiteTensorQuantizationParams(tensor);
        printf("[scale=%.2f, zero_point=%d]\n", qparams.scale, qparams.zero_point);
    } else {
        printf("\n");
    }
}

