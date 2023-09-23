#pragma once

#include <stddef.h>
#include <vector>
#include "IModel.h"
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"

class TensorflowLiteModel: public IModel
{
private:
    TfLiteModel *m_model;
    TfLiteInterpreterOptions *m_options;
    TfLiteInterpreter *m_interp;

    std::vector<RGB<float>> m_input_buffer;
    size_t m_width;
    size_t m_height;
    size_t m_channels;
    size_t m_num_pixels;

    Prediction m_result;
public:
    // num_threads <= 0 then use hardware concurrency amount
    TensorflowLiteModel(const char *filepath, uint32_t num_threads=0);
    ~TensorflowLiteModel() override;
    InputBuffer GetInputBuffer() override {
        return InputBuffer {
            m_input_buffer.data(), 
            m_width,
            m_height
        };
    }
    void Parse() override;
    Prediction GetPrediction() override { return m_result; };
    void PrintSummary() override;
};