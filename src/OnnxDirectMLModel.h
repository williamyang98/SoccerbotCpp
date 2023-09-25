#pragma once

#include <stddef.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include "IModel.h"
#include "Prediction.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

class OnnxDirectMLModel: public IModel
{
public:
    struct GPU_Options {
        int device_id = 0;
    };
    struct CPU_Options {
        int total_threads = 0; 
        bool is_sequential = false;
    };
private:
    std::unique_ptr<Ort::Env> m_env;
    Ort::SessionOptions m_session_options;
    std::unique_ptr<Ort::Session> m_session;
    Ort::AllocatorWithDefaultOptions m_allocator;
    const OrtApi& m_ort_api;
    
    std::vector<RGB<float>> m_input_buffer;
    size_t m_height;
    size_t m_width;
    size_t m_channels;
    size_t m_num_pixels;
    
    // input/output parameters for execution context
    std::vector<Ort::Value> m_input_tensors;
    int64_t m_input_shape[4];
    std::string m_input_name;
    std::string m_output_name;

    Prediction m_prediction;
public:
    OnnxDirectMLModel(const char* filepath, GPU_Options opts);
    OnnxDirectMLModel(const char* filepath, CPU_Options opts);
    ~OnnxDirectMLModel() override;
    InputBuffer GetInputBuffer() override {
        return InputBuffer {
            m_input_buffer.data(), 
            m_width,
            m_height
        };
    }
    void Parse() override;
    Prediction GetPrediction() override { return m_prediction; }
    void PrintSummary() override;
private:
    void InitModel(const char* filepath);
    void ORT_ABORT_ON_ERROR(OrtStatus* status);
};
