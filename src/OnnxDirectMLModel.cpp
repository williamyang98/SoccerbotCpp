#include "OnnxDirectMLModel.h"

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <cpu_provider_factory.h>

#include <cstdlib>
#include <memory>
#include <fmt/core.h>
#include <stdexcept>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>

const char* onnx_data_type_to_str(ONNXTensorElementDataType type);

std::basic_string<wchar_t> create_wchar_string(const char* src) {
    const size_t length = strlen(src)+1;
    auto dest = std::vector<wchar_t>(length);
    size_t total_written = 0;
    mbstowcs_s(&total_written, dest.data(), length, src, length);
    auto res = std::basic_string<wchar_t>(dest.data(), length);
    return res;
}

OnnxDirectMLModel::OnnxDirectMLModel(const char* filepath, OnnxDirectMLModel::GPU_Options opts) 
: m_ort_api(Ort::GetApi())
{
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx-directml-gpu");
    m_session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    ORT_ABORT_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_DML(m_session_options, opts.device_id));
    InitModel(filepath);
}

OnnxDirectMLModel::OnnxDirectMLModel(const char* filepath, OnnxDirectMLModel::CPU_Options opts)
: m_ort_api(Ort::GetApi())
{
    m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnx-cpu");
    if (opts.total_threads != 0) {
        m_session_options.SetIntraOpNumThreads(opts.total_threads);
    }
    if (opts.is_sequential) {
        m_session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    } else {
        m_session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    }
    InitModel(filepath);
}

OnnxDirectMLModel::~OnnxDirectMLModel() {}

void OnnxDirectMLModel::InitModel(const char* filepath) {
    auto w_filepath = create_wchar_string(filepath);

    m_session = std::make_unique<Ort::Session>(*m_env.get(), w_filepath.c_str(), m_session_options);

    if (m_session->GetInputCount() != 1) {
        throw std::runtime_error(fmt::format(
            "Model expected at 1 input tensor, got {}", m_session->GetInputCount()
        ));
    }

    if (m_session->GetOutputCount() != 1) {
        throw std::runtime_error(fmt::format(
            "Model expected at 1 output tensor, got {}", m_session->GetOutputCount()
        ));
    }

    const auto& input_info = m_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto& input_shape = input_info.GetShape();
    const auto input_type = input_info.GetElementType();
    if (input_shape.size() != 4) {
        throw std::runtime_error(fmt::format(
            "Model expected input tensor of dimension 4, got {}. (1,H,W,C)",
            input_shape.size()
        ));
    }

    const auto& output_info = m_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto& output_shape = output_info.GetShape();
    const auto output_type = output_info.GetElementType();
    if (output_shape.size() != 2) {
        throw std::runtime_error(fmt::format(
            "Model expected output tensor of dimension 2, got {}. (1,3)",
            output_shape.size()
        ));
    }
    
    m_height = size_t(input_shape[1]);
    m_width = size_t(input_shape[2]);
    m_channels = size_t(input_shape[3]);
    m_num_pixels = m_height * m_width;
    if (m_channels != 3) {
        throw std::runtime_error(fmt::format(
            "Model expected model with 3 channels, got ({},{},{})", 
            m_width, m_height, m_channels));
    }

    const size_t output_size = output_shape[1];
    if (output_size != 3) {
        throw std::runtime_error(fmt::format("Model expected 3 outputs (got {})", output_size));
    }
    
    // Allocate buffer and associate it with input tensor
    m_input_buffer.resize(m_num_pixels);
    m_input_shape[0] = 1;
    m_input_shape[1] = input_shape[1];
    m_input_shape[2] = input_shape[2]; 
    m_input_shape[3] = input_shape[3]; 

    auto input_mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, 
        OrtMemType::OrtMemTypeCPUInput
    );
    auto input_tensor = Ort::Value::CreateTensor<float>(
        input_mem_info, 
        reinterpret_cast<float*>(m_input_buffer.data()), m_num_pixels*m_channels,
        m_input_shape, 4
    );
    m_input_tensors.emplace_back(std::move(input_tensor));
    
    // Need to keep track of tensor labels for inference
    auto input_name = m_session->GetInputNameAllocated(0, m_allocator);
    auto output_name = m_session->GetOutputNameAllocated(0, m_allocator);
    m_input_name = std::string(input_name.get()); 
    m_output_name = std::string(output_name.get()); 
}

void OnnxDirectMLModel::Parse() {
    const char* input_names[1] = { m_input_name.c_str() };
    const char* output_names[1] = { m_output_name.c_str() };

    auto run_options = Ort::RunOptions { nullptr };
    auto output_tensors = m_session->Run(
        run_options, 
        input_names, m_input_tensors.data(), 1,
        output_names, 1
    );
    
    const auto& output_tensor = output_tensors[0];
    const float* output_data = output_tensor.GetTensorData<float>();
    m_prediction.x = output_data[0];
    m_prediction.y = output_data[1];
    m_prediction.confidence = output_data[2];
}

void OnnxDirectMLModel::PrintSummary() {
    const size_t total_inputs = m_session->GetInputCount();
    printf("[inputs: %zu]\n", total_inputs);
    for (size_t i = 0; i < total_inputs; i++) {
        const auto& name = m_session->GetInputNameAllocated(i, m_allocator);
        const auto& type_info = m_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        const auto type = type_info.GetElementType(); 
        const auto& shape = type_info.GetShape();
        printf("    %s: %s (", name.get(), onnx_data_type_to_str(type));
        for (size_t j = 0; j < shape.size(); j++) {
            printf("%" PRIi64, shape[j]);
            if (j != (shape.size()-1)) printf(",");
        }
        printf(")\n");
    }

    const size_t total_outputs = m_session->GetOutputCount();
    printf("[outputs: %zu]\n", total_outputs);
    for (size_t i = 0; i < total_outputs; i++) {
        const auto& name = m_session->GetOutputNameAllocated(i, m_allocator);
        const auto& type_info = m_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();
        const auto type = type_info.GetElementType(); 
        const auto& shape = type_info.GetShape();
        printf("    %s: %s (", name.get(), onnx_data_type_to_str(type));
        for (size_t j = 0; j < shape.size(); j++) {
            printf("%" PRIi64, shape[j]);
            if (j != (shape.size()-1)) printf(",");
        }
        printf(")\n");
    }

    auto providers = Ort::GetAvailableProviders();
    printf("[execution providers: %zu]\n", providers.size());
    for (const auto& provider: providers) {
        printf("    %*s\n", int(provider.length()), provider.c_str());
    }
}

void OnnxDirectMLModel::ORT_ABORT_ON_ERROR(OrtStatus* status) {
    if (status == nullptr) {
        return; 
    }

    const char* message = m_ort_api.GetErrorMessage(status);
    fprintf(stderr, "ORT_ABORT_ON_ERROR: %s\n", message);
    m_ort_api.ReleaseStatus(status);
    exit(1);
}

const char* onnx_data_type_to_str(ONNXTensorElementDataType type) {
    switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:      return "UNDEFINED";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:          return "FLOAT";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:          return "UINT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:           return "INT8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:         return "UINT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:          return "INT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:          return "INT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:          return "INT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:         return "STRING";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:           return "BOOL";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:        return "FLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:         return "DOUBLE";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:         return "UINT32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:         return "UINT64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:      return "COMPLEX64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:     return "COMPLEX128";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:       return "BFLOAT16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:   return "FLOAT8E4M3FN";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ: return "FLOAT8E4M3FNUZ";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:     return "FLOAT8E5M2";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ: return "FLOAT8E5M2FNUZ";
    default:
        return "Unknown type";
    }
}
