#pragma once

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/common.h"
#include "Prediction.h"

void LoadTfLiteModel(const char *filepath);

template <typename T>
struct RGBA {
    T r, g, b, a;
};

template <typename T>
struct RGB {
    T r, g, b;
};

// all of our models use floats for input/output
// all of our models have fixed sized inputs and outputs
// all of our models have single input and single output (tensor)
class Model
{
public:
    struct Vec2D {
        int x;
        int y;
    };
private:
    TfLiteModel *m_model;
    TfLiteInterpreterOptions *m_options;
    TfLiteInterpreter *m_interp;

    RGB<float> *m_input_buffer;
    Prediction m_result;

    int m_width;
    int m_height;
    int m_num_pixels;
    int m_channels;
public:
    Model(const char *filepath);
    ~Model();
    // need to do a mapping to uint8_t to float
    bool CopyDataToInput(const uint8_t *data, const int width, const int height);
    void Parse();
    inline RGB<float> *GetInputBuffer() { return m_input_buffer; }
    inline Vec2D GetInputSize() const { return {m_width, m_height}; }
    inline Prediction GetResult() const { return m_result; };
    void Print();
    void RunBenchmark(const int n=100);
};