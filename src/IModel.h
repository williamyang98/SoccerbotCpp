#pragma once
#include <stddef.h>
#include "Prediction.h"

template <typename T>
struct RGBA {
    T r, g, b, a;
};

template <typename T>
struct RGB {
    T r, g, b;
};

struct InputBuffer {
    RGB<float>* data;
    size_t width;
    size_t height;
};

class IModel
{
public:
    IModel() {}
    virtual ~IModel() {}
    virtual InputBuffer GetInputBuffer() = 0;
    virtual void Parse() = 0;
    virtual Prediction GetPrediction() = 0;
    virtual void PrintSummary() = 0;
};
