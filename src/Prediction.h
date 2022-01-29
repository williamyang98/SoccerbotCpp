#pragma once

struct Prediction {
    float x;
    float y;
    float confidence;

    Prediction(float _x=0, float _y=0, float _c=0) {
        x = _x;
        y = _y;
        confidence = _c;
    }
};