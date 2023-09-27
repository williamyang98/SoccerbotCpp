#pragma once

#include "Prediction.h"
#include "SoccerParams.h"
#include <memory>
#include <stdint.h>

class Predictor
{
private:
    std::shared_ptr<SoccerParams> m_params;

    int64_t m_last_time_us;
    int m_total_lost_frames;

    Prediction m_last_prediction;
    bool m_has_last_prediction;
public:
    struct FilteredOutput {
        Prediction prediction;
        struct Velocity {
            float x = 0.0f;
            float y = 0.0f;
        } velocity;
    };
public:
    Predictor(std::shared_ptr<SoccerParams> &params);
    FilteredOutput Filter(Prediction pred, float pred_delay_secs);
};