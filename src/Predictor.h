#pragma once

#include "Prediction.h"
#include "SoccerParams.h"
#include <memory>

class Predictor
{
private:
    std::shared_ptr<SoccerParams> m_params;

    long long m_last_time_us;
    int m_total_lost_frames;

    Prediction m_last_prediction;
    bool m_has_last_prediction;
public:
    Predictor(std::shared_ptr<SoccerParams> &params);
    Prediction Filter(Prediction pred, float pred_delay_secs);
    
};