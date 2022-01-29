#pragma once

#include <memory>
#include <mutex>
#include <atomic>

#include "model.h"
#include "util/MSS.h"
#include "Prediction.h"
#include "Predictor.h"
#include "SoccerParams.h"

class SoccerPlayer 
{
private:
    std::unique_ptr<Model> m_model; 
    std::unique_ptr<Predictor> m_predictor;
    std::shared_ptr<util::MSS> m_mss;
    std::shared_ptr<SoccerParams> m_params;

    uint8_t *m_resize_buffer;
    int m_width;
    int m_height;
    int m_channels;

    Prediction m_raw_pred;
    Prediction m_filtered_pred;
    std::mutex m_raw_pred_mutex;
    std::mutex m_filtered_pred_mutex;

    Prediction m_prev_filtered_pred;
    bool m_has_prev_filtered_pred;

    int64_t m_us_parse_time;

    std::atomic<bool> m_is_tracking;
    std::atomic<bool> m_is_clicking;
public:
    SoccerPlayer(
        std::unique_ptr<Model> &model,
        std::shared_ptr<util::MSS> &mss,
        std::shared_ptr<SoccerParams> &params);
    ~SoccerPlayer();
    bool Update(const int top, const int left);
    uint8_t *GetResizeBuffer() { return m_resize_buffer; }
    inline int64_t GetParseTimeMicroseconds() const { return m_us_parse_time; }

    Prediction GetRawPrediction();
    Prediction GetFilteredPrediction();

    inline bool GetIsTracking() const { return m_is_tracking; }
    inline bool GetIsClicking() const { return m_is_clicking; }
    inline void SetIsTracking(bool v) { m_is_tracking = v; }
    inline void SetIsClicking(bool v) { m_is_clicking = v; }
private:
    bool CheckIfClick(Prediction pred);
};