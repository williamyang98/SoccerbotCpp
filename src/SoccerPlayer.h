#pragma once

#include <stdint.h>
#include <memory>
#include <vector>

#include "IModel.h"
#include "util/MSS.h"
#include "Prediction.h"
#include "Predictor.h"
#include "SoccerParams.h"

class SoccerPlayer 
{
public: 
    template <typename T>
    struct Vec2D {
        T x = T(0); 
        T y = T(0);
    };
    struct Timings {
        int64_t us_image_grab = 0;
        int64_t us_image_resize = 0;
        int64_t us_image_convert = 0;
        int64_t us_model_inference = 0;
    };
    struct Controls {
        bool can_track = false;
        bool can_smart_click = false;
        bool can_always_click = false;
        bool can_use_predictor = true;
        int click_padding = 5;
    };
    struct Status {
        bool is_tracking = false;
        bool is_clicking = false;
        bool is_soft_trigger = false;
        bool is_hard_trigger = false;
    };
private:
    std::unique_ptr<IModel> m_model; 
    std::unique_ptr<Predictor> m_predictor;
    std::shared_ptr<util::MSS> m_mss;
    std::shared_ptr<SoccerParams> m_params;
    
    std::vector<RGBA<uint8_t>> m_resize_buffer;
    Vec2D<int> m_resize_buffer_size;
    Vec2D<int> m_capture_buffer_size;

    Prediction m_raw_pred;
    Prediction m_filtered_pred;
    Prediction m_prev_filtered_pred;
    bool m_has_prev_filtered_pred;
    Vec2D<float> m_velocity;

    Timings m_timings;
    Controls m_controls;
    Status m_status;
public:
    SoccerPlayer(
        std::unique_ptr<IModel>&& model,
        std::shared_ptr<util::MSS>& mss,
        std::shared_ptr<SoccerParams>& params);
    bool Update(const int top, const int left);
    const auto& GetResizeBuffer() const { return m_resize_buffer; }
    const auto& GetTimings() const { return m_timings; }
    const auto& GetStatus() const { return m_status; }
    auto& GetControls() { return m_controls; }

    Prediction GetRawPrediction() const { return m_raw_pred; }
    Prediction GetFilteredPrediction() const { return m_filtered_pred; }
    auto GetVelocity() const { return m_velocity; }
private:
    void ResizeImage();
    void ConvertImage();
    void UpdateTriggers(Prediction pred, const float vx, const float vy);
};