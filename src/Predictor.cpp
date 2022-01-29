#include "Predictor.h"
#include <chrono>

Predictor::Predictor(std::shared_ptr<SoccerParams> &params) {
    m_params = params;
    m_last_time_us = 0;
    m_total_lost_frames = 0;
    m_has_last_prediction = false;
}

Prediction Predictor::Filter(Prediction pred, float pred_delay_secs) {
    const auto get_us = []() {
        auto clock = std::chrono::high_resolution_clock::now(); 
        return std::chrono::time_point_cast<std::chrono::microseconds>(clock).time_since_epoch().count();
    };

    long long us_now = get_us();
    long long us_frame = us_now - m_last_time_us;
    m_last_time_us = us_now;

    // dt = seconds
    float dt_frame = (float)(us_frame) / 1000000.0f;

    auto &p = *m_params;

    if (pred.confidence < p.confidence_threshold) {
        m_total_lost_frames++;
        if (m_total_lost_frames >= p.max_lost_frames) {
            m_has_last_prediction = false;
        }
        return {0.0f, 0.0f, 0.0f};
    }

    // seed prediction if its missing (we track motion)
    m_total_lost_frames = 0;
    if (!m_has_last_prediction) {
        m_has_last_prediction = true;
        m_last_prediction = pred;
    }

    // calculate velocity
    auto &last_pred = m_last_prediction;
    float net_delay_secs = pred_delay_secs + p.additional_model_delay;
    float vx = (pred.x - last_pred.x) / dt_frame;
    float vy = (pred.y - last_pred.y) / dt_frame;

    float real_x = pred.x + vx*net_delay_secs;
    float real_y = pred.y + vy*net_delay_secs;

    // if ball is moving, then it is under influence of gravity
    if (vy != 0.0f && vx != 0.0f) {
        real_y -= p.acceleration * 0.5f * (net_delay_secs * net_delay_secs);
    }

    // calculate bounce
    float width_rel = p.relative_ball_width;
    float radius = width_rel / 2.0f;
    float right_border = 1.0f-radius;
    float left_border = radius;

    // reflect x position off the left or right border
    if (real_x > right_border) {
        float delta = real_x - right_border;
        real_x = right_border - delta;
    } else if (real_x < left_border) {
        float delta = left_border - real_x;
        real_x = left_border + delta;
    }

    m_last_prediction = pred;
    return {real_x, real_y, pred.confidence};
}
