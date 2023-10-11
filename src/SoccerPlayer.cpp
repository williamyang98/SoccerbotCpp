#include <__msvc_chrono.hpp>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize2.h"
#include "SoccerPlayer.h"
#include "util/AutoGui.h"
#include <chrono>

int clamp_value(int v, const int v_min, const int v_max) {
    if (v < v_min) v = v_min;
    if (v > v_max) v = v_max;
    return v;
}

SoccerPlayer::SoccerPlayer(
    std::unique_ptr<IModel>&& model,
    std::shared_ptr<util::MSS>& mss,
    std::shared_ptr<SoccerParams>& params)
{
    m_model = std::move(model);
    m_mss = mss;
    m_params = params;
    m_predictor = std::make_unique<Predictor>(params);
    
    auto in_buffer = m_model->GetInputBuffer();
    m_resize_buffer_size.x = in_buffer.width;
    m_resize_buffer_size.y = in_buffer.height;
    m_resize_buffer.resize(m_resize_buffer_size.x * m_resize_buffer_size.y);

    m_has_prev_filtered_pred = false;
    m_velocity = {0.0f, 0.0f};
    
    m_timing_index = 0;
    SetTimingHistoryLength(60*2);
}

void SoccerPlayer::SetTimingHistoryLength(const size_t N) {
    m_timings.resize(N);
    if (m_timing_index >= N) {
        m_timing_index = N-1;
    }
}

bool SoccerPlayer::Update(const int top, const int left) {
    // update the model from the bitmap
    const auto dt_grab_start = std::chrono::high_resolution_clock::now();
    m_mss->Grab(top, left);
    const auto dt_grab_end = std::chrono::high_resolution_clock::now();
    
    const auto dt_resize_start = std::chrono::high_resolution_clock::now();
    ResizeImage();
    const auto dt_resize_end = std::chrono::high_resolution_clock::now();
    
    const auto dt_convert_start = std::chrono::high_resolution_clock::now();
    ConvertImage();
    const auto dt_convert_end = std::chrono::high_resolution_clock::now();

    const auto dt_model_start = std::chrono::high_resolution_clock::now();
    m_model->Parse();
    const Prediction raw_pred = m_model->GetPrediction();
    const auto dt_model_end = std::chrono::high_resolution_clock::now();
    
    // Summarise timings
    Timings timing;
    timing.us_image_grab = std::chrono::duration_cast<std::chrono::microseconds>(dt_grab_end-dt_grab_start).count();
    timing.us_image_resize = std::chrono::duration_cast<std::chrono::microseconds>(dt_resize_end-dt_resize_start).count();
    timing.us_image_convert = std::chrono::duration_cast<std::chrono::microseconds>(dt_convert_end-dt_convert_start).count();
    timing.us_model_inference = std::chrono::duration_cast<std::chrono::microseconds>(dt_model_end-dt_model_start).count();
    timing.us_total = std::chrono::duration_cast<std::chrono::microseconds>(dt_model_end-dt_resize_start).count();
    m_timings[m_timing_index] = timing;
    m_timing_index = (m_timing_index + 1) % m_timings.size();

    const auto dt_prediction_delay = dt_model_end - dt_grab_end;
    const int64_t us_prediction_delay = std::chrono::duration_cast<std::chrono::microseconds>(dt_prediction_delay).count();
    const float sec_prediction_delay = float(us_prediction_delay) / 1e6f;

    // play soccer
    const auto filtered_output = m_predictor->Filter(raw_pred, sec_prediction_delay);
    const Prediction filtered_pred = filtered_output.prediction;
    m_status.is_tracking = filtered_pred.confidence > m_params->confidence_threshold;
    UpdateTriggers(filtered_pred, filtered_output.velocity.x, filtered_output.velocity.y);
    m_status.is_clicking = m_status.is_soft_trigger || m_status.is_hard_trigger;

    if (m_controls.can_track && m_status.is_tracking) {
        const auto pred = m_controls.can_use_predictor ? filtered_pred : raw_pred;
        const int screen_x = left + int(      pred.x  * float(m_capture_buffer_size.x));
        const int screen_y = top  + int((1.0f-pred.y) * float(m_capture_buffer_size.y));
        
        // NOTE: We do this to prevent unfocusing the window
        const int click_padding = m_controls.click_padding;
        const int click_x = clamp_value(screen_x, left+click_padding, left+m_capture_buffer_size.x-click_padding);
        const int click_y = clamp_value(screen_y, top+click_padding, top+m_capture_buffer_size.y-click_padding);
        util::SetCursorPosition(click_x, click_y);

        const bool is_click = (m_controls.can_smart_click && m_status.is_clicking) || m_controls.can_always_click;
        if (is_click) {
            util::Click(click_x, click_y, util::MouseButton::LEFT);
        }
    }

    // update predictions
    m_raw_pred = raw_pred;
    m_filtered_pred = Prediction { filtered_pred.x, filtered_pred.y, filtered_pred.confidence };

    return true;
}

void SoccerPlayer::ResizeImage() {
    auto bitmap = m_mss->GetBitmap();
    const auto src_size = bitmap.GetSize();
    m_capture_buffer_size.x = src_size.x;
    m_capture_buffer_size.y = src_size.y;

    // NOTE: The screenshot buffer can be resized, so we might have to do some cropping
    const auto buffer_max_size = m_mss->GetMaxSize();

    const int total_channels = 4;
    const int src_stride = sizeof(uint8_t)*total_channels*buffer_max_size.x;
    const int dst_stride = sizeof(uint8_t)*total_channels*m_resize_buffer_size.x;
    const int src_rows_skip = buffer_max_size.y - src_size.y;
    const int src_bytes_offset = src_rows_skip*src_stride;

    auto &sec = bitmap.GetBitmap();
    BITMAP &bmp = sec.dsBm;
    const uint8_t *src_buffer = reinterpret_cast<uint8_t *>(bmp.bmBits);
    uint8_t *dst_buffer = reinterpret_cast<uint8_t *>(m_resize_buffer.data());

    // resize with quality
    if ((m_resize_buffer_size.x != src_size.x) || (m_resize_buffer_size.y != src_size.y)) {
        stbir_resize_uint8_linear(
            src_buffer+src_bytes_offset, src_size.x, src_size.y, src_stride,
            dst_buffer, m_resize_buffer_size.x, m_resize_buffer_size.y, dst_stride,
            STBIR_RGBA
        );
    // copy without resizing
    } else {
        int i_src = src_bytes_offset;
        int i_dst = 0;
        for (int y = 0; y < m_resize_buffer_size.y; y++) {
            std::memcpy(&dst_buffer[i_dst], &src_buffer[i_src], dst_stride);
            i_src += src_stride;
            i_dst += dst_stride;
        }
    }
}

void SoccerPlayer::ConvertImage() {
    RGB<float> *dst_buffer = m_model->GetInputBuffer().data;
    const RGBA<uint8_t> *src_buffer = m_resize_buffer.data();
    const int total_pixels = m_resize_buffer_size.x * m_resize_buffer_size.y;
    for (int i = 0; i < total_pixels; i++) {
        dst_buffer[i].r = float(src_buffer[i].b) / 255.0f;
        dst_buffer[i].g = float(src_buffer[i].g) / 255.0f;
        dst_buffer[i].b = float(src_buffer[i].r) / 255.0f;
    }
}

void SoccerPlayer::UpdateTriggers(Prediction pred, const float vx, const float vy) {
    auto &p = *(m_params.get());
    if (pred.confidence < m_params->confidence_threshold) {
        m_status.is_soft_trigger = false;
        m_status.is_hard_trigger = false;
        m_velocity = {0.0f, 0.0f};
        return;
    }

    if (!m_has_prev_filtered_pred) {
        m_has_prev_filtered_pred = true;
        m_prev_filtered_pred = pred;
    }

    const auto last_pred = m_prev_filtered_pred;
    m_prev_filtered_pred = pred;
    m_velocity = {vx, vy};

    // ignore if outside of screen
    if ((pred.x >= 1.0f) || (pred.x <= 0.0f) || 
        (pred.y >= 1.0f) || (pred.y <= 0.0f)) 
    {
        m_status.is_soft_trigger = false;
        m_status.is_hard_trigger = false;
        m_velocity = {0.0f, 0.0f};
        return;
    }

    // falling down and near bottom of screen
    // or falling down really fast regardless of position
    if ((vy <= -p.fall_speed_trigger_soft) && (pred.y <= p.height_trigger_soft)) {
        m_status.is_soft_trigger = true;
        m_status.is_hard_trigger = false;
        return;
    }

    if ((vy <= -p.fall_speed_trigger_hard) || (pred.y <= p.height_trigger_hard)) {
        m_status.is_soft_trigger = false;
        m_status.is_hard_trigger = true;
        return;
    }

    m_status.is_soft_trigger = false;
    m_status.is_hard_trigger = false;
    return;
}
