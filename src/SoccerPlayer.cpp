#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "SoccerPlayer.h"
#include "util/AutoGui.h"

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
    m_width = in_buffer.width;
    m_height = in_buffer.height;
    m_channels = 4;
    m_resize_buffer = new uint8_t[m_width*m_height*m_channels];

    m_us_parse_time = 0;

    m_has_prev_filtered_pred = false;
    
    m_is_using_predictor = true;
    
    m_is_tracking = false;
    m_is_clicking = false;
    m_is_soft_trigger = false;
    m_is_hard_trigger = false;
    m_can_track = false;
    m_can_click = false;
    
    m_velocity = {0.0f, 0.0f};
}

SoccerPlayer::~SoccerPlayer() {
    delete[] m_resize_buffer;
}

bool SoccerPlayer::Update(const int top, const int left) {
    // update the model from the bitmap
    auto timer_grab = std::chrono::high_resolution_clock::now();
    m_mss->Grab(top, left);

    auto bitmap = m_mss->GetBitmap();
    const auto buffer_size = bitmap.GetSize();
    // NOTE: The screenshot buffer can be resized, so we might have to do some cropping
    const auto buffer_max_size = m_mss->GetMaxSize();
    const int input_stride = sizeof(uint8_t)*4*buffer_max_size.x;
    const int output_stride = sizeof(uint8_t)*4*m_width;
    const int rows_skip = buffer_max_size.y - buffer_size.y;
    const int rows_offset = rows_skip*input_stride;

    auto &sec = bitmap.GetBitmap();
    BITMAP &bmp = sec.dsBm;
    const uint8_t *buffer = (uint8_t *)(bmp.bmBits);

    // resize with quality
    if ((m_width != buffer_size.x) || (m_height != buffer_size.y)) {
        stbir_resize_uint8(
            buffer+rows_offset, buffer_size.x, buffer_size.y, input_stride, 
            m_resize_buffer, m_width, m_height, 0,
            m_channels);
    // copy without resizing
    } else {
        int i_src = rows_offset;
        int i_dst = 0;
        for (int y = 0; y < m_height; y++) {
            std::memcpy(&m_resize_buffer[i_dst], &buffer[i_src], output_stride);
            i_src += input_stride;
            i_dst += output_stride;
        }
    }

    // flip and convert
    RGB<float> *input_buffer = m_model->GetInputBuffer().data;
    const RGBA<uint8_t> *rgba_data = reinterpret_cast<const RGBA<uint8_t>*>(m_resize_buffer);
    for (int y = 0; y < m_height; y++) {
        for (int x = 0; x < m_width; x++) {
            int i = x + y*m_width;
            input_buffer[i].r = ((float)rgba_data[i].b) / 255.0f;
            input_buffer[i].g = ((float)rgba_data[i].g) / 255.0f;
            input_buffer[i].b = ((float)rgba_data[i].r) / 255.0f;
        }
    }

    // get telemetry for model parse time
    auto timer_start = std::chrono::high_resolution_clock::now();
    m_model->Parse();
    auto timer_end = std::chrono::high_resolution_clock::now();

    int64_t us_grab = std::chrono::time_point_cast<std::chrono::microseconds>(timer_grab).time_since_epoch().count();
    int64_t us_start = std::chrono::time_point_cast<std::chrono::microseconds>(timer_start).time_since_epoch().count();
    int64_t us_end = std::chrono::time_point_cast<std::chrono::microseconds>(timer_end).time_since_epoch().count();

    m_us_parse_time = us_end-us_start;
    int64_t us_forward_time = us_end-us_grab;

    // pass image through network
    Prediction raw_pred = m_model->GetPrediction();

    // play soccer
    float parse_time_secs = (float)(us_forward_time) / 1000000.0f;
    Prediction filtered_pred = m_predictor->Filter(raw_pred, parse_time_secs);
    m_can_track = filtered_pred.confidence > m_params->confidence_threshold;
    m_can_click = CheckIfClick(filtered_pred);

    // click on the ball
    if (m_is_using_predictor) {
        if (m_is_tracking && m_can_track) {
            int x = left +                 (int)(filtered_pred.x * (float)(buffer_size.x));
            int y = top  + buffer_size.y - (int)(filtered_pred.y * (float)(buffer_size.y));
            util::SetCursorPosition(x, y);
            if (m_is_clicking && m_can_click) {
                util::Click(x, y, util::MouseButton::LEFT);
            }
        }
    // using raw model predictions
    } else {
        if (m_is_tracking && m_can_track) {
            int x = left +                 (int)(raw_pred.x * (float)(buffer_size.x));
            int y = top  + buffer_size.y - (int)(raw_pred.y * (float)(buffer_size.y));
            util::SetCursorPosition(x, y);
            //  raw model predictions dont use velocity information
            if (m_is_clicking) {
                util::Click(x, y, util::MouseButton::LEFT);
            }
        }
    }

    // update predictions
    {
        m_raw_pred = raw_pred;
        m_filtered_pred = filtered_pred;
    }

    return true;
}

bool SoccerPlayer::CheckIfClick(Prediction pred) {
    auto &p = *m_params;
    if (pred.confidence < m_params->confidence_threshold) {
        return false;
    }

    if (!m_has_prev_filtered_pred) {
        m_has_prev_filtered_pred = true;
        m_prev_filtered_pred = pred;
    }

    auto &last_pred = m_prev_filtered_pred;
    float dx = pred.x - last_pred.x;
    float dy = pred.y - last_pred.y;

    // ignore if outside of screen
    if ((pred.x >= 1.0f) || (pred.x <= 0.0f) || 
        (pred.y >= 1.0f) || (pred.y <= 0.0f)) 
    {
        m_is_soft_trigger = false;
        m_is_hard_trigger = false;
        m_velocity = {0.0f, 0.0f};
        return false;
    }

    m_velocity = {dx, dy};
    // falling down and near bottom of screen
    // or falling down really fast regardless of position
    if ((dy <= -p.fall_speed_trigger_soft) && (pred.y <= p.height_trigger_soft)) {
        m_is_soft_trigger = true;
        m_is_hard_trigger = false;
        return true;
    }

    if ((dy <= -p.fall_speed_trigger_hard) || (pred.y <= p.height_trigger_hard)) {
        m_is_soft_trigger = false;
        m_is_hard_trigger = true;
        return true;
    }

    m_is_soft_trigger = false;
    m_is_hard_trigger = false;
    return false;
}

Prediction SoccerPlayer::GetRawPrediction() {
    return m_raw_pred;
}

Prediction SoccerPlayer::GetFilteredPrediction() {
    return m_filtered_pred;
}