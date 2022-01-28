#include "SoccerPlayer.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


SoccerPlayer::SoccerPlayer(
    std::unique_ptr<Model> &model,
    std::shared_ptr<util::MSS> &mss)
{
    m_model = std::move(model);
    m_mss = mss;

    auto in_size = m_model->GetInputSize();
    m_width = in_size.x;
    m_height = in_size.y;
    m_channels = 4;
    m_resize_buffer = new uint8_t[m_width*m_height*m_channels];

    m_result.x = 0;
    m_result.y = 0;
    m_result.confidence = 0.0f;

    m_us_parse_time = 0;
}

SoccerPlayer::~SoccerPlayer() {
    delete[] m_resize_buffer;
}

bool SoccerPlayer::Update(const int top, const int left) {
    m_mss->Grab(top, left);

    auto bitmap = m_mss->GetBitmap();
    auto buffer_size = bitmap.GetSize();
    auto buffer_max_size = m_mss->GetMaxSize();
    int input_stride = sizeof(uint8_t)*4*buffer_max_size.x;

    auto &sec = bitmap.GetBitmap();
    BITMAP &bmp = sec.dsBm;
    uint8_t *buffer = (uint8_t *)(bmp.bmBits);

    int rows_skip = buffer_max_size.y - buffer_size.y;
    int rows_offset = rows_skip*input_stride;

    stbir_resize_uint8(
        buffer+rows_offset, buffer_size.x, buffer_size.y, input_stride, 
        m_resize_buffer, m_width, m_height, 0,
        m_channels);


    // slower convert?
    // stbi__vertical_flip(m_resize_buffer, m_width, m_height, m_channels);
    // assert(m_model->CopyDataToInput(m_resize_buffer, m_width, m_height));

    // flip and convert
    RGB<float> *input_buffer = m_model->GetInputBuffer();
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
    int64_t us_start = std::chrono::time_point_cast<std::chrono::microseconds>(timer_start).time_since_epoch().count();
    int64_t us_end = std::chrono::time_point_cast<std::chrono::microseconds>(timer_end).time_since_epoch().count();
    m_us_parse_time = us_end-us_start;

    auto result = m_model->GetResult();
    {
        auto result_lock = std::scoped_lock(m_result_mutex);
        m_result = result;
    }
    return true;
}

Model::Result SoccerPlayer::GetResult() {
    auto lock = std::scoped_lock(m_result_mutex);
    return m_result;
}