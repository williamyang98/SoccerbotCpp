#pragma once

#include <memory>
#include <mutex>

#include "model.h"
#include "util/MSS.h"

class SoccerPlayer 
{
private:
    std::unique_ptr<Model> m_model; 
    std::shared_ptr<util::MSS> m_mss;

    uint8_t *m_resize_buffer;
    int m_width;
    int m_height;
    int m_channels;

    Model::Result m_result;
    std::mutex m_result_mutex;
    int64_t m_us_parse_time;
public:
    SoccerPlayer(
        std::unique_ptr<Model> &model,
        std::shared_ptr<util::MSS> &mss);
    ~SoccerPlayer();
    bool Update(const int top, const int left);
    uint8_t *GetResizeBuffer() { return m_resize_buffer; }
    inline int64_t GetParseTimeMicroseconds() const { return m_us_parse_time; }
    Model::Result GetResult();
};