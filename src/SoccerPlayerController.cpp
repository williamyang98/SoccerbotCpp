#include "SoccerPlayerController.h"

#include <chrono>

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

SoccerPlayerController::SoccerPlayerController(
    std::unique_ptr<Model> &model,
    std::shared_ptr<util::MSS> &mss) 
{
    m_player = std::make_unique<SoccerPlayer>(model, mss);
    m_is_model_running = false;
    m_is_alive = true;

    m_model_thread = std::make_unique<std::thread>([this]() {
        ThreadLoop();
    });

    m_us_frame_time = 0;
    m_us_forward_time = 0;
    m_us_parse_time = 0;
}

SoccerPlayerController::~SoccerPlayerController() {
    m_is_alive = false;
    m_model_thread->join();
}

Model::Result SoccerPlayerController::GetResult() const {
    if (m_is_model_running) {
        return m_player->GetResult();
    } else {
        return {0, 0, 0.0f};
    }
}

void SoccerPlayerController::SetPosition(const int top, const int left) {
    auto lock = std::scoped_lock(m_pos_mutex);
    m_top = top;
    m_left = left;
}

SoccerPlayerController::Position SoccerPlayerController::GetPosition() {
    auto lock = std::scoped_lock(m_pos_mutex);
    return {m_top, m_left};
}

// runs in a separate thread
void SoccerPlayerController::ThreadLoop() {
    auto timer_start = std::chrono::high_resolution_clock::now();
    while (m_is_alive) 
    {
        if (m_is_model_running) {
            // keep track of just the forward pass delay
            auto timer_start_forward = std::chrono::high_resolution_clock::now();
            m_player->Update(m_top, m_left);
            auto timer_end = std::chrono::high_resolution_clock::now();

            int64_t us_start_forward = std::chrono::time_point_cast<std::chrono::microseconds>(timer_start_forward).time_since_epoch().count();
            int64_t us_start = std::chrono::time_point_cast<std::chrono::microseconds>(timer_start).time_since_epoch().count();
            int64_t us_end = std::chrono::time_point_cast<std::chrono::microseconds>(timer_end).time_since_epoch().count();

            m_us_forward_time = us_end-us_start_forward;
            m_us_frame_time = us_end-us_start;
            m_us_parse_time = m_player->GetParseTimeMicroseconds();

            timer_start = timer_end;

        } else {
            Sleep(10);
            timer_start = std::chrono::high_resolution_clock::now();
        }
    }
}