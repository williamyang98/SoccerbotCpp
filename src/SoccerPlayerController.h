#pragma once

#include <memory>
#include <thread>
#include <mutex>
#include <atomic>

#include "SoccerPlayer.h"

class SoccerPlayerController
{
public:
    struct Position {
        int top, left;
    };
private:
    std::unique_ptr<std::thread> m_model_thread; 
    std::atomic<bool> m_is_model_running;
    std::atomic<bool> m_is_alive;
    std::unique_ptr<SoccerPlayer> m_player;

    std::atomic<int> m_top, m_left;
    std::mutex m_pos_mutex;

    // time between each frame
    std::atomic<int64_t> m_us_frame_time;
    // time take to do a forward pass of the network with screenshot
    std::atomic<int64_t> m_us_forward_time;
    // time take to do a pass of the network
    std::atomic<int64_t> m_us_parse_time;
public:
    SoccerPlayerController(
        std::unique_ptr<Model> &model,
        std::shared_ptr<util::MSS> &mss);
    ~SoccerPlayerController();
    Model::Result GetResult() const;
    bool GetIsRunning() const { return m_is_model_running; }
    void SetIsRunning(bool is_running) { m_is_model_running = is_running; }
    void SetPosition(const int top, const int left);
    Position GetPosition();
    uint8_t *GetResizeBuffer() { return m_player->GetResizeBuffer(); }
    inline int64_t GetFrameTimeMicroseconds() const { return m_us_frame_time; }
    inline int64_t GetForwardTimeMicroseconds() const { return m_us_forward_time; }
    inline int64_t GetParseTimeMicroseconds() const { return m_us_parse_time; }
private:
    void ThreadLoop();
};