#pragma once

#include <imgui.h>
#include <stddef.h>
#include <stdint.h>
#include "./SoccerPlayer.h"

namespace widgets 
{

void RenderVelocityMeter(const float value, const float v_min, const float v_max, const ImVec2& size_arg=ImVec2(0,0));

void RenderConfidenceMeter(const float value, const float threshold, const ImVec2& size_arg=ImVec2(0,0));

int RenderTimings(const char* label, const SoccerPlayer::Timings* timings, const size_t N, const ImVec2& size_arg=ImVec2(0,0));

};
