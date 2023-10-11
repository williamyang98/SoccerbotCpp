#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include "./gui_widgets.h"
#include "./SoccerPlayer.h"
#include <inttypes.h>

namespace widgets
{

void RenderVelocityMeter(const float value,  const float v_min, const float v_max, const ImVec2& size_arg) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = ImGui::CalcItemSize(size_arg, ImGui::CalcItemWidth(), g.FontSize + style.FramePadding.y * 2.0f);

    ImVec2 bb_end = pos;
    bb_end.x += size.x;
    bb_end.y += size.y;
    ImRect bb(pos, bb_end);

    ImGui::ItemSize(size, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, 0))
        return;

    const float range = v_max - v_min;
    const float range_position = value - v_min;
    const float fraction = (range_position / range) - 0.5f;
    
    const auto COLOUR_RED = ImColor(255,0,0) ;
    const auto COLOUR_GREEN = ImColor(0,255,0) ;
    const auto color = (fraction < 0.0f) ? COLOUR_RED: COLOUR_GREEN;
    
    float x_start = 0.0f;
    float x_end = 0.0f;
    if (fraction < 0.0f) {
        x_start = 0.5f+fraction;
        x_end = 0.5f;
    } else {
        x_start = 0.5f;
        x_end = 0.5f+fraction;
    }
    x_start = ImSaturate(x_start);
    x_end = ImSaturate(x_end);

    // Render
    ImGui::RenderFrame(bb.Min, bb.Max, ImGui::GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
    bb.Expand(ImVec2(-style.FrameBorderSize, -style.FrameBorderSize));
    ImGui::RenderRectFilledRangeH(window->DrawList, bb, color, x_start, x_end, style.FrameRounding);
}

void RenderConfidenceMeter(const float value,  const float threshold, const ImVec2& size_arg) {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 size = ImGui::CalcItemSize(size_arg, ImGui::CalcItemWidth(), g.FontSize + style.FramePadding.y * 2.0f);

    ImVec2 bb_end = pos;
    bb_end.x += size.x;
    bb_end.y += size.y;
    ImRect bb(pos, bb_end);

    ImGui::ItemSize(size, style.FramePadding.y);
    if (!ImGui::ItemAdd(bb, 0))
        return;

    const float fraction = ImSaturate(value);
    const auto COLOUR_RED = ImColor(255,0,0);
    const auto COLOUR_GREEN = ImColor(0,255,0) ;
    const auto color = (fraction < threshold) ? COLOUR_RED: COLOUR_GREEN;

    // Render
    ImGui::RenderFrame(bb.Min, bb.Max, ImGui::GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
    bb.Expand(ImVec2(-style.FrameBorderSize, -style.FrameBorderSize));
    ImGui::RenderRectFilledRangeH(window->DrawList, bb, color, 0.0f, fraction, style.FrameRounding);
}

int RenderTimings(const char* label, const SoccerPlayer::Timings* timings, const size_t N, const ImVec2& size_arg) {
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (window->SkipItems)
        return -1;

    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);

    const ImVec2 label_size = ImGui::CalcTextSize(label, NULL, true);
    const ImVec2 frame_size = ImGui::CalcItemSize(size_arg, ImGui::CalcItemWidth(), label_size.y + style.FramePadding.y * 2.0f);

    const ImRect frame_bb(window->DC.CursorPos, window->DC.CursorPos + frame_size);
    const ImRect inner_bb(frame_bb.Min + style.FramePadding, frame_bb.Max - style.FramePadding);
    const ImRect total_bb(frame_bb.Min, frame_bb.Max + ImVec2(label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, 0));
    ImGui::ItemSize(total_bb, style.FramePadding.y);
    if (!ImGui::ItemAdd(total_bb, 0, &frame_bb))
        return -1;

    const bool hovered = ImGui::ItemHoverable(frame_bb, id);

    // Determine scale from values if not specified
    int64_t v_max = 0;
    for (size_t i = 0; i < N; i++) {
        auto timing = timings[i];
        v_max = ImMax(v_max, timing.us_total);
    }

    ImGui::RenderFrame(frame_bb.Min, frame_bb.Max, ImGui::GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);
    
    // 0 = Model
    // 1 = Resize
    // 2 = Convert
    // 3 = Grab
    const uint8_t COL_INACTIVE_VAL = 180;
    const uint8_t COL_ACTIVE_VAL = 255;
    const ImColor COL_INACTIVE[4] = {
        ImColor(COL_INACTIVE_VAL,0,0),
        ImColor(0,COL_INACTIVE_VAL,0),
        ImColor(0,0,COL_INACTIVE_VAL),
        ImColor(0,COL_INACTIVE_VAL,COL_INACTIVE_VAL),
    };
    const ImColor COL_ACTIVE[4] = {
        ImColor(COL_ACTIVE_VAL,0,0),
        ImColor(0,COL_ACTIVE_VAL,0),
        ImColor(0,0,COL_ACTIVE_VAL),
        ImColor(0,COL_ACTIVE_VAL,COL_ACTIVE_VAL),
    };


    int idx_hovered = -1;
    if (N >= 1) {
        int res_w = ImMin(int(frame_size.x), int(N));

        // Tooltip on hover
        if (hovered && inner_bb.Contains(g.IO.MousePos)) {
            const float t = ImClamp((g.IO.MousePos.x - inner_bb.Min.x) / (inner_bb.Max.x - inner_bb.Min.x), 0.0f, 0.9999f);
            const int v_idx = int(t * float(N)) % N;

            const auto timing = timings[v_idx]; 
            ImGui::BeginTooltip();
            
            ImGui::PushStyleColor(ImGuiCol_Text, COL_ACTIVE[0].Value); ImGui::Text("#"); ImGui::PopStyleColor(); ImGui::SameLine();
            ImGui::Text("Model   %" PRIi64 "us", timing.us_model_inference);
            ImGui::PushStyleColor(ImGuiCol_Text, COL_ACTIVE[1].Value); ImGui::Text("#"); ImGui::PopStyleColor(); ImGui::SameLine();
            ImGui::Text("Resize  %" PRIi64 "us", timing.us_image_resize);
            ImGui::PushStyleColor(ImGuiCol_Text, COL_ACTIVE[2].Value); ImGui::Text("#"); ImGui::PopStyleColor(); ImGui::SameLine();
            ImGui::Text("Format  %" PRIi64 "us", timing.us_image_convert);
            ImGui::PushStyleColor(ImGuiCol_Text, COL_ACTIVE[3].Value); ImGui::Text("#"); ImGui::PopStyleColor(); ImGui::SameLine();
            ImGui::Text("Grab    %" PRIi64 "us", timing.us_image_grab);
            ImGui::EndTooltip();
            idx_hovered = v_idx;
        }

        const float x_step = 1.0f / float(res_w);
        const float inv_v_max = (v_max == 0) ? 0.0f : 1.0f/float(v_max);

        float x0 = 0.0f;
        for (int i = 0; i < res_w; i++) {
            const float x1 = x0 + x_step;
            const int v1_idx = int(x0*N + 0.5f) % N;

            const auto timing = timings[v1_idx];

            constexpr int TOTAL_VALUES = 2;
            int64_t y_values[TOTAL_VALUES];
            y_values[0] = timing.us_model_inference;
            y_values[1] = timing.us_image_resize + y_values[0];
            // y_values[2] = timing.us_image_convert + y_values[1];
            // y_values[3] = timing.us_image_grab + y_values[2];

            float yn_prev = 0.0f;
            for (int j = 0; j < TOTAL_VALUES; j++) {
                const float yn_new = float(y_values[j]) * inv_v_max;
                const ImVec2 tn0 = ImVec2(x0, 1.0f-yn_prev);                       // Point in the normalized space of our target rectangle
                const ImVec2 tn1 = ImVec2(x1, 1.0f-yn_new);
                const ImVec2 pos0 = ImLerp(inner_bb.Min, inner_bb.Max, tn0);
                const ImVec2 pos1 = ImLerp(inner_bb.Min, inner_bb.Max, tn1);
                const ImColor colour = (idx_hovered == v1_idx) ? COL_ACTIVE[j] : COL_INACTIVE[j];
                window->DrawList->AddRectFilled(pos0, pos1, colour);
                yn_prev = yn_new;
            }
            x0 = x1;
        }
    }

    if (label_size.x > 0.0f) {
        ImGui::RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, inner_bb.Min.y), label);
    }
    // Return hovered index or -1 if none are hovered.
    // This is currently not exposed in the public API because we need a larger redesign of the whole thing, but in the short-term we are making it available in PlotEx().
    return idx_hovered;
}

};
