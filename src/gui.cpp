#include "gui.h"
#include "util/AutoGui.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include "imgui_internal.h"

#include <algorithm>
#include <inttypes.h>

static void RenderControls(App &app);
static void RenderStatistics(App &app);
static void RenderAIParameters(App &app);
static void RenderModelView(App &app);

static void RenderScreenImage(App &app);
static void RenderModelImage(App &app);

void RenderApp(App &app) {
    app.Update();
    RenderControls(app);
    RenderAIParameters(app);
    RenderStatistics(app);
    RenderModelView(app);
}

void RenderControls(App &app) {
    auto &player_controller = *(app.m_player);
    const auto screen_size = util::GetScreenSize();

    ImGui::Begin("Controls");

    bool is_model_running = player_controller.GetIsRunning();
    if (ImGui::Checkbox("Is model (F1)", &is_model_running)) {
        player_controller.SetIsRunning(is_model_running);
    }

    ImGui::Checkbox("Is render (F2)", &app.m_is_render_running);
    
    bool is_tracking = player_controller->GetIsTracking();
    if (ImGui::Checkbox("Is tracking ball (F3)", &is_tracking)) {
        player_controller->SetIsTracking(is_tracking);
    }

    bool is_smart_clicking = player_controller->GetIsSmartClicking();
    if (ImGui::Checkbox("Is smart clicking ball (F4)", &is_smart_clicking)) {
        player_controller->SetIsSmartClicking(is_smart_clicking);
    }

    bool is_using_predictor = player_controller->GetIsUsingPredictor();
    if (ImGui::Checkbox("Is using predictor (F5)", &is_using_predictor)) {
        player_controller->SetIsUsingPredictor(is_using_predictor);
    }

    bool is_always_clicking = player_controller->GetIsAlwaysClicking();
    if (ImGui::Checkbox("Is always clicking ball (F6)", &is_always_clicking)) {
        player_controller->SetIsAlwaysClicking(is_always_clicking);
    }
    
    ImGui::Separator();

    ImGui::Checkbox("Show raw prediction", &app.m_render_overlay_flags.raw_pred);
    ImGui::Checkbox("Show filtered prediction", &app.m_render_overlay_flags.filtered_pred);

    ImGui::Separator();

    static int screen_width = app.m_screen_width;
    static int screen_height = app.m_screen_height;
    {
        auto& player_controller = app.m_player;
        auto screen_pos = player_controller->GetPosition();
        bool v = false;
        v = ImGui::DragInt("Top", &screen_pos.top, 1, 0, screen_size.height) || v;
        v = ImGui::DragInt("Left", &screen_pos.left, 1, 0, screen_size.width) || v;
        if (v) {
            player_controller->SetPosition(screen_pos.top, screen_pos.left);
        }
    }
    ImGui::DragInt("Width", &screen_width, 1, 0, screen_size.width);
    ImGui::DragInt("Height", &screen_height, 1, 0, screen_size.height);
    if ((screen_width != app.m_screen_width) || 
        (screen_height != app.m_screen_height)) 
    {
        if (ImGui::Button("Submit changes")) {
            app.SetScreenshotSize(screen_width, screen_height);
        }
    }

    ImGui::Separator();

    ImGui::Text("pointer = %p", app.m_screenshot_texture_view);
    ImGui::Text("model_size          = %d x %d", app.m_model_width, app.m_model_height);
    ImGui::Text("screenshot_size     = %d x %d", app.m_screen_width, app.m_screen_height);
    ImGui::Text("texture_size        = %d x %d", app.m_texture_width, app.m_texture_height);

    auto max_buffer_size = app.m_mss->GetMaxSize();
    ImGui::Text("max_screenshot_size = %d x %d", max_buffer_size.x, max_buffer_size.y);

    ImGui::End(); 
}

void RenderVelocityMeter(const float value,  const float v_min, const float v_max, const ImVec2& size_arg=ImVec2(0,0)) {
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

void RenderConfidenceMeter(const float value,  const float threshold, const ImVec2& size_arg=ImVec2(0,0)) {
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

void RenderAIParameters(App &app) {
    ImGui::Begin("AI Params");

    auto &p = *(app.m_params);
    auto &player_controller = *(app.m_player);

    ImGui::Text("Soccer parameters");
    ImGui::SliderFloat("acceleration", &p.acceleration, 0.0f, 10.0f);
    ImGui::SliderFloat("additional delay", &p.additional_model_delay, 0.0f, 0.5f);
    ImGui::SliderFloat("confidence threshold", &p.confidence_threshold, 0.0f, 1.0f);
    ImGui::SliderInt("max lost frames", &p.max_lost_frames, 0, 5);

    const float VMAX = 10.0f;
    ImGui::Separator();
    ImGui::Text("Triggers");
    ImGui::SliderFloat("fall speed soft", &p.fall_speed_trigger_soft, 0.0f, VMAX);
    ImGui::SliderFloat("fall height soft", &p.height_trigger_soft, 0.0f, 1.0f);
    ImGui::SliderFloat("fall speed hard", &p.fall_speed_trigger_hard, 0.0f, VMAX);
    ImGui::SliderFloat("fall height hard", &p.height_trigger_hard, 0.0f, 1.0f);

    ImGui::Separator();
    auto raw_pred = player_controller->GetRawPrediction();
    ImGui::Text("Confidence: %+.3f", raw_pred.confidence);
    RenderConfidenceMeter(raw_pred.confidence, p.confidence_threshold);
    ImGui::SameLine();
    ImGui::Text("confidence");
    
    ImGui::Separator();
    auto vel = player_controller->GetVelocity();
    ImGui::Text("Velocity: x=%+.3f y=%+.3f", vel.x, vel.y);
    RenderVelocityMeter(vel.x, -VMAX, +VMAX);
    ImGui::SameLine();
    ImGui::Text("dx");
    RenderVelocityMeter(vel.y, -VMAX, +VMAX);
    ImGui::SameLine();
    ImGui::Text("dy");
    ImGui::Separator();
    ImGui::RadioButton("Tracking", player_controller->GetCanTrack());
    ImGui::RadioButton("Soft trigger", player_controller->GetIsSoftTrigger());
    ImGui::RadioButton("Hard trigger", player_controller->GetIsHardTrigger());

    ImGui::End();
}

void RenderStatistics(App &app) {
    auto &player_controller = app.m_player;

    ImGui::Begin("Statistics");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    auto us_parse = player_controller->GetParseTimeMicroseconds();
    auto us_forward = player_controller->GetForwardTimeMicroseconds();
    auto us_frame = player_controller->GetFrameTimeMicroseconds();
    ImGui::Text("Parse   time: %" PRIi64 " us", us_parse);
    ImGui::Text("Forward time: %" PRIi64 " us", us_forward);
    ImGui::Text("Frame   time: %" PRIi64 " us", us_frame);
    ImGui::End();
}



void RenderModelView(App &app) {
    ImGui::Begin("Render");
    if (ImGui::BeginTabBar("Views")) {
        if (ImGui::BeginTabItem("Screen view")) {
            RenderScreenImage(app);
        }
        if (ImGui::BeginTabItem("Model view")) {
            RenderModelImage(app);
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
} 

void RenderScreenImage(App &app) {
    static float zoom_scale = 1.0f;
    static auto last_mouse_pos = ImGui::GetMousePos();
    static auto is_drag_enabled = false;
    const auto screen_size = util::GetScreenSize();
    auto &player_controller = app.m_player;
    auto screen_pos = player_controller->GetPosition();
    ImGui::SliderFloat("Scale", &zoom_scale, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateScreenshotTexture();

    ImVec4 border_col = ImGui::GetStyleColorVec4(ImGuiCol_Border);
    ImGui::Image(
        (void*)app.m_screenshot_texture_view, 
        ImVec2(app.m_texture_width*zoom_scale, app.m_texture_height*zoom_scale),
        ImVec2(0,0), ImVec2(1,1),
        ImVec4(1,1,1,1),
        ImGui::GetStyleColorVec4(ImGuiCol_Border)
    );
    ImGui::EndTabItem();

    // create dragging controls for the image
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 10.0f)) {
            is_drag_enabled = true;
        }
    } 
    
    if (is_drag_enabled) {
        if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            is_drag_enabled = false;
        } else {
            auto latest_pos = ImGui::GetMousePos();
            ImVec2 delta;
            delta.x = latest_pos.x - last_mouse_pos.x;
            delta.y = latest_pos.y - last_mouse_pos.y;
            screen_pos.top   = std::clamp(screen_pos.top  - (int)(delta.y), 0, screen_size.height);
            screen_pos.left  = std::clamp(screen_pos.left - (int)(delta.x), 0, screen_size.width);
            player_controller->SetPosition(screen_pos.top, screen_pos.left);
        }
    }
    last_mouse_pos = ImGui::GetMousePos();
}

void RenderModelImage(App &app) {
    static float zoom_scale = (float)(app.m_screen_width) / (float)(app.m_model_width);
    static auto last_mouse_pos = ImGui::GetMousePos();
    static auto is_drag_enabled = false;
    const auto screen_size = util::GetScreenSize();

    auto &player_controller = app.m_player;
    auto screen_pos = player_controller->GetPosition();

    ImGui::SliderFloat("Scale", &zoom_scale, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateModelTexture();
    ImGui::Image(
        (void*)app.m_model_texture_view, 
        ImVec2(app.m_model_width*zoom_scale, app.m_model_height*zoom_scale),
        ImVec2(0,0), ImVec2(1,1),
        ImVec4(1,1,1,1),
        ImGui::GetStyleColorVec4(ImGuiCol_Border)
    );
    ImGui::EndTabItem();

    // create dragging controls for the image
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 10.0f)) {
            is_drag_enabled = true;
        }
    } 
    
    if (is_drag_enabled) {
        if (!ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            is_drag_enabled = false;
        } else {
            auto latest_pos = ImGui::GetMousePos();
            ImVec2 delta;
            delta.x = latest_pos.x - last_mouse_pos.x;
            delta.y = latest_pos.y - last_mouse_pos.y;
            screen_pos.top   = std::clamp(screen_pos.top  - (int)(delta.y), 0, screen_size.height);
            screen_pos.left  = std::clamp(screen_pos.left - (int)(delta.x), 0, screen_size.width);
            player_controller->SetPosition(screen_pos.top, screen_pos.left);
        }
    }
    last_mouse_pos = ImGui::GetMousePos();
}
