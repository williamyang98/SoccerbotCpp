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
    RenderControls(app);
    RenderAIParameters(app);
    RenderStatistics(app);
    RenderModelView(app);
}

void RenderControls(App &app) {
    auto& model_controls = app.m_player->GetControls();
    ImGui::Begin("Controls");
    ImGui::Checkbox("Is model (F1)", &app.m_is_model_running);
    ImGui::Checkbox("Is render (F2)", &app.m_is_render_running);
    ImGui::Checkbox("Is tracking ball (F3)", &model_controls.can_track);
    ImGui::Checkbox("Is smart clicking ball (F4)", &model_controls.can_smart_click);
    ImGui::Checkbox("Is using predictor (F5)", &model_controls.can_use_predictor);
    ImGui::Checkbox("Is always clicking ball (F6)", &model_controls.can_always_click);
    ImGui::Separator();
    ImGui::Checkbox("Show raw prediction", &app.m_render_overlay_flags.raw_pred);
    ImGui::Checkbox("Show filtered prediction", &app.m_render_overlay_flags.filtered_pred);
    ImGui::SliderInt("Click padding", &model_controls.click_padding, 0, 10);
    ImGui::Separator();

    const auto screen_size = util::GetScreenSize();
    ImGui::Text("Screenshot area");
    ImGui::DragInt("Top", &app.m_screenshot_position.top, 1, 0, screen_size.height);
    ImGui::DragInt("Left", &app.m_screenshot_position.left, 1, 0, screen_size.width);

    int screen_width = app.m_screen_width;
    int screen_height = app.m_screen_height;
    ImGui::DragInt("Width", &screen_width, 1, 0, screen_size.width);
    ImGui::DragInt("Height", &screen_height, 1, 0, screen_size.height);
    if ((screen_width != app.m_screen_width) || (screen_height != app.m_screen_height)) {
        app.SetScreenshotSize(screen_width, screen_height);
    }

    ImGui::Separator();

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

    auto &params = *(app.m_params.get());
    auto &player = *(app.m_player.get());

    ImGui::Text("Soccer parameters");
    ImGui::SliderFloat("acceleration", &params.acceleration, 0.0f, 10.0f);
    ImGui::SliderFloat("additional delay", &params.additional_model_delay, 0.0f, 0.5f);
    ImGui::SliderFloat("confidence threshold", &params.confidence_threshold, 0.0f, 1.0f);
    ImGui::SliderInt("max lost frames", &params.max_lost_frames, 0, 5);

    const float VMAX = 10.0f;
    ImGui::Separator();
    ImGui::Text("Triggers");
    ImGui::SliderFloat("fall speed soft", &params.fall_speed_trigger_soft, 0.0f, VMAX);
    ImGui::SliderFloat("fall height soft", &params.height_trigger_soft, 0.0f, 1.0f);
    ImGui::SliderFloat("fall speed hard", &params.fall_speed_trigger_hard, 0.0f, VMAX);
    ImGui::SliderFloat("fall height hard", &params.height_trigger_hard, 0.0f, 1.0f);

    ImGui::Separator();
    auto raw_pred = player.GetRawPrediction();
    ImGui::Text("Confidence: %+.3f", raw_pred.confidence);
    RenderConfidenceMeter(raw_pred.confidence, params.confidence_threshold);
    ImGui::SameLine();
    ImGui::Text("confidence");
    
    ImGui::Separator();
    auto vel = player.GetVelocity();
    ImGui::Text("Velocity: x=%+.3f y=%+.3f", vel.x, vel.y);
    RenderVelocityMeter(vel.x, -VMAX, +VMAX);
    ImGui::SameLine();
    ImGui::Text("dx");
    RenderVelocityMeter(vel.y, -VMAX, +VMAX);
    ImGui::SameLine();
    ImGui::Text("dy");
    ImGui::Separator();
    const auto status = player.GetStatus();
    ImGui::RadioButton("Tracking", status.is_tracking);
    ImGui::SameLine();
    ImGui::RadioButton("Soft trigger", status.is_soft_trigger);
    ImGui::SameLine();
    ImGui::RadioButton("Hard trigger", status.is_hard_trigger);

    ImGui::End();
}

void RenderStatistics(App &app) {
    ImGui::Begin("Statistics");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    const auto timings = app.m_player->GetTimings();
    ImGui::Text("Grab   : %" PRIi64 " us", timings.us_image_grab);
    ImGui::Text("Resize : %" PRIi64 " us", timings.us_image_resize);
    ImGui::Text("Convert: %" PRIi64 " us", timings.us_image_convert);
    ImGui::Text("Model  : %" PRIi64 " us", timings.us_model_inference);
    ImGui::End();
}

void RenderImageControls(App& app) {
    static auto last_mouse_pos = ImGui::GetMousePos();
    static auto is_drag_enabled = false;
    const auto screen_size = util::GetScreenSize();

    // create dragging controls for the image
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 10.0f)) {
            is_drag_enabled = true;
        }

        const float scroll_value = ImGui::GetIO().MouseWheel;
        if (scroll_value != 0.0f) {
            constexpr float zoom_in = 1.1f;
            constexpr float zoom_out = 1.0f/zoom_in;
            float zoom_factor = 1.0f + scroll_value;
            zoom_factor = std::clamp(zoom_factor, zoom_out, zoom_in);

            int new_width = int(float(app.m_screen_width) * zoom_factor);
            int new_height = int(float(app.m_screen_height) * zoom_factor);
            new_width = std::clamp(new_width, 0, screen_size.width);
            new_height = std::clamp(new_height, 0, screen_size.height);
            app.SetScreenshotSize(new_width, new_height);
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
            auto& pos = app.m_screenshot_position;
            pos.top   = std::clamp(pos.top  - int(delta.y), 0, screen_size.height);
            pos.left  = std::clamp(pos.left - int(delta.x), 0, screen_size.width);
        }
    }
    last_mouse_pos = ImGui::GetMousePos();
}

void RenderModelView(App &app) {
    ImGui::Begin("Render");
    if (ImGui::BeginTabBar("Views")) {
        if (ImGui::BeginTabItem("Screen view")) {
            RenderScreenImage(app);
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Model view")) {
            RenderModelImage(app);
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
} 

void RenderScreenImage(App &app) {
    static float zoom_scale = 1.0f;
    ImGui::SliderFloat("Scale", &zoom_scale, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateScreenshotTexture();

    ImVec4 border_col = ImGui::GetStyleColorVec4(ImGuiCol_Border);
    ImGui::SetItemUsingMouseWheel();
    ImGui::Image(
        (void*)app.m_screenshot_texture_view, 
        ImVec2(app.m_texture_width*zoom_scale, app.m_texture_height*zoom_scale),
        ImVec2(0,0), ImVec2(1,1),
        ImVec4(1,1,1,1),
        ImGui::GetStyleColorVec4(ImGuiCol_Border)
    );
    RenderImageControls(app);
}

void RenderModelImage(App &app) {
    static float zoom_scale = (float)(app.m_screen_width) / (float)(app.m_model_width);
    ImGui::SliderFloat("Scale", &zoom_scale, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateModelTexture();
    ImGui::SetItemUsingMouseWheel();
    ImGui::Image(
        (void*)app.m_model_texture_view, 
        ImVec2(app.m_model_width*zoom_scale, app.m_model_height*zoom_scale),
        ImVec2(0,0), ImVec2(1,1),
        ImVec4(1,1,1,1),
        ImGui::GetStyleColorVec4(ImGuiCol_Border)
    );
    RenderImageControls(app);
}
