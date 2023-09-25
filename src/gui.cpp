#include "gui.h"
#include "util/AutoGui.h"

#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"


#include <algorithm>

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

    bool is_clicking = player_controller->GetIsClicking();
    if (ImGui::Checkbox("Is clicking ball (F4)", &is_clicking)) {
        player_controller->SetIsClicking(is_clicking);
    }

    bool is_using_predictor = player_controller->GetIsUsingPredictor();
    if (ImGui::Checkbox("Is using predictor (F5)", &is_using_predictor)) {
        player_controller->SetIsUsingPredictor(is_using_predictor);
    }
    

    ImGui::Text("pointer = %p", app.m_screenshot_texture_view);
    ImGui::Text("model_size          = %d x %d", app.m_model_width, app.m_model_height);
    ImGui::Text("screenshot_size     = %d x %d", app.m_screen_width, app.m_screen_height);
    ImGui::Text("texture_size        = %d x %d", app.m_texture_width, app.m_texture_height);

    auto max_buffer_size = app.m_mss->GetMaxSize();
    ImGui::Text("max_screenshot_size = %d x %d", max_buffer_size.x, max_buffer_size.y);

    static int screen_width = app.m_screen_width;
    static int screen_height = app.m_screen_height;
    ImGui::DragInt("screen width", &screen_width, 1, 0, screen_size.width);
    ImGui::DragInt("screen height", &screen_height, 1, 0, screen_size.height);
    if ((screen_width != app.m_screen_width) || 
        (screen_height != app.m_screen_height)) 
    {
        if (ImGui::Button("Submit changes")) {
            app.SetScreenshotSize(screen_width, screen_height);
        }
    }

    ImGui::End(); 
}

void RenderAIParameters(App &app) {
    ImGui::Begin("AI Params");

    auto &p = *(app.m_params);
    auto &player_controller = *(app.m_player);

    ImGui::Text("Soccer parameters");
    ImGui::DragFloat("acceleration", &p.acceleration, 0.1f, 0.0f, 10.0f);
    ImGui::DragFloat("additional delay", &p.additional_model_delay, 0.001f, 0.0f, 0.5f);
    ImGui::DragFloat("confidence threshold", &p.confidence_threshold, 0.01f, 0.0f, 1.0f);
    ImGui::DragInt("max lost frames", &p.max_lost_frames, 1, 0, 5);

    ImGui::Separator();
    ImGui::Text("Triggers");
    ImGui::DragFloat("fall speed soft", &p.fall_speed_trigger_soft, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("fall height soft", &p.height_trigger_soft, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("fall speed hard", &p.fall_speed_trigger_hard, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("fall height hard", &p.height_trigger_hard, 0.01f, 0.0f, 1.0f);
    
    ImGui::Separator();
    ImGui::Text("AI state");
    auto vel = player_controller->GetVelocity();
    ImGui::Text("Velocity: x=%.3f y=%.3f", vel.x, vel.y);
    ImGui::Text("Soft trigger: %d", player_controller->GetIsSoftTrigger());
    ImGui::Text("Hard trigger: %d", player_controller->GetIsHardTrigger());
    ImGui::Text("Can track: %d", player_controller->GetCanTrack());
    ImGui::Text("Can click: %d", player_controller->GetCanClick());

    ImGui::End();
}

void RenderStatistics(App &app) {
    auto &player_controller = app.m_player;

    ImGui::Begin("Statistics");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    auto us_parse = player_controller->GetParseTimeMicroseconds();
    auto us_forward = player_controller->GetForwardTimeMicroseconds();
    auto us_frame = player_controller->GetFrameTimeMicroseconds();
    ImGui::Text("Parse   time: %ld us", us_parse);
    ImGui::Text("Forward time: %ld us", us_forward);
    ImGui::Text("Frame   time: %ld us", us_frame);
    ImGui::End();
}



void RenderModelView(App &app) {
    const auto screen_size = util::GetScreenSize();
    auto &player_controller = app.m_player;
    auto raw_pred = player_controller->GetRawPrediction();
    auto filtered_pred = player_controller->GetFilteredPrediction();
    auto screen_pos = player_controller->GetPosition();

    ImGui::Begin("Render");
    {
        bool v = false;
        v = ImGui::DragInt("Top", &screen_pos.top, 1, 0, screen_size.height) || v;
        v = ImGui::DragInt("Left", &screen_pos.left, 1, 0, screen_size.width) || v;
        if (v) {
            player_controller->SetPosition(screen_pos.top, screen_pos.left);
        }
    }

    ImGui::Text(
        "[raw]      x: %.2f, y: %.2f, confidence: %.2f", 
        raw_pred.x, raw_pred.y, raw_pred.confidence);
    ImGui::Text(
        "[filtered] x: %.2f, y: %.2f, confidence: %.2f", 
        filtered_pred.x, filtered_pred.y, filtered_pred.confidence);

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

    ImGui::DragFloat("Scale", &zoom_scale, 0.001f, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateScreenshotTexture();
    ImGui::Image(
        (void*)app.m_screenshot_texture_view, 
        ImVec2(app.m_texture_width*zoom_scale, app.m_texture_height*zoom_scale));
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

    ImGui::DragFloat("Scale", &zoom_scale, 0.001f, 0.1f, 10.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (app.m_is_render_running) app.UpdateModelTexture();
    ImGui::Image(
        (void*)app.m_model_texture_view, 
        ImVec2(app.m_model_width*zoom_scale, app.m_model_height*zoom_scale));
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
