#include "App.h"

#include <algorithm>
#include <memory>

#include "IModel.h"
#include "SoccerPlayerController.h"
#include "SoccerParams.h"
#include "util/MSS.h"
#include "util/AutoGui.h"
#include "util/KeyListener.h"

void DrawRectInBuffer(
    int cx, int cy, int wx, int wy,
    const RGBA<uint8_t> pen_color, const int pen_width,
    RGBA<uint8_t> *buffer, int width, int height, int row_stride);

App::App(
    std::unique_ptr<IModel>&& model, 
    ID3D11Device *dx11_device, ID3D11DeviceContext *dx11_context)
{
    m_mss = std::make_shared<util::MSS>();
    m_params = std::make_shared<SoccerParams>();
    {
        auto &p = *m_params;
        p.acceleration = 2.5f;
        p.relative_ball_width = 0.24f;
        p.additional_model_delay = 0.0f;
        p.confidence_threshold = 0.5f;
        p.max_lost_frames = 2;

        p.height_trigger_soft = 0.70f;
        p.height_trigger_hard = 0.45f;
        p.fall_speed_trigger_soft = 1.00f;
        p.fall_speed_trigger_hard = 4.00f;
    }

    m_dx11_device = dx11_device;
    m_dx11_context = dx11_context;

    // setup model 
    auto input_buffer = model->GetInputBuffer();
    m_model_width = input_buffer.width;
    m_model_height = input_buffer.height;

    // setup screen shotter
    SetScreenshotSize(320, 455);

    // create a texture for the model resize buffer
    {
        auto res = CreateTexture(m_model_width, m_model_height);
        m_model_texture = res.texture;
        m_model_texture_view = res.view;
    }

    // create the player
    auto player = std::make_shared<SoccerPlayer>(std::move(model), m_mss, m_params);
    m_player = std::make_unique<SoccerPlayerController>(player);

    m_is_render_running = true;
    m_player->SetIsRunning(true);


    // create application bindings
    util::InitGlobalListener();

    util::AttachKeyboardListener(VK_F1, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            bool v = m_player->GetIsRunning();
            m_player->SetIsRunning(!v);
        }
    });

    util::AttachKeyboardListener(VK_F2, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            m_is_render_running = !m_is_render_running;
        }
    });

    util::AttachKeyboardListener(VK_F3, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            bool v = (*m_player)->GetIsTracking();
            (*m_player)->SetIsTracking(!v);
        }
    });

    util::AttachKeyboardListener(VK_F4, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            bool v = (*m_player)->GetIsSmartClicking();
            (*m_player)->SetIsSmartClicking(!v);
        }
    });

    util::AttachKeyboardListener(VK_F5, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            bool v = (*m_player)->GetIsUsingPredictor();
            (*m_player)->SetIsUsingPredictor(!v);
        }
    });

    util::AttachKeyboardListener(VK_F6, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            bool v = (*m_player)->GetIsAlwaysClicking();
            (*m_player)->SetIsAlwaysClicking(!v);
        }
    });

    m_player->SetPosition(316, 799);
}

void App::SetScreenshotSize(const int width, const int height) {
    m_screen_width = width;
    m_screen_height = height;
    m_mss->SetSize(width, height);
    auto res = CreateTexture(width, height);
    m_screenshot_texture = res.texture;
    m_screenshot_texture_view = res.view;
    m_texture_width = width;
    m_texture_height = height;
}

void App::Update() {
    // something do to here? 
}

App::TextureWrapper App::CreateTexture(const int width, const int height) {
    App::TextureWrapper wrapper;
    wrapper.texture = NULL;
    wrapper.view = NULL;

    // Create texture
    // We are creating a dynamic texture
    D3D11_TEXTURE2D_DESC desc;
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DYNAMIC;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    auto status = m_dx11_device->CreateTexture2D(&desc, NULL, &wrapper.texture);

    // Create texture view
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    srvDesc.Texture2D.MostDetailedMip = 0;
    m_dx11_device->CreateShaderResourceView(wrapper.texture, &srvDesc, &wrapper.view);

    return wrapper;
}


void App::UpdateScreenshotTexture() {
    auto bitmap = m_mss->GetBitmap();
    auto buffer_size = bitmap.GetSize();
    auto buffer_max_size = m_mss->GetMaxSize();
    DIBSECTION &sec = bitmap.GetBitmap();
    BITMAP &bmp = sec.dsBm;
    uint8_t *buffer = (uint8_t*)(bmp.bmBits);

    // setup dx11 to modify texture
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    const UINT subresource = 0;
    m_dx11_context->Map(m_screenshot_texture, subresource, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

    // update texture from screen shotter
    int row_width = mappedResource.RowPitch / 4;

    RGBA<uint8_t> *dst_buffer = (RGBA<uint8_t> *)(mappedResource.pData);
    RGBA<uint8_t> *src_buffer = (RGBA<uint8_t> *)(buffer);

    for (int x = 0; x < buffer_size.x; x++) {
        for (int y = 0; y < buffer_size.y; y++) {
            int i = x + y*row_width;
            int j = x + (buffer_max_size.y-y-1)*buffer_max_size.x;
            dst_buffer[i] = src_buffer[j];
        }
    }

    DrawPredictions(dst_buffer, buffer_size.x, buffer_size.y, row_width);
    m_dx11_context->Unmap(m_screenshot_texture, subresource);
}

void App::UpdateModelTexture() {
    uint8_t *buffer = m_player->GetResizeBuffer();
    struct {
        int x, y;
    } buffer_size, buffer_max_size; 
    buffer_size.x = m_model_width;
    buffer_size.y = m_model_height;
    buffer_max_size = buffer_size;

    // setup dx11 to modify texture
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    const UINT subresource = 0;
    m_dx11_context->Map(m_model_texture, subresource, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

    // update texture from screen shotter
    int row_width = mappedResource.RowPitch / 4;

    RGBA<uint8_t> *dst_buffer = (RGBA<uint8_t> *)(mappedResource.pData);
    RGBA<uint8_t> *src_buffer = (RGBA<uint8_t> *)(buffer);

    for (int x = 0; x < buffer_size.x; x++) {
        for (int y = 0; y < buffer_size.y; y++) {
            int i = x + y*row_width;
            int j = x + (buffer_max_size.y-y-1)*buffer_max_size.x;
            dst_buffer[i] = src_buffer[j];
        }
    }

    DrawPredictions(dst_buffer, buffer_size.x, buffer_size.y, row_width);
    m_dx11_context->Unmap(m_model_texture, subresource);
}

void App::DrawPredictions(RGBA<uint8_t>* buf, const int width, const int height, const int row_stride) {
    // render the bounding box of ball prediction
    auto raw_pred = m_player->GetRawPrediction();
    auto filtered_pred = m_player->GetFilteredPrediction();

    int wx = (int)(m_params->relative_ball_width * width * 0.5f);
    int wy = wx;
    const int pen_width = (int)std::max(1.0f, 0.01f*width);
    
    // This is in BGR format
    const RGBA<uint8_t> raw_color = {255,0,0,255}; // blue
    const RGBA<uint8_t> inactive_color = {240,220,5,120}; // light blue
    const RGBA<uint8_t> track_color = {26,140,0,255}; // green
    const RGBA<uint8_t> soft_trigger_color = {0,146,204,255}; // orange
    const RGBA<uint8_t> hard_trigger_color = {0,0,255,255}; // red

    auto& player_controller = *(m_player.get());
    RGBA<uint8_t> pred_color = inactive_color;
    // if we aren't showing the filtered position, use the darker colour for the raw prediction by default
    if (!m_render_overlay_flags.filtered_pred) {
        pred_color = raw_color;
    }
    if (player_controller->GetIsTracking()) {
        pred_color = track_color;
    } 
    if (player_controller->GetIsSoftTrigger()) {
        pred_color = soft_trigger_color;
    }
    if (player_controller->GetIsHardTrigger()) {
        pred_color = hard_trigger_color;
    }

    if (m_render_overlay_flags.raw_pred && (raw_pred.confidence > m_params->confidence_threshold)) {
        int cx = (int)((     raw_pred.x) * width);
        int cy = (int)((1.0f-raw_pred.y) * height);
        
        // show the trigger status colours even if we are only showing the raw prediction
        const RGBA<uint8_t> color = m_render_overlay_flags.filtered_pred ? raw_color : pred_color;
        if (!m_render_overlay_flags.filtered_pred)
        DrawRectInBuffer(
            cx, cy, wx, wy, 
            color, pen_width,
            buf, width, height, row_stride);
    }
    if (m_render_overlay_flags.filtered_pred && (filtered_pred.confidence > m_params->confidence_threshold)) {
        int cx = (int)((     filtered_pred.x) * width);
        int cy = (int)((1.0f-filtered_pred.y) * height);
        DrawRectInBuffer(
            cx, cy, wx, wy, 
            pred_color, pen_width,
            buf, width, height, row_stride);
    }
}

void DrawRectInBuffer(
    int cx, int cy, int wx, int wy,
    const RGBA<uint8_t> pen_color, const int pen_width,
    RGBA<uint8_t> *buffer, int width, int height, int row_stride)
{
    int px_start = std::clamp(cx-wx, 0          , width-pen_width);
    int px_end   = std::clamp(cx+wx, pen_width-1, width-1);
    int py_start = std::clamp(cy-wy, 0          , height-pen_width);
    int py_end   = std::clamp(cy+wy, pen_width-1, height-1);

    // draw each of the line
    for (int x = px_start; x <= px_end; x++) {
        for (int j = 0; j < pen_width; j++) {
            buffer[x + (py_start+j)*row_stride] = pen_color;
            buffer[x + (py_end  -j)*row_stride] = pen_color;
        }
    }
    for (int y = py_start; y <= py_end; y++) {
        for (int j = 0; j < pen_width; j++) {
            buffer[px_start+j + y*row_stride] = pen_color;
            buffer[px_end  -j + y*row_stride] = pen_color;
        }
    }
}