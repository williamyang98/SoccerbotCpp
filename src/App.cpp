#include "App.h"

#include <algorithm>
#include <memory>

#include "model.h"
#include "SoccerPlayerController.h"
#include "util/MSS.h"
#include "util/AutoGui.h"
#include "util/KeyListener.h"


App::App(
    std::unique_ptr<Model> &model, 
    ID3D11Device *dx11_device, ID3D11DeviceContext *dx11_context)
{
    m_mss = std::make_shared<util::MSS>();

    m_dx11_device = dx11_device;
    m_dx11_context = dx11_context;

    // setup model 
    auto model_input_size = model->GetInputSize();
    m_model_width = model_input_size.x;
    m_model_height = model_input_size.y;

    // setup screen shotter
    SetScreenshotSize(320, 452);

    // create a texture for the model resize buffer
    {
        auto res = CreateTexture(m_model_width, m_model_height);
        m_model_texture = res.texture;
        m_model_texture_view = res.view;
    }

    // create the player
    m_player = std::make_unique<SoccerPlayerController>(model, m_mss);

    m_is_tracking_ball = false;
    m_is_render_running = true;
    m_player->SetIsRunning(true);
    
    // create application bindings
    util::InitGlobalListener();
    util::AttachKeyboardListener(VK_F3, [this](WPARAM type) {
        if (type == WM_KEYDOWN) {
            m_is_tracking_ball = !m_is_tracking_ball;
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
    // non gui stuff here (for the moment)
    if (m_is_tracking_ball) {
        Model::Result result = m_player->GetResult();
        if (result.confidence > 0.5f) {
            // top and left are mixed up?
            auto pos = m_player->GetPosition();
            int x = pos.left;
            int y = pos.top;

            int dx = (int)(std::floor(result.x * (float)(m_screen_width)));
            int dy = (int)(std::floor(result.y * (float)(m_screen_height)));
            x = x + dx;
            y = y + m_screen_height - dy;
            util::SetCursorPosition(x, y);
        }
    }
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
    m_dx11_context->Map(m_screenshot_texture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

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

    // render the bounding box of ball prediction
    auto result = m_player->GetResult();
    if (result.confidence > 0.5f) {
        int cx = (int)((     result.x) * (float)buffer_size.x);
        int cy = (int)((1.0f-result.y) * (float)buffer_size.y);
        int wx = 50;
        int wy = 50;
        int px_start = std::max(cx-wx, 0);
        int px_end   = std::min(cx+wx, buffer_size.x-1);
        int py_start = std::max(cy-wy, 0);
        int py_end   = std::min(cy+wy, buffer_size.y-1);

        // draw each of the line
        const RGBA<uint8_t> pen_color = {255,0,0,255};
        const int pen_width = 3;
        for (int x = px_start; x <= px_end; x++) {
            for (int j = 0; j < pen_width; j++) {
                dst_buffer[x + (py_start+j)*row_width] = pen_color;
                dst_buffer[x + (py_end  -j)*row_width] = pen_color;
            }
        }
        for (int y = py_start; y <= py_end; y++) {
            for (int j = 0; j < pen_width; j++) {
                dst_buffer[px_start+j + y*row_width] = pen_color;
                dst_buffer[px_end  -j + y*row_width] = pen_color;
            }
        }
    }

    m_dx11_context->Unmap(m_screenshot_texture, 0);
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
    m_dx11_context->Map(m_model_texture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);

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

    // render the bounding box of ball prediction
    auto result = m_player->GetResult();
    if (result.confidence > 0.5f) {
        int cx = (int)((     result.x) * (float)buffer_size.x);
        int cy = (int)((1.0f-result.y) * (float)buffer_size.y);
        int wx = 50 / 4;
        int wy = 50 / 4;
        int px_start = std::max(cx-wx, 0);
        int px_end   = std::min(cx+wx, buffer_size.x-1);
        int py_start = std::max(cy-wy, 0);
        int py_end   = std::min(cy+wy, buffer_size.y-1);

        // draw each of the line
        const RGBA<uint8_t> pen_color = {255,0,0,255};
        const int pen_width = 1;
        for (int x = px_start; x <= px_end; x++) {
            for (int j = 0; j < pen_width; j++) {
                dst_buffer[x + (py_start+j)*row_width] = pen_color;
                dst_buffer[x + (py_end  -j)*row_width] = pen_color;
            }
        }
        for (int y = py_start; y <= py_end; y++) {
            for (int j = 0; j < pen_width; j++) {
                dst_buffer[px_start+j + y*row_width] = pen_color;
                dst_buffer[px_end  -j + y*row_width] = pen_color;
            }
        }
    }

    m_dx11_context->Unmap(m_model_texture, 0);
}