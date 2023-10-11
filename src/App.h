#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

#include <stdint.h>
#include <memory>
#include <thread>

#include "IModel.h"
#include "SoccerPlayer.h"
#include "SoccerParams.h"
#include "util/MSS.h"

class App
{
private:
    struct TextureWrapper {
        ID3D11ShaderResourceView* view;
        ID3D11Texture2D* texture;
    };
private:
    std::unique_ptr<std::thread> m_model_thread; 
    bool m_is_model_thread_running;
public:
    std::shared_ptr<util::MSS> m_mss;
    std::unique_ptr<SoccerPlayer> m_player;
    std::shared_ptr<SoccerParams> m_params;
    
    // model controls
    bool m_is_model_running;
    struct ScreenshotPosition {
        int top = 0;
        int left = 0;
    } m_screenshot_position;
    
    // realtime overlays 
    bool m_is_render_running;
    struct OverlayRender {
        bool raw_pred = true;
        bool filtered_pred = false;
    } m_render_overlay_flags;

    int m_texture_width, m_texture_height;
    int m_screen_width, m_screen_height;
    int m_model_width, m_model_height;

    ID3D11ShaderResourceView* m_screenshot_texture_view;
    ID3D11Texture2D* m_screenshot_texture;

    ID3D11ShaderResourceView* m_model_texture_view;
    ID3D11Texture2D* m_model_texture;

    ID3D11Device *m_dx11_device; 
    ID3D11DeviceContext *m_dx11_context;
public:
    App(std::unique_ptr<IModel>&& model, ID3D11Device *dx11_device, ID3D11DeviceContext *dx11_context);
    ~App();
    void UpdateScreenshotTexture();
    void UpdateModelTexture();
    void SetScreenshotSize(const int width, const int height);
private:
    TextureWrapper CreateTexture(const int width, const int height);
    void DrawPredictions(RGBA<uint8_t>* buf, const int width, const int height, const int row_stride);
};

