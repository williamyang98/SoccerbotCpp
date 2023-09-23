#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

#include <memory>

#include "IModel.h"
#include "SoccerPlayerController.h"
#include "SoccerParams.h"
#include "util/MSS.h"

class App
{
private:
    struct TextureWrapper {
        ID3D11ShaderResourceView* view;
        ID3D11Texture2D* texture;
    };
public:
    std::shared_ptr<util::MSS> m_mss;
    std::unique_ptr<SoccerPlayerController> m_player;
    std::shared_ptr<SoccerParams> m_params;
    
    bool m_is_render_running;
    
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
    App(std::unique_ptr<IModel>&& model, 
        ID3D11Device *dx11_device, ID3D11DeviceContext *dx11_context);
    void Update();
    void UpdateScreenshotTexture();
    void UpdateModelTexture();
    void SetScreenshotSize(const int width, const int height);
private:
    TextureWrapper CreateTexture(const int width, const int height);
};

