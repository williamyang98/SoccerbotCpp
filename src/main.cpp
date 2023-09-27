#include <exception>
#include <stdio.h>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>

#include <argparse/argparse.hpp>
#include <string>
#include "App.h"
#include "gui.h"
#include "IModel.h"
#include "TensorflowLiteModel.h"
#include "OnnxDirectMLModel.h"

int run_app(std::unique_ptr<IModel>&& pModel);

// Main code
int _main(int argc, char** argv) {
    auto parser = argparse::ArgumentParser("SoccerBot Inference Application", "2.0.0");
    parser.add_argument("--model")
        .default_value(std::string("./models/cnn_113_80_quantized.tflite"))
        .required()
        .help("Path to model to load");
    parser.add_argument("--runtime")
        .default_value(std::string("tflite"))
        .required()
        .help("Type of runtime for model. Options: [onnx, tflite]");
    parser.add_argument("--tflite-cpus")
        .default_value(1)
        .scan<'i', int>()
        .required()
        .help("Number of threads for tflite to use. If 0 is provided then number of logical processors is used.");
    parser.add_argument("--onnx-device")
        .default_value(std::string("directml"))
        .required()
        .help("Type of device for onnx runtime to use. Options: [directml, cpu]");
    parser.add_argument("--onnx-directml-gpu-id")
        .default_value(0)
        .scan<'i', int>()
        .required()
        .help("GPU ID to use for onnx directml backend");
    parser.add_argument("--onnx-cpu-threads")
        .default_value(2)
        .scan<'i', int>()
        .required()
        .help("Number of threads for use for CPU inference. If 0 is provided then onnx will optimise for best performance.");
    parser.add_argument("--onnx-cpu-sequential")
        .default_value(false)
        .implicit_value(true)
        .help("Sets onnx cpu backend to run the model sequentially");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& ex) {
        std::cerr << ex.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    
    auto model_path = parser.get<std::string>("--model");
    printf("Loading model: %s\n", model_path.c_str());

    auto runtime_type = parser.get<std::string>("--runtime");
    bool is_onnx = true;
    if (runtime_type.compare("onnx") == 0) {
        is_onnx = true;
    } else if (runtime_type.compare("tflite") == 0) {
        is_onnx = false;
    } else {
        std::cerr << "Invalid runtime selected: " << runtime_type << std::endl;
        std::cerr << parser;
        return 1;
    }
    std::cout << "Selected backend: " << runtime_type << std::endl;

    std::unique_ptr<IModel> pModel = nullptr;
    if (is_onnx) {
        auto onnx_device = parser.get<std::string>("--onnx-device");
        if (onnx_device.compare("cpu") == 0) {
            std::cout << "Selected onnx CPU backend" << std::endl;
            int total_threads = parser.get<int>("--onnx-cpu-threads");
            if (total_threads != 0) {
                std::cout << "Using " << total_threads << " threads for CPU inference" << std::endl;
            } else {
                std::cout << "Letting onnx optimise the number of CPU threads" << std::endl;
            }

            bool is_sequential = parser.get<bool>("--onnx-cpu-sequential");
            std::cout << "CPU backend is running in parallel: " << !is_sequential << std::endl;

            auto opts = OnnxDirectMLModel::CPU_Options{};
            opts.total_threads = total_threads;
            opts.is_sequential = is_sequential;
            pModel = std::make_unique<OnnxDirectMLModel>(model_path.c_str(), opts);
        } else if (onnx_device.compare("directml") == 0) {
            const int gpu_id = parser.get<int>("--onnx-directml-gpu-id");
            std::cout << "Selected onnx DirectML backend GPU:" << gpu_id << std::endl;

            auto opts = OnnxDirectMLModel::GPU_Options{};
            opts.device_id = gpu_id;
            pModel = std::make_unique<OnnxDirectMLModel>(model_path.c_str(), opts);
        } else {
            std::cerr << "Invalid onnx device: " << onnx_device << std::endl;
            return 1;
        }
    } else {
        int total_threads = parser.get<int>("--tflite-cpus");
        if (total_threads == 0) {
            total_threads = std::thread::hardware_concurrency();
        }
        std::cout << "Select tflite backend with " << total_threads << " CPU threads" << std::endl;
        pModel = std::make_unique<TensorflowLiteModel>(model_path.c_str(), total_threads);
    }

    pModel->PrintSummary();
    return run_app(std::move(pModel));
}

// Release mode builds don't have an exception output window
int main(int argc, char** argv) {
    try {
        return _main(argc, argv);
    } catch (std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << std::endl;
        return 1;
    }
}

// Dear ImGui: standalone example application for DirectX 11
// If you are new to Dear ImGui, read documentation from the docs/ folder + read the top of imgui.cpp.
// Read online: https://github.com/ocornut/imgui/tree/master/docs
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
// DX11 variables for imgui backend
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <tchar.h>

// Data
static ID3D11Device*            g_pd3dDevice = NULL;
static ID3D11DeviceContext*     g_pd3dDeviceContext = NULL;
static IDXGISwapChain*          g_pSwapChain = NULL;
static ID3D11RenderTargetView*  g_mainRenderTargetView = NULL;
// Forward declarations of helper functions
bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

int run_app(std::unique_ptr<IModel>&& pModel) {
    // Create application window
    //ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("SoccerBot"), NULL };
    ::RegisterClassEx(&wc);
    HWND hwnd = ::CreateWindow(wc.lpszClassName, _T("SoccerBot"), WS_OVERLAPPEDWINDOW, 100, 100, 800, 650, NULL, NULL, wc.hInstance, NULL);

    // Initialize Direct3D
    if (!CreateDeviceD3D(hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return 1;
    }

    // Show the window
    ::ShowWindow(hwnd, SW_SHOWDEFAULT);
    ::UpdateWindow(hwnd);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
    //io.ConfigViewportsNoAutoMerge = true;
    //io.ConfigViewportsNoTaskBarIcon = true;
    //io.ConfigViewportsNoDefaultParent = true;
    //io.ConfigDockingAlwaysTabBar = true;
    io.ConfigDockingTransparentPayload = true;
    io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleFonts;     // FIXME-DPI: Experimental. THIS CURRENTLY DOESN'T WORK AS EXPECTED. DON'T USE IN USER APP!
    io.ConfigFlags |= ImGuiConfigFlags_DpiEnableScaleViewports; // FIXME-DPI: Experimental.

    // Setup Dear ImGui style
    // ImGui::StyleColorsDark();
    ImGui::StyleColorsLight();
    //ImGui::StyleColorsClassic();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    // Load Fonts
    // - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
    // - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
    // - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
    // - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
    // - Read 'docs/FONTS.md' for more instructions and details.
    // - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
    //io.Fonts->AddFontDefault();
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
    //IM_ASSERT(font != NULL);

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // create app after setting up the dx11 context
    auto main_app = App(std::move(pModel), g_pd3dDevice, g_pd3dDeviceContext);

    // Main loop
    bool done = false;
    while (!done) {
        MSG msg;
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE)) {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT) {
                done = true;
            }
        }

        if (done) {
            break;
        }

        // Start the Dear ImGui frame
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        // render our app
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
        RenderApp(main_app);

        // Rendering
        ImGui::Render();
        const float clear_color_with_alpha[4] = { clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
        }

        g_pSwapChain->Present(1, 0); // Present with vsync
        //g_pSwapChain->Present(0, 0); // Present without vsync
    }

    // Cleanup
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(hwnd);
    ::UnregisterClass(wc.lpszClassName, wc.hInstance);

    return 0;
}

// Helper functions
bool CreateDeviceD3D(HWND hWnd)
{
    // Setup swap chain
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 120;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    UINT createDeviceFlags = 0;
    //createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0, };
    if (D3D11CreateDeviceAndSwapChain(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, createDeviceFlags, featureLevelArray, 2, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext) != S_OK)
        return false;

    CreateRenderTarget();
    return true;
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

#ifndef WM_DPICHANGED
#define WM_DPICHANGED 0x02E0 // From Windows SDK 8.1+ headers
#endif

// Forward declare message handler from imgui_impl_win32.cpp
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Win32 message handler
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, (UINT)LOWORD(lParam), (UINT)HIWORD(lParam), DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;
    case WM_SYSCOMMAND:
        if ((wParam & 0xfff0) == SC_KEYMENU) // Disable ALT application menu
            return 0;
        break;
    case WM_DESTROY:
        ::PostQuitMessage(0);
        return 0;
    case WM_DPICHANGED:
        if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_DpiEnableScaleViewports)
        {
            //const int dpi = HIWORD(wParam);
            //printf("WM_DPICHANGED to %d (%.0f%%)\n", dpi, (float)dpi / 96.0f * 100.0f);
            const RECT* suggested_rect = (RECT*)lParam;
            ::SetWindowPos(hWnd, NULL, suggested_rect->left, suggested_rect->top, suggested_rect->right - suggested_rect->left, suggested_rect->bottom - suggested_rect->top, SWP_NOZORDER | SWP_NOACTIVATE);
        }
        break;
    }
    return ::DefWindowProc(hWnd, msg, wParam, lParam);
}
