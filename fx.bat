@echo off

:GETOPTS
if /I "%1" == "cmake-conf" call^
 cmake --no-warn-unused-cli^
 -DCMAKE_TOOLCHAIN_FILE:STRING=C:/tools/vcpkg/scripts/buildsystems/vcpkg.cmake^
 -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE^
 -B"build" -G"Visual Studio 16 2019"

if /I "%1" == "build" goto BUILD
if /I "%1" == "help" goto HELP
goto EXIT

:BUILD
set ARGS=/p:CL_MPCOUNT=4 /m:4 /v:minimal /nologo

:: if "%~2" == "" ^
:: echo Usage: build [.vcprojx] [build-type] &^
:: goto EXIT

if "%~2" == "" ^
call msbuild build\main.vcxproj /p:Configuration=Debug %ARGS% &^
goto EXIT

if /I "%3" == ""        call msbuild "%2" /p:Configuration=Debug            %ARGS%
if /I "%3" == "release" call msbuild "%2" /p:Configuration=Release          %ARGS%
if /I "%3" == "debug"   call msbuild "%2" /p:Configuration=Debug            %ARGS%
if /I "%3" == "reldeb"  call msbuild "%2" /p:Configuration=RelWithDebInfo   %ARGS% 
if /I "%3" == "minsize" call msbuild "%2" /p:Configuration=MinSizeRel       %ARGS%
goto EXIT

:HELP
echo Available commands
echo cmake-conf        - Run cmake configure
echo build             - Run msbuild (Usage: build [.vcprojx] [build_type]
goto EXIT

:EXIT
