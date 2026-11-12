@echo off
setlocal

cd /d "%~dp0"

set "FFMPEG_BIN=%~dp0..\tools\ffmpeg\bin"
if exist "%FFMPEG_BIN%\ffmpeg.exe" set "PATH=%FFMPEG_BIN%;%PATH%"

if not exist "%~dp0venv\Scripts\python.exe" (
    echo Virtual environment not found at "%~dp0venv".
    echo Create it first with: py -3.10 -m venv venv
    pause
    exit /b 1
)

where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo WARNING: ffmpeg not found in PATH. Live streaming to Colab will fail.
)

echo ============================================================
echo  Remote GPU mode (Colab) - local GUI only, no local models
echo ------------------------------------------------------------
echo  1. Fill "In", "Out S", "Out T" with the Colab tunnel URLs
echo     Example: tcp://0.tcp.ap.ngrok.io:5557  (In)
echo              tcp://0.tcp.ap.ngrok.io:5555  (Out S)
echo              tcp://0.tcp.ap.ngrok.io:5556  (Out T)
echo  2. Select a face (source) and a target
echo  3. Turn ON the "Remote Processor" switch
echo  4. Click "Live"; press q or x in the preview window or close the Preview window to stop
echo ============================================================
echo.

"%~dp0venv\Scripts\python.exe" run.py --execution-provider cpu

pause
