@echo off
setlocal

REM ============================================================
REM  Memento Chat one-click launcher
REM  Default: mock mode. Fill Base+Key+model in the web UI settings.
REM ============================================================

cd /d "%~dp0"

echo.
echo   Memento Chat starting...
echo.

REM 1. Check Python
where python >nul 2>nul
if errorlevel 1 (
    echo   [ERROR] python not found. Please install Python 3.10+ and add to PATH.
    pause
    exit /b 1
)

REM 2. Check server deps (fastapi/uvicorn), auto-install if missing
python -c "import fastapi,uvicorn" 2>nul
if errorlevel 1 (
    echo   [First run] Installing server dependencies...
    python -m pip install -r requirements-server.txt
    if errorlevel 1 (
        echo   [ERROR] Install failed. Run manually: pip install -r requirements-server.txt
        pause
        exit /b 1
    )
    echo   Dependencies installed.
    echo.
)

REM 3. Port (default 7878, override via arg)
set PORT=7878
if not "%~1"=="" set PORT=%~1

REM 4. Launch (mock mode; web UI settings can switch to real LLM)
echo   URL:   http://127.0.0.1:%PORT%
echo   Mode:  mock (no LLM). Open the page, click Settings (top-right) to configure.
echo   Press Ctrl+C to stop.
echo.

REM 5. Auto-open browser after 3s
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:%PORT%"

python -m apps.chat.app --mock --port %PORT%

endlocal
