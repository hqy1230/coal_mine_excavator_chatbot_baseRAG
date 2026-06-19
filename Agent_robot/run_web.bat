@echo off
cd /d "%~dp0"
echo Stopping old api_server on 8765 if present...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8765.*LISTENING"') do (
  taskkill /PID %%a /F >nul 2>&1
)
echo Starting webapp on http://127.0.0.1:8765/
python webapp\main.py
pause
