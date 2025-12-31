@echo off
REM Launch Streamlit UI and open the browser.

set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Activate venv if present.
if exist ".venv\\Scripts\\activate.bat" (
    call ".venv\\Scripts\\activate.bat"
)

REM Start Streamlit in a new window so we can open the browser here.
start "Backtest Core Streamlit" cmd /c "streamlit run ui\\app.py"

REM Wait for port 8501, then open the UI.
powershell -NoProfile -Command "$url='http://localhost:8501'; for($i=0; $i -lt 60; $i++){ try{ $tcp = New-Object Net.Sockets.TcpClient; $tcp.Connect('127.0.0.1',8501); $tcp.Close(); Start-Process $url; break } catch{ Start-Sleep -Seconds 1 } }" >nul 2>&1
