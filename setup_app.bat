@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -r requirements-app.txt
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click start_app.bat to open the app.
echo Training needs the PyTorch setup described in README.md.
pause
exit /b 0
:failed
echo.
echo Setup failed. Install Python 3.11 and check the messages above.
pause
exit /b 1
