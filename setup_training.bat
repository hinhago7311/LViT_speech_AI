@echo off
cd /d "%~dp0"
echo This training setup targets the current GTX 1050 Ti using CUDA 11.8.
echo Other GPUs: follow the PyTorch installation section in README.md.
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-export.txt
if errorlevel 1 goto failed
echo.
echo Training setup complete. Start the app with start_app.bat.
echo GPU memory fit still depends on the model and data.
pause
exit /b 0
:failed
echo.
echo Setup failed. Check the messages above and README.md.
pause
exit /b 1
