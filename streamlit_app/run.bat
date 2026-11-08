@echo off
REM Road Anomaly Detector - installs the packages (first run only) and opens http://localhost:8501
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  if exist "%USERPROFILE%\anaconda3\python.exe" (set "PATH=%USERPROFILE%\anaconda3;%USERPROFILE%\anaconda3\Scripts;%PATH%") else (
    echo Python was not found. Install Python 3.10+ or Anaconda, then run this file again.
    pause & exit /b 1 )
)
if not exist ".venv\Scripts\python.exe" (
  echo Creating a virtual environment and installing packages, this takes 1-2 minutes the first time...
  python -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package installation failed. & pause & exit /b 1)
)
echo Starting the app at http://localhost:8501  (close this window to stop it)
".venv\Scripts\python.exe" -m streamlit run app.py
pause
