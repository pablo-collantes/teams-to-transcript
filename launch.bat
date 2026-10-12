@echo off
rem Launcher for Teams Call Transcriber (Windows).
rem First run creates a virtual environment and installs dependencies.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First run: creating virtual environment and installing dependencies...
    echo This can take a few minutes.
    py -3 -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

".venv\Scripts\python.exe" src\main.py %*
goto :eof

:error
echo.
echo Setup failed. Install Python 3.10+ from https://www.python.org/downloads/
echo (check "Add python.exe to PATH" during install), then run this again.
pause
