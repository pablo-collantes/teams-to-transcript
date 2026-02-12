@echo off
echo ====================================
echo Teams Recorder Installation Script
echo ====================================
echo.

REM Delete old venv if exists
if exist .venv (
    echo Removing old virtual environment...
    rmdir /s /q .venv
)

echo Creating new virtual environment...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo Installing main dependencies...
pip install -r requirements.txt

echo.
echo ====================================
echo Installing PyAudio (this may take a moment)...
echo ====================================
echo Trying pip install first...
pip install pyaudio

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo PyAudio pip install failed. Trying pipwin method...
    pip install pipwin
    pipwin install pyaudio

    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ====================================
        echo WARNING: PyAudio installation failed!
        echo ====================================
        echo Please install PyAudio manually:
        echo 1. Visit: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
        echo 2. Download the appropriate .whl file for Python 3.12
        echo 3. Run: pip install [downloaded-file].whl
        echo.
    )
)

echo.
echo ====================================
echo Installation Complete!
echo ====================================
echo.
echo To run the application:
echo 1. cd src
echo 2. python main.py
echo.
pause
