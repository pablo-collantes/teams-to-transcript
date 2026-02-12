@echo off
echo ====================================
echo Environment Setup for Teams Recorder
echo ====================================
echo.

echo Setting environment variables...

REM Suppress HuggingFace symlink warning
setx HF_HUB_DISABLE_SYMLINKS_WARNING "1"

echo.
echo Environment variables set successfully!
echo.
echo Note: You may need to restart your terminal for changes to take effect.
echo.

echo ====================================
echo Optional: Hugging Face Token Setup
echo ====================================
echo.
echo To enable faster downloads and higher rate limits:
echo 1. Create a free account at https://huggingface.co/
echo 2. Go to Settings -^> Access Tokens
echo 3. Create a new token (read access is sufficient)
echo 4. Run: setx HF_TOKEN "your_token_here"
echo.

echo ====================================
echo Optional: Enable Developer Mode
echo ====================================
echo.
echo To enable symlinks support (saves disk space):
echo 1. Open Settings -^> Update ^& Security -^> For developers
echo 2. Turn on "Developer Mode"
echo 3. Restart your computer
echo.
echo OR run this PowerShell command as Administrator:
echo reg add "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /t REG_DWORD /f /v "AllowDevelopmentWithoutDevLicense" /d "1"
echo.

pause
