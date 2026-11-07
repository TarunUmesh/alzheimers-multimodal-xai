@echo off
rem One-time setup: creates a virtual environment in .venv and installs everything.
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 - 3.12 from python.org and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup finished. Next: python tests\smoke_test.py
pause
