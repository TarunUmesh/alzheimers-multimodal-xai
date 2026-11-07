@echo off
rem Starts the web app in your browser.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
