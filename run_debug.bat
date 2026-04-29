@echo off
REM Same as run.bat but opens the tracker preview window.
cd /d "%~dp0"
"venv\Scripts\python.exe" main.py --debug %*
pause
