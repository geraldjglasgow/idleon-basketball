@echo off
REM Same as run.bat but opens a small window with just the green debug text.
cd /d "%~dp0"
"venv\Scripts\python.exe" main.py --light %*
pause
