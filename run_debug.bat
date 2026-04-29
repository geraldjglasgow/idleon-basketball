@echo off
REM Launch the basketball bot with full game stream + tracker overlay.
cd /d "%~dp0"
"venv\Scripts\python.exe" game.py --mode debug %*
pause
