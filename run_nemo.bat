@echo off
cd /d %~dp0
call venv\Scripts\activate
python transcribe_nemo.py
pause
