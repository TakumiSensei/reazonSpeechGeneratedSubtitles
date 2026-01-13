@echo off
cd /d %~dp0
call venv\Scripts\activate
python transcribe_k2v2.py
pause
