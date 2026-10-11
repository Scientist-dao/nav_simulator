@echo off
rem navsim launcher -> navsim.py (auto-fills --scenario / default --backend rowmap)
chcp 65001 >nul
"%~dp0.venv\Scripts\python.exe" "%~dp0navsim.py" %*
