@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local .venv is missing.
  echo Run INSTALL_AND_RUN.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
