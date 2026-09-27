$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host ""
Write-Host "QNP + Quantum MRI Observatory" -ForegroundColor Cyan
Write-Host "Project folder: $PSScriptRoot"
Write-Host ""

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local .venv ..." -ForegroundColor Yellow
    py -3.11 -m venv .venv
}

$python = ".\.venv\Scripts\python.exe"

Write-Host "Installing / verifying requirements ..." -ForegroundColor Yellow
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Starting local MRI application ..." -ForegroundColor Green
Write-Host "Open: http://127.0.0.1:8080" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""
& $python app.py
