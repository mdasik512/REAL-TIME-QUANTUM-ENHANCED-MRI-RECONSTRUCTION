$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "Local .venv is missing." -ForegroundColor Red
    Write-Host "Run INSTALL_AND_RUN.ps1 first." -ForegroundColor Yellow
    exit 1
}
& $python app.py
