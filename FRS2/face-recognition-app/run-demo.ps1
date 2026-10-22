$ErrorActionPreference = 'Stop'

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $project

if (Test-Path '.venv\Scripts\Activate.ps1') {
    . '.venv\Scripts\Activate.ps1'
}

Write-Host 'Starting FaceVerify at http://127.0.0.1:5050'
Write-Host 'Press Ctrl+C to stop the application.'
python -m demo.demo_backend
