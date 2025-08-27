# Script to launch the Chess Neural Network game.
Write-Host "Running Chess Neural Network Game..." -ForegroundColor Green

# Change to project root directory
Set-Location -Path (Join-Path $PSScriptRoot "..")

# Executes the main Python script for the chess game.
python src/chess_game.py

Write-Host "`nGame finished. Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
