<#
Install script for setting up PyTorch + PyG for CUDA 11.8 (torch==2.5.1+cu118)

Usage: Run from repository root (where .venv or project workspace exists):
  PowerShell (from repo root):
    & .\scripts\install_pyg_gpu_cu118.ps1

What it does:
  - Activates an existing `./.venv` virtual environment, or offers to create one.
  - Uninstalls conflicting PyTorch/PyG compiled packages.
  - Installs the correct torch trio (cu118) using the PyTorch index.
  - Installs prebuilt PyG compiled extensions that match the chosen torch build.
  - Installs `torch-geometric` and the rest of `requirements.txt`.
  - Runs a basic verification script.

Notes:
  - This script assumes a Windows PowerShell environment.
  - If you prefer conda or CPU-only wheels, choose the relevant install instructions instead.
#>

# --- Safety: ensure working directory is repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Set-Location ..

# --- If there's no venv, offer to create one
$venvPath = Join-Path (Get-Location) '.venv'
if (-Not (Test-Path $venvPath)) {
    Write-Host "Virtual environment not found at $venvPath." -ForegroundColor Yellow
    $create = Read-Host "Create a new venv at '.venv'? (y/n)"
    if ($create -eq 'y' -or $create -eq 'Y') {
        python -m venv .venv
    } else {
        Write-Host 'Exiting - please create/activate the venv and rerun.' -ForegroundColor Red
        exit 1
    }
}

# --- Activate the venv
Write-Host "Activating .venv..."
& .\.venv\Scripts\Activate.ps1

# --- Clean previous compiled and torch installs to avoid conflicts
Write-Host "Removing conflicting packages (if any)" -ForegroundColor Cyan
python -m pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# --- Install the Torch (cu118) wheel trio
Write-Host "Installing Torch (2.5.1+cu118) wheel trio..." -ForegroundColor Green
python -m pip install "torch==2.5.1+cu118" "torchvision==0.20.1+cu118" "torchaudio==2.5.1+cu118" --extra-index-url https://download.pytorch.org/whl/cu118

# --- Install PyG compiled dependencies with the matching wheel repo
Write-Host "Installing PyG compiled dependencies (matching cu118)..." -ForegroundColor Green
python -m pip install torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 -f https://data.pyg.org/whl/torch-2.5.1+cu118.html

# --- Install the final PyG and other requirements
Write-Host "Installing torch-geometric and extras..." -ForegroundColor Green
python -m pip install torch-geometric==2.7.0
python -m pip install -r requirements.txt

# --- Verify using the helper script
if (Test-Path .\scripts\verify_pyg.py) {
    Write-Host "Running verification script..." -ForegroundColor Cyan
    python .\scripts\verify_pyg.py
} else {
    Write-Host 'Verification script not found; please verify manually.' -ForegroundColor Yellow
}

Write-Host "Install/verify script finished. If errors occurred during package installs, scroll up for messages or consider installing with conda if wheel mismatches occur." -ForegroundColor Green
