param(
    [switch]$WithTorch,
    [switch]$WithoutTorch,
    [string]$PythonBin = "python",
    [string]$VenvDir = "venv"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($WithTorch -and $WithoutTorch) {
    throw "Use only one of -WithTorch or -WithoutTorch."
}

$InstallTorch = "auto"
if ($WithTorch) {
    $InstallTorch = "true"
}
if ($WithoutTorch) {
    $InstallTorch = "false"
}

if (-not (Test-Path $VenvDir)) {
    & $PythonBin -m venv $VenvDir
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Could not find Python inside virtual environment: $VenvPython"
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r requirements.txt

$ShouldInstallTorch = $false
switch ($InstallTorch.ToLowerInvariant()) {
    "true" { $ShouldInstallTorch = $true }
    "false" { $ShouldInstallTorch = $false }
    "auto" {
        & $VenvPython -c "import torch, torchvision" *> $null
        $ShouldInstallTorch = ($LASTEXITCODE -ne 0)
    }
    default { throw "Invalid torch installation mode: $InstallTorch" }
}

if ($ShouldInstallTorch) {
    & $VenvPython -m pip install `
        --index-url https://download.pytorch.org/whl/cpu `
        --extra-index-url https://www.piwheels.org/simple `
        torch

    & $VenvPython -m pip install torchvision --no-cache-dir `
        --index-url https://download.pytorch.org/whl/cpu `
        --extra-index-url https://www.piwheels.org/simple
}

Write-Host ""
Write-Host "Environment setup complete."
Write-Host ""
Write-Host "Activate it with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "For manual module runs, set:"
Write-Host "  `$env:PYTHONPATH = `"$ProjectRoot`""
