<#
.SYNOPSIS
  Clone, build Draco in Release mode, build the Python wheel and install it.
#>

param(
  [string]$DracoRepo = "https://github.com/google/draco.git",
  [string]$SourceDir = "producer\deps\draco_src",
  [string]$BuildDir  = "producer\build\draco_build",
  [string]$InstallPrefix = "$(Resolve-Path .)\draco_install"
)

# 1) Clone or update
if (Test-Path $SourceDir) {
    Write-Host "Updating existing Draco repo..."
    Push-Location $SourceDir
    git pull
    Pop-Location
} else {
    Write-Host "Cloning Draco..."
    git clone $DracoRepo $SourceDir
}

# 2) Configure CMake in Release
Write-Host "Configuring CMake Release build..."
if (-Not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
Push-Location $BuildDir
cmake "..\..\..\$SourceDir" `
  -DCMAKE_BUILD_TYPE=Release `
  -DBUILD_SHARED_LIBS=ON `
  -DBUILD_PYTHON_BINDINGS=ON `
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON `
  -DCMAKE_INSTALL_PREFIX="$InstallPrefix" `
  -DCMAKE_CXX_FLAGS="/O2 /DNDEBUG"
  
# 3) Build & install
Write-Host "Building and installing Draco..."
cmake --build . --config Release --target install
Pop-Location

# 4) Build & install the Python wheel
$pyDir = Join-Path $SourceDir "bindings\python"
Write-Host "Building Python wheel..."
Push-Location $pyDir

# (activate your venv first if you like)
python -m pip install --upgrade pip setuptools wheel
python setup.py bdist_wheel

# uninstall old, install new wheel
python -m pip uninstall -y DracoPy 
python -m pip install dist\DracoPy-*.whl

Pop-Location

Write-Host "✅ DracoPy Release build and install complete!"