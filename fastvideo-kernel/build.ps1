# PowerShell build script for fastvideo-kernel on Windows.
# Mirrors build.sh; see docs/getting_started/installation/windows.md.
#
# Env overrides: TORCH_CUDA_ARCH_LIST, CMAKE_ARGS, CMAKE_GENERATOR.

[CmdletBinding()]
param(
    [ValidateSet('CUDA')]
    [string]$GpuBackend = 'CUDA'
)

$ErrorActionPreference = 'Stop'

function Write-Info { param([string]$Message) Write-Host $Message -ForegroundColor Cyan }

function Format-EnvValue {
    param([string]$Value)
    if ([string]::IsNullOrEmpty($Value)) { return '<unset>' }
    return $Value
}

# If cl.exe isn't on PATH, locate a VS install via vswhere and enter its
# dev shell in-process so CMake can find the toolchain.
function Enter-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        Write-Info "MSVC toolchain already on PATH (cl.exe found)."
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw "cl.exe not found on PATH and vswhere.exe is missing at '$vswhere'. " +
              "Install Visual Studio 2019+ with the 'Desktop development with C++' workload, " +
              "or launch this script from a Developer PowerShell for VS."
    }

    $vsInstallDir = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsInstallDir) {
        throw "vswhere could not locate a Visual Studio install with the C++ tools component."
    }

    $devShellModule = Join-Path $vsInstallDir 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    if (-not (Test-Path $devShellModule)) {
        throw "Found VS install at '$vsInstallDir' but DevShell module is missing: '$devShellModule'."
    }

    Write-Info "Entering VS developer shell from: $vsInstallDir"
    Import-Module $devShellModule
    Enter-VsDevShell -VsInstallPath $vsInstallDir -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "Entered VS dev shell but cl.exe is still not on PATH. Check your VS install."
    }
}

function Get-ActivePython {
    if ($env:VIRTUAL_ENV) {
        $p = Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe'
        if (Test-Path $p) { return $p }
    }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command python3 -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "No python interpreter found. Activate your venv or install Python."
}

function Get-CudaComputeCapability {
    param([string]$Python)

    # Write the probe to a temp file rather than passing via `python -c`.
    # PowerShell's native-command layer mangles multi-line strings containing
    # embedded double-quotes (e.g. f-strings), which breaks `-c` invocations.
    $code = @'
import sys
try:
    import torch
except Exception as e:
    sys.stderr.write('import torch failed: ' + repr(e) + '\n')
    sys.exit(2)
if not torch.cuda.is_available():
    sys.stderr.write('torch.cuda.is_available() is false\n')
    sys.exit(3)
mj, mn = torch.cuda.get_device_capability(0)
sys.stdout.write(str(mj) + '.' + str(mn) + '\n')
'@

    $scriptPath = Join-Path ([System.IO.Path]::GetTempPath()) ("fv_detect_cc_" + [System.Guid]::NewGuid().ToString('N') + '.py')
    try {
        Set-Content -Path $scriptPath -Value $code -Encoding utf8
        $out = & $Python $scriptPath
        if ($LASTEXITCODE -ne 0) {
            throw "torch-based CUDA arch detection failed (exit $LASTEXITCODE). " +
                  "Ensure torch is installed in the active environment and a CUDA GPU is visible."
        }
        return ($out | Out-String).Trim()
    } finally {
        if (Test-Path $scriptPath) { Remove-Item $scriptPath -Force -ErrorAction SilentlyContinue }
    }
}

function Test-CMakeArg {
    param([string]$Key)
    if (-not $env:CMAKE_ARGS) { return $false }
    return $env:CMAKE_ARGS -match "(^|\s)-D${Key}(=|$)"
}

Write-Info "Building fastvideo-kernel..."

Enter-MsvcEnvironment

Write-Info "Initializing submodules..."
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "git submodule update failed." }

Write-Info "Installing build dependencies (scikit-build-core, cmake, ninja)..."
uv pip install scikit-build-core cmake ninja
if ($LASTEXITCODE -ne 0) { throw "uv pip install of build deps failed." }

$extraCMakeArgs = @()

if ($GpuBackend -eq 'CUDA') {
    $python = Get-ActivePython
    Write-Info "Using Python: $python"

    $detected_cc = Get-CudaComputeCapability -Python $python
    $cc_parts = $detected_cc -split '\.'
    if ($cc_parts.Count -lt 2) {
        throw "Unexpected compute-capability format from torch: '$detected_cc'"
    }
    $cc_major = $cc_parts[0]
    $cc_minor = $cc_parts[1]
    $cmake_arch = "${cc_major}${cc_minor}"
    $isHopper = ($cc_major -eq '9' -and $cc_minor -eq '0')
    Write-Info "Detected compute capability via torch: $detected_cc (sm_$cmake_arch)"

    if (-not $env:TORCH_CUDA_ARCH_LIST) {
        $env:TORCH_CUDA_ARCH_LIST = if ($isHopper) { '9.0a' } else { "${cc_major}.${cc_minor}" }
    }

    # ThunderKittens build targeting:
    #   SM90 -> compile Hopper/TK kernels with 90a.
    #   Else -> compile non-TK path with detected arch.
    if (-not (Test-CMakeArg 'CMAKE_CUDA_ARCHITECTURES')) {
        $arch = if ($isHopper) { '90a' } else { $cmake_arch }
        $extraCMakeArgs += "-DCMAKE_CUDA_ARCHITECTURES=$arch"
    }

    if (-not (Test-CMakeArg 'FASTVIDEO_KERNEL_BUILD_TK')) {
        $tk = if ($isHopper) { 'ON' } else { 'OFF' }
        $extraCMakeArgs += "-DFASTVIDEO_KERNEL_BUILD_TK=$tk"
    }
}

if (-not (Test-CMakeArg 'GPU_BACKEND')) {
    $extraCMakeArgs += "-DGPU_BACKEND=$GpuBackend"
}

if ($extraCMakeArgs.Count -gt 0) {
    $joined = $extraCMakeArgs -join ' '
    if ($env:CMAKE_ARGS) {
        $env:CMAKE_ARGS = "$($env:CMAKE_ARGS.Trim()) $joined"
    } else {
        $env:CMAKE_ARGS = $joined
    }
}

# Prefer Ninja on Windows: MSBuild serializes poorly and mishandles long
# command-lines from CUDA compiles.
if (-not $env:CMAKE_GENERATOR) {
    $env:CMAKE_GENERATOR = 'Ninja'
}

Write-Info "TORCH_CUDA_ARCH_LIST: $(Format-EnvValue $env:TORCH_CUDA_ARCH_LIST)"
Write-Info "CMAKE_ARGS:           $(Format-EnvValue $env:CMAKE_ARGS)"
Write-Info "CMAKE_GENERATOR:      $(Format-EnvValue $env:CMAKE_GENERATOR)"
Write-Info "GPU_BACKEND:          $GpuBackend"

Write-Info "Building and installing fastvideo-kernel (this may take several minutes)..."
uv pip install . -v --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw "uv pip install of fastvideo-kernel failed." }

Write-Host "Build complete!" -ForegroundColor Green
