[CmdletBinding()]
param(
    [string]$MindVisionRoot = 'G:\mindvision'
)

$ErrorActionPreference = 'Stop'

$runtimeVlm = 'dist\SwallowabilityConsole\config\vlm.yaml'
$runtimeInference = 'dist\SwallowabilityConsole\config\inference.yaml'
$runtimeCalibration = 'dist\SwallowabilityConsole\data\calibration\calibration_new.json'
$runtimeBackground = 'dist\SwallowabilityConsole\runs\measurement\background.jpg'
$backups = @{}

function Backup-RuntimeFile {
    param([string]$Source)
    if (-not (Test-Path -LiteralPath $Source)) {
        return
    }
    $backup = Join-Path $env:TEMP ("swallowability_build_" + [guid]::NewGuid().ToString("N") + [IO.Path]::GetExtension($Source))
    Copy-Item -LiteralPath $Source -Destination $backup -Force
    $script:backups[$Source] = $backup
}

function Restore-RuntimeFile {
    param([string]$Destination, [string]$Backup)
    if (-not (Test-Path -LiteralPath $Backup)) {
        return
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $Backup -Destination $Destination -Force
    Remove-Item -LiteralPath $Backup -Force -ErrorAction SilentlyContinue
}

Backup-RuntimeFile $runtimeVlm
Backup-RuntimeFile $runtimeInference
Backup-RuntimeFile $runtimeCalibration
Backup-RuntimeFile $runtimeBackground

if (-not (Test-Path -LiteralPath 'best.pt')) {
    throw '缺少本地模型 best.pt'
}

try {
    python -m PyInstaller --noconfirm --clean SwallowabilityConsole.spec
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller 构建失败，退出码：$LASTEXITCODE"
    }

    # Keep editable runtime assets beside the EXE. The GUI checks this directory
    # first, so camera and model settings can be changed without rebuilding.
    Copy-Item -Recurse -Force config dist\SwallowabilityConsole\config
    Copy-Item -Force best.pt dist\SwallowabilityConsole\best.pt

    # Never ship a plaintext API key with the local YOLO build. The API backend
    # can still read VLM_API_KEY from the environment if it is enabled later.
    $distInference = 'dist\SwallowabilityConsole\config\inference.yaml'
    $distVlm = 'dist\SwallowabilityConsole\config\vlm.yaml'
    if ((Get-Content -Raw -LiteralPath $distInference) -match '(?m)^backend:\s*["'']?yolo["'']?\s*$') {
        $vlmText = Get-Content -Raw -LiteralPath $distVlm
        $vlmText = [regex]::Replace($vlmText, '(?m)^api_key\s*:.*$', 'api_key: ""')
        [IO.File]::WriteAllText((Resolve-Path $distVlm), $vlmText, [Text.UTF8Encoding]::new($false))
    }

    # Runtime calibration and background are setup-specific, editable assets.
    New-Item -ItemType Directory -Force -Path dist\SwallowabilityConsole\data\calibration | Out-Null
    Copy-Item -Force data\calibration\calibration_new.json dist\SwallowabilityConsole\data\calibration\calibration_new.json

    New-Item -ItemType Directory -Force -Path dist\SwallowabilityConsole\runs\measurement | Out-Null
    Copy-Item -Force runs\measurement\background.jpg dist\SwallowabilityConsole\runs\measurement\background.jpg
    # Bundle the minimal MindVision SDK beside the EXE so relative sdk_path works.
    $mindvisionPython = Join-Path $MindVisionRoot 'Demo\Python\Basic\mvsdk.py'
    $mindvisionDll = Join-Path $MindVisionRoot 'SDK\X64\MVCAMSDK_X64.dll'
    if (-not (Test-Path -LiteralPath $mindvisionPython) -or -not (Test-Path -LiteralPath $mindvisionDll)) {
        throw "缺少迈德威视 SDK：$MindVisionRoot"
    }
    New-Item -ItemType Directory -Force -Path dist\SwallowabilityConsole\mindvision\Demo\Python | Out-Null
    New-Item -ItemType Directory -Force -Path dist\SwallowabilityConsole\mindvision\SDK | Out-Null
    Copy-Item -Recurse -Force (Join-Path $MindVisionRoot 'Demo\Python\Basic') dist\SwallowabilityConsole\mindvision\Demo\Python\Basic
    Copy-Item -Recurse -Force (Join-Path $MindVisionRoot 'SDK\X64') dist\SwallowabilityConsole\mindvision\SDK\X64
}
finally {
    foreach ($entry in $backups.GetEnumerator()) {
        Restore-RuntimeFile -Destination $entry.Key -Backup $entry.Value
    }
}

Write-Host "完成：dist\SwallowabilityConsole\SwallowabilityConsole.exe"
