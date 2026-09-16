[CmdletBinding()]
param(
    [string]$MindVisionRoot = 'G:\mindvision',
    [string]$OutputRoot = '',
    [string]$DistRootPath = '',
    [switch]$SkipBuild,
    [switch]$NoZip
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot 'release'
}
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$ReleaseName = 'SwallowabilityConsole_OneClick'
$ReleaseRoot = [IO.Path]::GetFullPath((Join-Path $OutputRoot $ReleaseName))
if ([string]::IsNullOrWhiteSpace($DistRootPath)) {
    $DistRoot = Join-Path $ProjectRoot 'dist\SwallowabilityConsole'
} elseif ([IO.Path]::IsPathRooted($DistRootPath)) {
    $DistRoot = [IO.Path]::GetFullPath($DistRootPath)
} else {
    $DistRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $DistRootPath))
}
$BuildScript = Join-Path $ProjectRoot 'build_exe.ps1'

function Assert-SafeChildPath {
    param([string]$Parent, [string]$Child)
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $childFull = [IO.Path]::GetFullPath($Child).TrimEnd('\', '/')
    $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
    if (-not $childFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝操作发布目录之外的路径：$childFull"
    }
}

function Copy-Tree {
    param([string]$Source, [string]$Destination)
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "缺少发布源文件或目录：$Source"
    }
    $parent = Split-Path -Parent $Destination
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
}

function Assert-RequiredFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "发布文件缺失：$Path"
    }
}

if (-not $SkipBuild) {
    Write-Host '[1/6] 构建最新 EXE...'
    & $BuildScript
    if ($LASTEXITCODE -ne 0) {
        throw "EXE 构建失败，退出码：$LASTEXITCODE"
    }
} else {
    Write-Host '[1/6] 跳过构建，使用现有 dist 产物。'
}

Assert-RequiredFile (Join-Path $DistRoot 'SwallowabilityConsole.exe')
Assert-RequiredFile (Join-Path $DistRoot '_internal\base_library.zip')
Assert-RequiredFile (Join-Path $DistRoot 'config\vlm.yaml')
Assert-RequiredFile (Join-Path $DistRoot 'data\calibration\calibration_new.json')
Assert-RequiredFile (Join-Path $DistRoot 'runs\measurement\background.jpg')

$profileText = Get-Content -Raw -LiteralPath (Join-Path $DistRoot 'config\camera_profile.yaml')
if ($profileText -notmatch '(?m)^sdk_path:\s*["'']?mindvision["'']?\s*$') {
    throw 'dist 中的 camera_profile.yaml 仍不是相对 SDK 路径 mindvision，已停止打包。'
}

$apiKeyLine = Get-Content -LiteralPath (Join-Path $DistRoot 'config\vlm.yaml') |
    Where-Object { $_ -match '^\s*api_key\s*:' } |
    Select-Object -First 1
$apiKey = ($apiKeyLine -replace '^\s*api_key\s*:\s*', '').Trim().Trim('"').Trim("'")
if ($apiKey.Length -lt 20) {
    throw 'dist 中的 config/vlm.yaml 未包含可用密钥，已停止打包。'
}

Assert-RequiredFile (Join-Path $ProjectRoot 'install.ps1')
Assert-RequiredFile (Join-Path $ProjectRoot 'install.bat')
Assert-RequiredFile (Join-Path $ProjectRoot 'README_安装说明.txt')
Assert-RequiredFile (Join-Path $MindVisionRoot 'MVDCP2_Setup.exe')
Assert-RequiredFile (Join-Path $MindVisionRoot 'Demo\Python\Basic\mvsdk.py')
Assert-RequiredFile (Join-Path $MindVisionRoot 'SDK\X64\MVCAMSDK_X64.dll')
Assert-RequiredFile (Join-Path $MindVisionRoot 'Drivers\USB\WIN10\MvU2Camera.inf')
Assert-RequiredFile (Join-Path $MindVisionRoot 'Drivers\GIGE\WIN10\60\MvGigeDriverLwf.inf')

Assert-SafeChildPath -Parent $OutputRoot -Child $ReleaseRoot
if (Test-Path -LiteralPath $ReleaseRoot) {
    Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null
$AppTarget = Join-Path $ReleaseRoot 'SwallowabilityConsole'

Write-Host '[2/6] 复制程序、API 配置、标定文件和背景图...'
Copy-Tree -Source $DistRoot -Destination $AppTarget
# Do not ship previous analysis results, captures, or error logs.
foreach ($relative in @('runs\analysis', 'runs\captures', 'runs\gui_error.log', 'runs\analysis_error.log')) {
    $generated = Join-Path $AppTarget $relative
    if (Test-Path -LiteralPath $generated) {
        Assert-SafeChildPath -Parent $AppTarget -Child $generated
        Remove-Item -LiteralPath $generated -Recurse -Force
    }
}

Write-Host '[3/6] 复制迈德威视最小运行库...'
$mvTarget = Join-Path $AppTarget 'mindvision'
$mvBasicTarget = Join-Path $mvTarget 'Demo\Python\Basic'
$mvSdkTarget = Join-Path $mvTarget 'SDK\X64'
if (-not (Test-Path -LiteralPath (Join-Path $mvBasicTarget 'mvsdk.py'))) {
    Copy-Tree -Source (Join-Path $MindVisionRoot 'Demo\Python\Basic') -Destination $mvBasicTarget
}
if (-not (Test-Path -LiteralPath (Join-Path $mvSdkTarget 'MVCAMSDK_X64.dll'))) {
    Copy-Tree -Source (Join-Path $MindVisionRoot 'SDK\X64') -Destination $mvSdkTarget
}

Write-Host '[4/6] 复制相机驱动安装器和 INF 驱动...'
$driverTarget = Join-Path $ReleaseRoot 'camera_driver'
Copy-Tree -Source (Join-Path $MindVisionRoot 'MVDCP2_Setup.exe') -Destination (Join-Path $driverTarget 'MVDCP2_Setup.exe')
Copy-Tree -Source (Join-Path $MindVisionRoot 'Drivers') -Destination (Join-Path $driverTarget 'Drivers')
Copy-Tree -Source (Join-Path $ProjectRoot 'install.ps1') -Destination (Join-Path $ReleaseRoot 'install.ps1')
Copy-Tree -Source (Join-Path $ProjectRoot 'install.bat') -Destination (Join-Path $ReleaseRoot 'install.bat')
Copy-Tree -Source (Join-Path $ProjectRoot 'README_安装说明.txt') -Destination (Join-Path $ReleaseRoot 'README_安装说明.txt')

Write-Host '[5/6] 校验发布文件和相对路径...'
Assert-RequiredFile (Join-Path $AppTarget 'SwallowabilityConsole.exe')
Assert-RequiredFile (Join-Path $AppTarget 'config\vlm.yaml')
Assert-RequiredFile (Join-Path $AppTarget 'mindvision\Demo\Python\Basic\mvsdk.py')
Assert-RequiredFile (Join-Path $AppTarget 'mindvision\SDK\X64\MVCAMSDK_X64.dll')
Assert-RequiredFile (Join-Path $ReleaseRoot 'install.ps1')
Assert-RequiredFile (Join-Path $ReleaseRoot 'install.bat')
Assert-RequiredFile (Join-Path $ReleaseRoot 'camera_driver\MVDCP2_Setup.exe')

$profileText = Get-Content -Raw -LiteralPath (Join-Path $AppTarget 'config\camera_profile.yaml')
if ($profileText -notmatch '(?m)^sdk_path:\s*["'']?mindvision["'']?\s*$') {
    throw '发布包中的 camera_profile.yaml SDK 路径校验失败。'
}

$manifest = @()
foreach ($relative in @(
    'SwallowabilityConsole\SwallowabilityConsole.exe',
    'SwallowabilityConsole\config\vlm.yaml',
    'SwallowabilityConsole\config\camera_profile.yaml',
    'SwallowabilityConsole\data\calibration\calibration_new.json',
    'SwallowabilityConsole\runs\measurement\background.jpg',
    'SwallowabilityConsole\mindvision\Demo\Python\Basic\mvsdk.py',
    'SwallowabilityConsole\mindvision\SDK\X64\MVCAMSDK_X64.dll',
    'camera_driver\MVDCP2_Setup.exe'
)) {
    $full = Join-Path $ReleaseRoot $relative
    $hash = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
    $manifest += "$hash  $relative"
}
$manifestPath = Join-Path $ReleaseRoot 'SHA256SUMS.txt'
[IO.File]::WriteAllLines($manifestPath, $manifest, [Text.UTF8Encoding]::new($true))

$zipPath = $null
if (-not $NoZip) {
    Write-Host '[6/6] 生成 ZIP 发布包，文件较大，请稍候...'
    $zipPath = Join-Path $OutputRoot 'SwallowabilityConsole_OneClick.zip'
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $ReleaseRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal
} else {
    Write-Host '[6/6] 已按参数跳过 ZIP 压缩。'
}

$releaseSize = [math]::Round(((Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
Write-Host ''
Write-Host "发布目录：$ReleaseRoot"
Write-Host "目录大小：$releaseSize MB"
if ($zipPath) {
    $zipSize = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
    Write-Host "ZIP 文件：$zipPath"
    Write-Host "ZIP 大小：$zipSize MB"
}
Write-Host '发布包制作完成。'