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
$ReleaseName = 'DatasetCapture_Portable'
$ReleaseRoot = [IO.Path]::GetFullPath((Join-Path $OutputRoot $ReleaseName))
if ([string]::IsNullOrWhiteSpace($DistRootPath)) {
    $DistRoot = Join-Path $ProjectRoot 'dist_capture\DatasetCapture'
} elseif ([IO.Path]::IsPathRooted($DistRootPath)) {
    $DistRoot = [IO.Path]::GetFullPath($DistRootPath)
} else {
    $DistRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $DistRootPath))
}

function Assert-SafeChildPath {
    param([string]$Parent, [string]$Child)
    $parentFull = [IO.Path]::GetFullPath($Parent).TrimEnd('\', '/')
    $childFull = [IO.Path]::GetFullPath($Child).TrimEnd('\', '/')
    $prefix = $parentFull + [IO.Path]::DirectorySeparatorChar
    if (-not $childFull.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝操作发布目录之外的路径：$childFull"
    }
}

function Assert-RequiredFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "发布文件缺失：$Path"
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

if (-not $SkipBuild) {
    Write-Host '[1/6] 构建采集工具 EXE...'
    & python -m PyInstaller --noconfirm --clean --distpath (Join-Path $ProjectRoot 'dist_capture') --workpath (Join-Path $ProjectRoot 'build_capture') (Join-Path $ProjectRoot 'DatasetCapture.spec')
    if ($LASTEXITCODE -ne 0) {
        throw "EXE 构建失败，退出码：$LASTEXITCODE"
    }
} else {
    Write-Host '[1/6] 跳过构建，使用现有 dist 产物。'
}

Assert-RequiredFile (Join-Path $DistRoot 'DatasetCapture.exe')
Assert-RequiredFile (Join-Path $DistRoot '_internal\base_library.zip')
Assert-RequiredFile (Join-Path $DistRoot 'config\camera_profile.yaml')
Assert-RequiredFile (Join-Path $DistRoot 'config\capture_classes.txt')

# 安全检查：采集工具包里绝不能出现分析服务的密钥配置
$leaks = @(Get-ChildItem -LiteralPath $DistRoot -Recurse -File -Filter 'vlm.yaml' -ErrorAction SilentlyContinue)
if ($leaks.Count -gt 0) {
    throw "安全检查失败：采集工具包中出现了 vlm.yaml，可能泄露密钥。"
}
$keyHit = $false
foreach ($file in (Get-ChildItem -LiteralPath $DistRoot -Recurse -File -Include '*.yaml','*.txt','*.json','*.py' -ErrorAction SilentlyContinue)) {
    $content = Get-Content -LiteralPath $file.FullName -Raw -ErrorAction SilentlyContinue
    if ($content -match 'ark-[A-Za-z0-9]') { $keyHit = $true; break }
}
if ($keyHit) {
    throw '安全检查失败：采集工具包中检测到疑似 API 密钥。'
}

Assert-RequiredFile (Join-Path $ProjectRoot 'install_dataset.ps1')
Assert-RequiredFile (Join-Path $ProjectRoot 'install_dataset.bat')
Assert-RequiredFile (Join-Path $ProjectRoot 'README_采集工具说明.txt')
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
$AppTarget = Join-Path $ReleaseRoot 'DatasetCapture'

Write-Host '[2/6] 复制采集程序、相机参数和类别列表...'
Copy-Tree -Source $DistRoot -Destination $AppTarget
if (-not (Test-Path -LiteralPath (Join-Path $AppTarget 'dataset'))) {
    New-Item -ItemType Directory -Force -Path (Join-Path $AppTarget 'dataset') | Out-Null
}

Write-Host '[3/6] 复制迈德威视最小运行库...'
$mvBasicTarget = Join-Path $AppTarget 'mindvision\Demo\Python\Basic'
$mvSdkTarget = Join-Path $AppTarget 'mindvision\SDK\X64'
if (-not (Test-Path -LiteralPath (Join-Path $mvBasicTarget 'mvsdk.py'))) {
    Copy-Tree -Source (Join-Path $MindVisionRoot 'Demo\Python\Basic') -Destination $mvBasicTarget
}
if (-not (Test-Path -LiteralPath (Join-Path $mvSdkTarget 'MVCAMSDK_X64.dll'))) {
    Copy-Tree -Source (Join-Path $MindVisionRoot 'SDK\X64') -Destination $mvSdkTarget
}

Write-Host '[4/6] 复制相机驱动和说明文档...'
$driverTarget = Join-Path $ReleaseRoot 'camera_driver'
Copy-Tree -Source (Join-Path $MindVisionRoot 'MVDCP2_Setup.exe') -Destination (Join-Path $driverTarget 'MVDCP2_Setup.exe')
Copy-Tree -Source (Join-Path $MindVisionRoot 'Drivers') -Destination (Join-Path $driverTarget 'Drivers')
Copy-Tree -Source (Join-Path $ProjectRoot 'install_dataset.ps1') -Destination (Join-Path $ReleaseRoot 'install_dataset.ps1')
Copy-Tree -Source (Join-Path $ProjectRoot 'install_dataset.bat') -Destination (Join-Path $ReleaseRoot 'install_dataset.bat')
Copy-Tree -Source (Join-Path $ProjectRoot 'README_采集工具说明.txt') -Destination (Join-Path $ReleaseRoot 'README_采集工具说明.txt')

Write-Host '[5/6] 校验发布文件...'
Assert-RequiredFile (Join-Path $AppTarget 'DatasetCapture.exe')
Assert-RequiredFile (Join-Path $AppTarget 'config\camera_profile.yaml')
Assert-RequiredFile (Join-Path $AppTarget 'config\capture_classes.txt')
Assert-RequiredFile (Join-Path $AppTarget 'mindvision\Demo\Python\Basic\mvsdk.py')
Assert-RequiredFile (Join-Path $AppTarget 'mindvision\SDK\X64\MVCAMSDK_X64.dll')
Assert-RequiredFile (Join-Path $ReleaseRoot 'install_dataset.bat')
Assert-RequiredFile (Join-Path $ReleaseRoot 'camera_driver\MVDCP2_Setup.exe')

$postLeaks = @(Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File -Filter 'vlm.yaml' -ErrorAction SilentlyContinue)
if ($postLeaks.Count -gt 0) {
    throw "安全检查失败：发布包中出现 vlm.yaml。"
}

$manifest = @()
foreach ($relative in @(
    'DatasetCapture\DatasetCapture.exe',
    'DatasetCapture\config\camera_profile.yaml',
    'DatasetCapture\config\capture_classes.txt',
    'DatasetCapture\mindvision\Demo\Python\Basic\mvsdk.py',
    'DatasetCapture\mindvision\SDK\X64\MVCAMSDK_X64.dll',
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
    $zipPath = Join-Path $OutputRoot 'DatasetCapture_Portable.zip'
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $ReleaseRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal -Force
} else {
    Write-Host '[6/6] 已跳过 ZIP 打包。'
}

$sizeMb = [math]::Round(((Get-ChildItem -LiteralPath $ReleaseRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 1)
Write-Host ''
Write-Host "发布目录：$ReleaseRoot"
Write-Host "目录大小：$sizeMb MB"
if ($zipPath) {
    $zipMb = [math]::Round((Get-Item -LiteralPath $zipPath).Length / 1MB, 1)
    Write-Host "ZIP 文件：$zipPath"
    Write-Host "ZIP 大小：$zipMb MB"
}
Write-Host '发布包制作完成。'
