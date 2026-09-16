#Requires -Version 5.1
[CmdletBinding()]
param(
    [switch]$NoLaunch,
    [switch]$SkipDriverInstall,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
$script:LogPath = Join-Path $env:TEMP 'SwallowabilityConsole_install.log'

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = 'INFO'
    )
    $line = '[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    try {
        Add-Content -LiteralPath $script:LogPath -Value $line -Encoding UTF8
    } catch {
    }
    Write-Host $line
}

function Test-Administrator {
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Host '相机驱动安装需要管理员权限，正在请求授权...'
    $argumentList = '-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $PSCommandPath
    if ($NoLaunch) { $argumentList += ' -NoLaunch' }
    if ($SkipDriverInstall) { $argumentList += ' -SkipDriverInstall' }
    if ($NoPause) { $argumentList += ' -NoPause' }
    try {
        $elevated = Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentList -Verb RunAs -Wait -PassThru
        exit $elevated.ExitCode
    } catch {
        Write-Host "无法获取管理员权限：$($_.Exception.Message)" -ForegroundColor Red
        if (-not $NoPause) { Read-Host '按 Enter 退出' | Out-Null }
        exit 1
    }
}

function Get-UninstallEntries {
    $keys = @(
        'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
        'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*'
    )
    foreach ($key in $keys) {
        Get-ItemProperty -Path $key -ErrorAction SilentlyContinue |
            Where-Object { $_.DisplayName -match 'MindVision|MVDCP' }
    }
}

function Test-DriverInstalled {
    param([string]$DriverFileName)
    $pnputil = Join-Path $env:SystemRoot 'System32\pnputil.exe'
    $output = (& $pnputil /enum-drivers 2>&1 | Out-String)
    return $output -match [regex]::Escape($DriverFileName)
}

function Install-DriverInf {
    param([string]$InfPath)
    if (-not (Test-Path -LiteralPath $InfPath -PathType Leaf)) {
        Write-Log "未找到驱动文件，跳过：$InfPath" 'WARN'
        return
    }
    $pnputil = Join-Path $env:SystemRoot 'System32\pnputil.exe'
    Write-Log "安装相机驱动：$InfPath"
    $output = (& $pnputil /add-driver $InfPath /install 2>&1)
    $exitCode = $LASTEXITCODE
    foreach ($line in @($output)) {
        if (-not [string]::IsNullOrWhiteSpace([string]$line)) {
            Write-Log ([string]$line) 'DRIVER'
        }
    }
    if ($exitCode -notin @(0, 1641, 3010)) {
        Write-Log "驱动安装返回代码 $exitCode；如果相机已被系统正确识别，可继续使用。" 'WARN'
    }
}

function Invoke-CameraSetup {
    param([string]$InstallerPath)
    if (-not (Test-Path -LiteralPath $InstallerPath -PathType Leaf)) {
        throw "缺少相机驱动安装器：$InstallerPath"
    }

    Write-Log '开始安装迈德威视相机驱动组件...'
    $silent = Start-Process -FilePath $InstallerPath -ArgumentList '/s' -Wait -PassThru
    if ($silent.ExitCode -in @(0, 1641, 3010)) {
        Write-Log "驱动组件静默安装已返回代码 $($silent.ExitCode)。"
        return
    }

    Write-Log "静默安装返回代码 $($silent.ExitCode)，改用可视安装程序。" 'WARN'
    $visible = Start-Process -FilePath $InstallerPath -Wait -PassThru
    if ($visible.ExitCode -notin @(0, 1641, 3010)) {
        throw "相机驱动组件安装失败，退出代码：$($visible.ExitCode)"
    }
    Write-Log "驱动组件安装已返回代码 $($visible.ExitCode)。"
}

function New-Shortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$WorkingDirectory
    )
    $parent = Split-Path -Parent $ShortcutPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = $WorkingDirectory
    $shortcut.IconLocation = "$TargetPath,0"
    $shortcut.Description = '小物件吞咽识别控制台'
    $shortcut.Save()
}

function Main {
    Write-Log '开始安装小物件吞咽识别控制台。'
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw '本程序仅支持 64 位 Windows。'
    }

    $packageRoot = $PSScriptRoot
    $appDir = Join-Path $packageRoot 'SwallowabilityConsole'
    $exePath = Join-Path $appDir 'SwallowabilityConsole.exe'
    if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) {
        if (Test-Path -LiteralPath (Join-Path $packageRoot 'SwallowabilityConsole.exe') -PathType Leaf) {
            $appDir = $packageRoot
            $exePath = Join-Path $appDir 'SwallowabilityConsole.exe'
        } else {
            throw "未找到 SwallowabilityConsole.exe。请保持发布包目录完整，不要只复制 EXE。"
        }
    }

    foreach ($required in @(
        (Join-Path $appDir '_internal\base_library.zip'),
        (Join-Path $appDir 'config\vlm.yaml'),
        (Join-Path $appDir 'config\camera_profile.yaml'),
        (Join-Path $appDir 'data\calibration\calibration_new.json'),
        (Join-Path $appDir 'runs\measurement\background.jpg'),
        (Join-Path $appDir 'mindvision\Demo\Python\Basic\mvsdk.py'),
        (Join-Path $appDir 'mindvision\SDK\X64\MVCAMSDK_X64.dll')
    )) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "发布包文件不完整，缺少：$required"
        }
    }

    if (-not $SkipDriverInstall) {
        $installed = @(Get-UninstallEntries)
        $hasMindVision = [bool]($installed | Where-Object { $_.DisplayName -match '^MindVision' } | Select-Object -First 1)
        $hasMvdcp = [bool]($installed | Where-Object { $_.DisplayName -match '^MVDCP' } | Select-Object -First 1)
        if (-not $hasMindVision -or -not $hasMvdcp) {
            Invoke-CameraSetup (Join-Path $packageRoot 'camera_driver\MVDCP2_Setup.exe')
        } else {
            Write-Log '检测到迈德威视驱动组件已经安装。'
        }

        if (-not (Test-DriverInstalled -DriverFileName 'mvu2camera.inf')) {
            Install-DriverInf (Join-Path $packageRoot 'camera_driver\Drivers\USB\WIN10\MvU2Camera.inf')
        } else {
            Write-Log 'USB 相机驱动已经安装。'
        }
        if (-not (Test-DriverInstalled -DriverFileName 'mvgigedriverlwf.inf')) {
            Install-DriverInf (Join-Path $packageRoot 'camera_driver\Drivers\GIGE\WIN10\60\MvGigeDriverLwf.inf')
        } else {
            Write-Log 'GigE 相机驱动已经安装。'
        }
    } else {
        Write-Log '已按参数跳过相机驱动安装。' 'WARN'
    }

    $desktop = [Environment]::GetFolderPath('Desktop')
    $programs = [Environment]::GetFolderPath('Programs')
    $shortcutName = '小物件吞咽识别控制台.lnk'
    New-Shortcut -ShortcutPath (Join-Path $desktop $shortcutName) -TargetPath $exePath -WorkingDirectory $appDir
    New-Shortcut -ShortcutPath (Join-Path $programs "吞咽识别\$shortcutName") -TargetPath $exePath -WorkingDirectory $appDir
    Write-Log '已创建桌面和开始菜单快捷方式。'

    if (-not $NoLaunch) {
        Start-Process -FilePath $exePath -WorkingDirectory $appDir
        Write-Log '程序已启动。'
    }
    Write-Log "安装完成。日志文件：$script:LogPath"
}

try {
    Main
} catch {
    Write-Log "安装失败：$($_.Exception.Message)" 'ERROR'
    if (-not $NoPause) {
        Read-Host '安装失败，按 Enter 退出' | Out-Null
    }
    exit 1
}

if (-not $NoPause) {
    Read-Host '安装完成，按 Enter 退出' | Out-Null
}
exit 0