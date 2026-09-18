小物件吞咽识别控制台 - 一键部署说明
========================================

一、安装前准备
1. 使用 64 位 Windows 10 或 Windows 11。
2. 连接迈德威视相机。
3. 请先完整解压整个压缩包，不要直接在压缩包中运行，也不要只复制其中的 EXE。

二、一键安装
1. 打开解压后的文件夹。
2. 双击 install.bat。
3. 弹出管理员授权时选择“是”。
4. 安装脚本会自动检查并安装相机驱动组件，然后创建桌面快捷方式并启动程序。

三、程序使用
1. 点击“开始相机”确认画面正常。
2. 将目标物放在实验台中央，保持相机位置和光照不变。
3. 点击“手动拍照”。
4. 点击“开始分析”，等待结果显示。
5. 可同时放置多个目标，程序会分别给出名称、实测尺寸和吞咽等级。

当前发布版本使用本地识别，不需要联网。界面中的“采集空台背景”仅为兼容旧流程保留，
本地识别模式不要求先采集背景，也不影响分析。

四、可以直接运行的软件
如果目标电脑已经安装好迈德威视驱动，也可以直接运行：
SwallowabilityConsole\SwallowabilityConsole.exe

五、目录说明
SwallowabilityConsole\SwallowabilityConsole.exe
    主程序。

SwallowabilityConsole\best.pt
    本地识别模型，禁止删除或替换为其他模型。

SwallowabilityConsole\_internal\
    程序运行库，禁止删除或移动。

SwallowabilityConsole\config\
    相机、识别和尺寸参数，发布包内已经预置。
    config\inference.yaml 当前 backend 为 yolo。

SwallowabilityConsole\data\calibration\
    台面尺寸标定数据，禁止删除。

SwallowabilityConsole\mindvision\
    程序使用的迈德威视 64 位运行库。

camera_driver\
    相机驱动安装组件。

六、常见问题
1. 提示未发现相机：
   关闭可能占用相机的官方相机程序，拔插相机 USB 线，再重新打开软件。

2. 分析失败：
   检查 best.pt、标定文件是否完整，并确认目标在画面中清晰可见。

3. 检测不到目标或漏检：
   可降低 config\inference.yaml 中的 confidence_threshold，例如从 0.25 调到 0.15。

4. 误检较多：
   可提高 confidence_threshold，例如从 0.25 调到 0.35 或 0.45。

5. 换相机、移动相机位置或改变相机高度后：
   原有尺寸标定可能失效，需要重新标定。

七、重要提醒
1. 必须保留完整目录结构，尤其是 _internal、best.pt、config、data 和 mindvision。
2. 不要删除或改名 SwallowabilityConsole\best.pt。
3. 新电脑首次安装相机驱动需要管理员权限。
