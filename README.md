# 小物件吞咽识别控制台（本地 YOLO）

固定相机拍摄实验台上的目标物，在本地运行 `best.pt` 识别目标，并使用相机标定把检测框或分割轮廓换算成毫米尺寸，最后输出五级吞咽等级。

当前发布版默认使用本地 YOLO，不依赖网络。界面不展示模型名称或接口信息。

## 使用流程

1. 点击“开始相机”。
2. 把目标物放在实验台上。
3. 点击“手动拍照”。
4. 点击“开始分析”。
5. 结果区显示目标名称、实测尺寸、置信度和吞咽等级。

本地 YOLO 模式不要求先点击“采集空台背景”。背景图只在把 `config/inference.yaml` 的 `backend` 改为 `vlm` 后用于兼容旧流程。

## 本地模型

- 模型：`best.pt`
- 配置：`config/inference.yaml`
- 当前模型任务：分割（segment）
- 类别数：12
- 默认输入尺寸：1280
- 默认置信度阈值：0.25

中文显示名可在 `config/inference.yaml` 的 `class_names` 中修改。

## 尺寸和等级

尺寸来自原标定流程，检测框或分割轮廓通过台面单应矩阵转换为毫米。五级规则：

- 有效尺寸小于 12 mm：极易吞咽
- 小于 20 mm：容易吞咽
- 小于 30 mm：可能吞咽
- 小于 40 mm：不容易吞咽
- 40 mm 及以上：无法吞咽

有效尺寸为 `短边 + 0.35 × (长边 - 短边)`。

## 运行源码

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scriptspp.py
```

## 构建 EXE

```powershell
.uild_exe.ps1
```

生成位置：

```text
dist\SwallowabilityConsole\SwallowabilityConsole.exe
```

构建脚本会把 `best.pt` 和 `config` 复制到 EXE 同级目录。PyInstaller 会打包 Ultralytics 和 PyTorch CPU 运行库，因此发布体积较大。

## 切换到原 API 模式

编辑 `config/inference.yaml`：

```yaml
backend: vlm
```

再按 `config/vlm.yaml` 配置接口。默认发布配置为 `yolo`，不会调用网络接口。
