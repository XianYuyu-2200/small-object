# 小物件吞咽识别控制台（相机 + 多模态 API）

比赛演示版桌面程序：迈德威视相机实时预览，手动拍照后调用兼容 OpenAI Chat Completions 协议的多模态 API，并将分析结果以 JSON 展示。

当前 EXE 主流程不加载 YOLO，也不需要识别目标类别。

## 操作流程

1. 启动程序，点击“开始相机”。
2. 把目标物放在实验台上，确认实时画面清晰。
3. 点击“手动拍照”，原图保存到 `runs/captures/`。
4. 点击“开始分析”，后台线程调用多模态 API。
5. 界面显示“容易吞咽”“不容易吞咽”或“无法判断”，下方展示完整 JSON。
6. JSON 保存到 `runs/analysis/latest.json`。

分析期间不会阻塞相机和界面；同一张照片可以重复分析。

## 配置 API

编辑 `config/vlm.yaml`：

```yaml
endpoint: "https://your-provider.example/v1/chat/completions"
model: "your-vision-model"
api_key_env: "VLM_API_KEY"
api_key: ""
timeout_seconds: 90
max_image_edge: 1280
jpeg_quality: 88
```

要求：

- `endpoint` 必须是可直接 POST 的完整 Chat Completions 地址。
- `model` 必须是支持图片输入的多模态模型。
- API Key 优先读取 `api_key_env` 指定的环境变量；没有环境变量时使用 `api_key`。
- 不要把包含真实 API Key 的 `config/vlm.yaml` 提交或发给他人。
- 如果服务需要额外请求头，可填写 `extra_headers`。

设置环境变量的示例：

```powershell
$env:VLM_API_KEY = "your-api-key"
```

## 运行源码

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts\app.py
```

API 客户端使用 Python 标准库 `urllib`，不依赖特定厂商 SDK。

## 构建 EXE

```powershell
.\build_exe.ps1
```

生成位置：

```text
dist\SwallowabilityConsole\SwallowabilityConsole.exe
```

打包脚本会把 `config` 复制到 EXE 同级目录。之后可直接编辑：

- `config/camera_profile.yaml`：相机、曝光、分辨率和预览宽度。
- `config/vlm.yaml`：API 地址、模型和密钥。

## API 返回格式

程序提示模型只返回以下 JSON：

```json
{
  "object_name": "红色小方块",
  "decision": "可能吞咽",
  "confidence": 0.82,
  "reasons": [
    "目标整体尺寸较小",
    "外形接近规则块体"
  ]
}
```

界面只显示简短物件名称、尺寸、置信度和五级吞咽等级，不展示 JSON 内容。`decision` 只接受：

- `无法吞咽`
- `不容易吞咽`
- `可能吞咽`
- `容易吞咽`
- `极易吞咽`

如果 API 返回 Markdown 代码块、百分比置信度或常见同义措辞，程序也会尽量解析；格式无法解析时会在界面和 `runs/analysis_error.log` 中报告错误。

## 注意事项

- 固定相机高度、视角和拍照位置可提高同一实验台内结果的一致性。
- 单张俯视图缺少尺度参照，API 无法知道目标真实毫米尺寸；多个相近大小目标的判断可能不稳定。
- 当前程序用于比赛演示，不适合作为医疗或产品安全结论。
- 历史 YOLO 训练和命令行推理代码仍保留在项目中，但新的桌面 EXE 不再使用它们。
