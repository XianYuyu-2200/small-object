# 小物件吞咽风险识别演示（YOLO）

面向教学比赛的固定俯视相机演示工程：检测约 31 类小物件，利用赛前离线标定把检测框的平面投影换算为毫米，并以可审计的规则输出 `易误食`、`不易误食` 或 `无法判断`。

> 这不是儿童产品安全认证、医疗建议或合规判定。单目二维图像无法可靠判断厚度、可拆卸性、球体/倾斜摆放等三维风险；这些情形必须人工复核或输出“无法判断”。

## 先决条件

- Python 3.10+。
- 已安装迈德威视相机驱动；该相机能作为 OpenCV 视频设备使用，或按其 SDK 将画面桥接为 OpenCV 帧。
- 赛前固定相机高度（约 32.8 cm）、俯角、焦距、对焦、分辨率与台面位置。比赛期间不显示任何标尺/ArUco。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,yolo]"
```

## 1. 维护类别和风险规则

先把 [config/classes.yaml](G:\codex\codex-yolov26\config\classes.yaml) 的 31 个占位名改为真实物件名称。开始标注后，**不要改变已有编号**。

[config/risk_rules.yaml](G:\codex\codex-yolov26\config\risk_rules.yaml) 的 `ingestible_max_mm` 默认是 `null`；未填入且未记录教学标准来源时，推理将只输出“无法判断”。不要用未经核实的网络阈值替代课程要求。

## 2. 赛前离线标定

在最终装配状态下拍摄至少 10 张棋盘格照片（不同位置、角度、清晰），另拍一张棋盘格平放于最终台面的参考图：

```powershell
python scripts/calibrate.py --images data/calibration/chessboard --pattern 9x6 --square-mm 20 --table-reference data/calibration/table_reference.jpg
```

这将生成 `data/calibration/calibration.json`。棋盘格随后撤走；比赛现场无须出现标记。相机、分辨率、焦距、对焦、支架或台面有变化时重新标定。

## 3. 采集与标注

```powershell
python scripts/capture.py --label class_00 --camera 0 --width 1920 --height 1080
```

空格保存、Esc 退出。每类建议先采集 150–300 张有效图，覆盖位置、旋转、光照、反光、遮挡、正反面和多物件。使用 CVAT、LabelImg 或 Roboflow 标注为 YOLO 检测格式，按 [data/README.md](G:\codex\codex-yolov26\data\README.md) 放置。请按“物件实例”而不是连续帧随机切分 train/val/test。

```powershell
python scripts/check_dataset.py --root data/dataset --classes 31
```

## 4. 训练与演示

```powershell
python scripts/train.py --data config/dataset.yaml --classes config/classes.yaml --model yolo11n.pt --epochs 100 --imgsz 640 --device 0
python scripts/infer.py --source 0 --model runs/detect/train/weights/best.pt --calibration data/calibration/calibration.json
```

推理结果写至 `runs/inference/latest.jpg` 与 `runs/inference/latest.json`。Esc 退出实时画面。

## 验证与限制

```powershell
python -m pytest tests -q
```

在独立于训练数据的实物上、台面多个位置测量并记录误差。接近阈值、低置信度、遮挡、叠放、贴边、未平贴台面或三维形状不可靠时，系统应返回“无法判断”，不要在教学比赛中夸大其尺寸精度或安全结论。
