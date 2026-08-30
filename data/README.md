# 数据目录

目录结构：

```text
data/dataset/
  images/{train,val,test}/
  labels/{train,val,test}/
```

图片与同名 `.txt` 标签配对。标签格式为 YOLO 检测格式：
`class_id x_center y_center width height`，坐标均归一化到 0--1。

建议按“物件实例”切分，而不是把连续视频帧随机分散到不同集合。正式标注前先维护 `config/classes.yaml`，类别编号一旦开始标注不要重排。
