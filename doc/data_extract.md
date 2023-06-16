# 从原始数据提取训练数据集

当您下载我们提供的原始数据集并解压后，您将看到下面的内容：

```
.
├── lumbar_train150
├── lumbar_train150_annotation.json
├── lumbar_train51_annotation.json
└── train
```

将它们全部移动到 `Spine_diagnose/tools` 目录下，并运行 `gen_data.py`

```
python gen_data.py
```

等待一段时间后，您将在 `Spine_diagnose/tools/datasets` 目录下获取到与我们提供的预处理后的数据相同的内容。