# YOLO 微調訓練指南

## 資料集準備完成

已成功合併三個資料集:
- **訓練集**: 12,013 張圖片
- **驗證集**: 1,844 張圖片
- **類別**:
  - 0: tennisball (網球)
  - 1: plastic-bottle (寶特瓶)
  - 2: person (人)

資料集位置: `merged_dataset/`

## 訓練模型選項

### 選項 1: 輕量快速訓練 (推薦入門)
適合快速測試,GPU 記憶體要求低,訓練速度快

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # Nano model - 最小最快
model.train(
    data="merged_dataset/data.yaml",
    epochs=30,              # 較少的訓練輪數
    imgsz=640,
    batch=16,
    lr0=0.001,              # 低學習率用於微調
    device=0,               # GPU
    name="quick_finetune",
    patience=10,
    pretrained=True
)
```

**特點**:
- 訓練時間: ~1-2 小時 (取決於 GPU)
- GPU 記憶體: ~4GB
- 適合: 快速驗證效果

### 選項 2: 平衡訓練 (推薦)
平衡了精度和速度,適合大多數情況

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="merged_dataset/data.yaml",
    epochs=50,              # 中等訓練輪數
    imgsz=640,
    batch=16,
    lr0=0.001,
    lrf=0.01,
    device=0,
    name="balanced_finetune",
    patience=15,
    save=True,
    plots=True,
    pretrained=True,
    optimizer='AdamW'
)
```

**特點**:
- 訓練時間: ~3-4 小時
- GPU 記憶體: ~4GB
- 適合: 一般使用,這是 `train.py` 的預設配置

### 選項 3: 高精度訓練
使用更大模型,追求更高精度

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")  # Small model - 比 nano 大
model.train(
    data="merged_dataset/data.yaml",
    epochs=100,             # 更多訓練輪數
    imgsz=640,
    batch=8,                # 減小 batch size (更大模型需要更多記憶體)
    lr0=0.0005,             # 更低的初始學習率
    lrf=0.01,
    device=0,
    name="high_accuracy_finetune",
    patience=20,
    save=True,
    plots=True,
    pretrained=True,
    optimizer='AdamW'
)
```

**特點**:
- 訓練時間: ~6-10 小時
- GPU 記憶體: ~6-8GB
- 適合: 追求最佳精度

### 選項 4: 超高精度訓練
使用中等大小模型,獲得最好效果

```python
from ultralytics import YOLO

model = YOLO("yolo11m.pt")  # Medium model - 更大
model.train(
    data="merged_dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4,                # 小 batch size
    lr0=0.0003,
    lrf=0.01,
    device=0,
    name="ultra_finetune",
    patience=25,
    save=True,
    plots=True,
    pretrained=True,
    optimizer='AdamW',
    amp=True                # 混合精度訓練,節省記憶體
)
```

**特點**:
- 訓練時間: ~10-15 小時
- GPU 記憶體: ~10-12GB
- 適合: 追求極致精度,不在乎訓練時間

### 選項 5: CPU 訓練 (無 GPU 時)
使用 CPU 訓練,速度較慢

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="merged_dataset/data.yaml",
    epochs=20,              # 減少輪數
    imgsz=416,              # 減小圖片尺寸
    batch=4,                # 小 batch
    device='cpu',           # 使用 CPU
    name="cpu_finetune",
    patience=10,
    pretrained=True
)
```

**特點**:
- 訓練時間: 可能需要數天
- 適合: 沒有 GPU 的情況

## 如何訓練

### 方法 1: 使用預設的 train.py (推薦)
直接執行已經配置好的訓練腳本:

```bash
python train.py
```

這會使用**選項 2: 平衡訓練**的配置。

### 方法 2: 自訂訓練
建立你自己的訓練腳本,從上面選擇一個配置複製下來:

```python
# my_custom_train.py
from ultralytics import YOLO

# 選擇一個上面的配置
model = YOLO("yolo11n.pt")
model.train(
    # ... 你選擇的參數
)
```

然後執行:
```bash
python my_custom_train.py
```

### 方法 3: 命令列訓練
直接使用 YOLO CLI:

```bash
# 快速訓練
yolo detect train data=merged_dataset/data.yaml model=yolo11n.pt epochs=30 imgsz=640

# 平衡訓練
yolo detect train data=merged_dataset/data.yaml model=yolo11n.pt epochs=50 imgsz=640 lr0=0.001

# 高精度訓練
yolo detect train data=merged_dataset/data.yaml model=yolo11s.pt epochs=100 imgsz=640 batch=8
```

## 模型大小對比

| 模型 | 參數量 | 模型大小 | 推理速度 | 精度 | 推薦用途 |
|------|--------|----------|----------|------|----------|
| yolo11n.pt | 2.6M | 5.5 MB | 最快 | 中等 | 即時偵測,資源受限 |
| yolo11s.pt | 9.4M | 19 MB | 快 | 較高 | 平衡性能和精度 |
| yolo11m.pt | 20.1M | 40 MB | 中等 | 高 | 高精度需求 |
| yolo11l.pt | 25.3M | 50 MB | 較慢 | 很高 | 最高精度 |
| yolo11x.pt | 56.9M | 113 MB | 慢 | 最高 | 極致精度 |

## 訓練參數說明

- **epochs**: 訓練輪數,越多越好但容易過擬合
- **batch**: 批次大小,越大訓練越穩定但需要更多記憶體
- **imgsz**: 圖片大小,640 是標準值
- **lr0**: 初始學習率,微調時應該用較小的值 (0.0003-0.001)
- **lrf**: 最終學習率倍數
- **patience**: 早停耐心值,多少輪沒改進就停止
- **device**: 'cpu' 或 0,1,2... (GPU 編號)
- **pretrained**: True 保留預訓練權重(重要!)

## 訓練結果

訓練完成後,結果會儲存在:
```
runs/detect/<name>/
├── weights/
│   ├── best.pt      # 最佳模型
│   └── last.pt      # 最後一輪模型
├── results.png      # 訓練曲線
├── confusion_matrix.png  # 混淆矩陣
└── ...
```

## 使用訓練好的模型

```python
from ultralytics import YOLO

# 載入訓練好的模型
model = YOLO('runs/detect/balanced_finetune/weights/best.pt')

# 推理
results = model('path/to/image.jpg')

# 顯示結果
results[0].show()

# 取得偵測框
boxes = results[0].boxes
for box in boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    print(f"Class: {cls}, Confidence: {conf:.2f}")
```

## 建議

1. **首次訓練**: 使用**選項 1 (輕量快速)**快速驗證流程
2. **正式訓練**: 使用**選項 2 (平衡)**獲得不錯的結果
3. **追求精度**: 嘗試**選項 3 或 4**
4. **GPU 記憶體不足**: 減小 batch size 或使用 `amp=True`
5. **過擬合**: 減少 epochs 或使用資料增強

## 監控訓練

訓練過程中會顯示:
- Loss (損失): 越低越好
- mAP (平均精度): 越高越好
- Precision/Recall: 精確率和召回率

可以使用 TensorBoard 查看訓練曲線:
```bash
tensorboard --logdir runs/detect
```

## 重要提醒

**關於微調 (Finetune)**:
- YOLO 預訓練模型已經很強大,可以檢測 person 和許多物體
- 我們的目標是**微調**網球偵測,同時**保留**原本的能力
- 使用較低的學習率 (lr0=0.001) 和 `pretrained=True` 是關鍵
- 不需要太多 epochs,30-50 輪通常就足夠

## 下一步

訓練完成後,你可以:
1. 在 `evalutation.py` 中測試模型性能
2. 在 `main.py` 中整合新模型進行即時偵測
3. 比較不同模型的效果,選擇最適合的
