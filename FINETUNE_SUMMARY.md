# YOLO 微調專案總結

## 已完成的工作

### 1. 資料集準備 ✅
已下載並合併三個資料集:
- Tennis Ball Detection (80 train, 27 valid)
- Plastic Bottles (1,000 train, 247 valid)
- People (10,933 train, 1,570 valid)

**合併後資料集**: `merged_dataset/`
- 訓練集: 12,013 張圖片
- 驗證集: 1,844 張圖片
- 類別: tennisball (0), plastic-bottle (1), person (2)

### 2. 訓練腳本 ✅
建立了多個訓練選項:

#### 主要訓練腳本
- **`train_yolo12l.py`** - 專為 RTX 4080 優化的 YOLO12L 訓練腳本 (推薦)
- **`train.py`** - 通用的 YOLO11n 訓練腳本 (輕量級)

#### 訓練指南
- **`TRAINING_GUIDE.md`** - 詳細的訓練指南,包含 5 種不同配置

## 推薦配置 (RTX 4080 + YOLO12L)

執行以下指令開始訓練:
```bash
python train_yolo12l.py
```

### 訓練參數
- **模型**: YOLO v12 Large
- **Epochs**: 100
- **Batch size**: 16
- **Image size**: 640
- **Learning rate**: 0.0005 (初始)
- **Optimizer**: AdamW
- **Mixed precision**: 啟用 (amp=True)

### 預期結果
- **訓練時間**: 6-10 小時
- **GPU 記憶體使用**: ~10-12GB / 16GB
- **輸出位置**: `runs/detect/yolo12l_finetune_100ep/weights/best.pt`

## 訓練策略說明

### 為什麼是微調 (Finetune)?
1. **保留預訓練能力**: YOLO 預訓練模型已經很強,能偵測 person 和許多物體
2. **重點改進網球偵測**: 你的主要需求是改進網球偵測 (避免誤認為橘子)
3. **較低學習率**: 使用 lr0=0.0005 而非 0.01,避免破壞預訓練權重
4. **較少 epochs**: 100 輪對微調來說已足夠 (從頭訓練需要 300+)

### 關鍵參數解釋
- **`pretrained=True`**: 保留 YOLO 的預訓練權重
- **`lr0=0.0005`**: 低學習率,溫和調整權重
- **`patience=20`**: 20 輪沒改進就停止,避免過擬合
- **`amp=True`**: 混合精度訓練,加速 30-50%

## 如何開始訓練

### 步驟 1: 確認資料集
```bash
ls merged_dataset/
# 應該看到: train/, valid/, data.yaml
```

### 步驟 2: 確認 GPU
```bash
python verify_cuda.py
# 確認 CUDA 可用
```

### 步驟 3: 開始訓練
```bash
python train_yolo12l.py
```

### 步驟 4: 監控訓練
訓練過程中會顯示:
- Epoch 進度
- Loss (box_loss, cls_loss, dfl_loss)
- mAP50, mAP50-95
- Precision, Recall

可以用 TensorBoard 查看:
```bash
tensorboard --logdir runs/detect
```

## 訓練完成後

### 1. 查看結果
```bash
# 訓練曲線
runs/detect/yolo12l_finetune_100ep/results.png

# 混淆矩陣
runs/detect/yolo12l_finetune_100ep/confusion_matrix.png

# 驗證範例
runs/detect/yolo12l_finetune_100ep/val_batch*.jpg
```

### 2. 測試模型
```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo12l_finetune_100ep/weights/best.pt')
results = model('test_image.jpg')
results[0].show()
```

### 3. 評估性能
```bash
python evalutation.py
# 可能需要修改腳本中的模型路徑
```

### 4. 整合到專案
在 `main.py` 或其他偵測腳本中:
```python
model = YOLO('runs/detect/yolo12l_finetune_100ep/weights/best.pt')
```

## 其他訓練選項

如果 YOLO12L 訓練時間太長或記憶體不足:

### 選項 A: 使用 YOLO11 Nano (更快)
```bash
python train.py
```
- 訓練時間: 3-4 小時
- 記憶體: ~4GB

### 選項 B: 減少 batch size
修改 `train_yolo12l.py` 中的 `batch=16` 改為 `batch=8`

### 選項 C: 減少 epochs
修改 `epochs=100` 改為 `epochs=50`

## 常見問題

### Q: 訓練時 GPU 記憶體不足?
A: 降低 batch size: `batch=8` 或 `batch=4`

### Q: 訓練太慢?
A:
1. 確認 `amp=True` 啟用混合精度
2. 減少 epochs: `epochs=50`
3. 考慮使用較小模型 (yolo11m 或 yolo11s)

### Q: 過擬合 (training loss 低但 val loss 高)?
A:
1. 增加資料增強強度
2. 減少 epochs
3. 使用早停 (patience)

### Q: 如何判斷訓練效果好?
A: 查看 mAP50-95:
- > 0.7: 很好
- 0.5-0.7: 良好
- < 0.5: 需要改進

### Q: 網球偵測還是不準?
A: 可以考慮:
1. 蒐集更多你實際環境的網球圖片
2. 增加網球類別的樣本權重
3. 調整信心閾值 (confidence threshold)

## 檔案清單

### 資料集
- `merged_dataset/` - 合併後的資料集
- `tennis_dataset/` - 網球資料集
- `plastic_bottles/` - 寶特瓶資料集
- `people/` - 人物資料集

### 訓練腳本
- `train_yolo12l.py` - YOLO12L 訓練 (推薦用於 4080)
- `train.py` - YOLO11n 訓練 (較快)
- `merge_datasets.py` - 資料集合併腳本

### 文件
- `TRAINING_GUIDE.md` - 詳細訓練指南
- `FINETUNE_SUMMARY.md` - 本文件
- `DOWNLOAD_DATASETS_GUIDE.md` - 資料集下載指南

### 工具
- `evalutation.py` - 模型評估
- `verify_cuda.py` - 驗證 CUDA

## 建議流程

1. ✅ **資料準備** (已完成)
2. ⏳ **開始訓練**: `python train_yolo12l.py`
3. ⏳ **監控訓練**: 觀察 loss 和 mAP
4. ⏳ **評估結果**: 查看訓練曲線和混淆矩陣
5. ⏳ **測試模型**: 用實際圖片測試
6. ⏳ **整合應用**: 更新 main.py

## 預期改進

微調後,你的模型應該能夠:
- ✅ **準確辨識網球** (不會誤認為橘子)
- ✅ **保持人物偵測能力** (YOLO 原本就很強)
- ✅ **保持寶特瓶偵測能力**
- ✅ **同時處理三種物體**

祝訓練順利! 🚀
