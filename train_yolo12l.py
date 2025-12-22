#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 YOLO v12 Large 進行微調訓練
針對 RTX 4080 (16GB VRAM) 優化
"""
import os
from ultralytics import YOLO

def train_yolo12l():
    """
    使用 YOLO12L 訓練多類別偵測模型

    硬體: RTX 4080 (16GB VRAM)
    模型: YOLO v12 Large
    類別: tennisball, plastic-bottle, person
    """

    # 檢查資料集
    dataset_yaml = "merged_dataset/data.yaml"
    if not os.path.exists(dataset_yaml):
        print(f"錯誤: {dataset_yaml} 不存在")
        print("請先執行: python merge_datasets.py")
        return

    print("="*70)
    print("YOLO v12 Large 微調訓練")
    print("="*70)
    print(f"\n硬體: RTX 4080")
    print(f"模型: yolo12l.pt (Large model)")
    print(f"資料集: {dataset_yaml}")
    print("類別: 0=tennisball, 1=plastic-bottle, 2=person")
    print(f"訓練集: 12,013 張圖片")
    print(f"驗證集: 1,844 張圖片")

    # 載入 YOLO12L 預訓練模型
    print("\n載入 YOLO12 Large 預訓練模型...")
    model = YOLO("yolo12l.pt")

    # 訓練配置 - 針對 4080 優化
    print("\n開始訓練...")
    print("預估訓練時間: 6-10 小時 (取決於資料增強設定)")

    model.train(
        # 資料設定
        data=dataset_yaml,

        # 訓練參數
        epochs=100,             # 訓練輪數
        imgsz=640,              # 圖片大小 (640 是標準值)
        batch=16,               # 批次大小 (4080 可以處理更大,但 16 較穩定)

        # 學習率設定 (微調用較低的學習率)
        lr0=0.0005,             # 初始學習率 (比從頭訓練低)
        lrf=0.01,               # 最終學習率倍數
        momentum=0.937,         # SGD 動量
        weight_decay=0.0005,    # 權重衰減

        # 優化器
        optimizer='AdamW',      # AdamW 對微調效果好

        # 裝置設定
        device=0,               # GPU 0 (你的 4080)
        workers=8,              # 資料載入工作執行緒

        # 早停與儲存
        patience=20,            # 20 輪沒改進就停止
        save=True,              # 儲存檢查點
        save_period=10,         # 每 10 輪儲存一次

        # 輸出設定
        name="yolo12l_finetune_100ep",
        exist_ok=True,
        plots=True,             # 生成訓練圖表
        verbose=True,           # 顯示詳細輸出

        # 預訓練與混合精度
        pretrained=True,        # 保留預訓練權重 (重要!)
        amp=True,               # 混合精度訓練 (加速且節省記憶體)

        # 驗證設定
        val=True,               # 每輪驗證

        # 資料增強 (可選,增強泛化能力)
        hsv_h=0.015,            # 色相增強
        hsv_s=0.7,              # 飽和度增強
        hsv_v=0.4,              # 明度增強
        degrees=10.0,           # 旋轉角度
        translate=0.1,          # 平移
        scale=0.5,              # 縮放
        shear=0.0,              # 剪切
        perspective=0.0,        # 透視
        flipud=0.0,             # 上下翻轉機率
        fliplr=0.5,             # 左右翻轉機率
        mosaic=1.0,             # Mosaic 增強
        mixup=0.0,              # Mixup 增強
        copy_paste=0.0          # Copy-paste 增強
    )

    print("\n" + "="*70)
    print("訓練完成!")
    print("="*70)
    print("\n模型位置:")
    print("  最佳模型: runs/detect/yolo12l_finetune_100ep/weights/best.pt")
    print("  最後模型: runs/detect/yolo12l_finetune_100ep/weights/last.pt")

    print("\n訓練結果:")
    print("  訓練曲線: runs/detect/yolo12l_finetune_100ep/results.png")
    print("  混淆矩陣: runs/detect/yolo12l_finetune_100ep/confusion_matrix.png")

    print("\n使用訓練好的模型:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('runs/detect/yolo12l_finetune_100ep/weights/best.pt')")
    print("  results = model('test_image.jpg')")

    print("\n下一步:")
    print("  1. 查看訓練曲線評估效果")
    print("  2. 使用 evalutation.py 測試模型")
    print("  3. 在 main.py 中整合模型")

if __name__ == "__main__":
    train_yolo12l()
