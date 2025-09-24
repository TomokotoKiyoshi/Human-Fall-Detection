#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8n微调脚本 - 摔倒检测

第一阶段：只微调分类头，区分正常/摔倒
第二阶段：冻结前21层，深度微调
"""

from ultralytics import YOLO
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    # 加载预训练模型
    model_name = 'yolov8s.pt'  # 使用yolov8s模型
    model_path = PROJECT_ROOT / 'yolo_models' / model_name

    if not model_path.exists():
        print(f"下载模型到: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 首次运行会自动下载到当前目录
        import os
        original_dir = os.getcwd()
        os.chdir(str(model_path.parent))  # 切换到目标目录

        try:
            model = YOLO(model_name)  # 这会下载到当前目录
            print(f"模型已下载到: {model_path}")
        finally:
            os.chdir(original_dir)  # 恢复原目录
    else:
        print(f"加载现有模型: {model_path}")
        model = YOLO(str(model_path))

    # 训练配置
    config_path = PROJECT_ROOT / 'configs' / 'fall_detection.yaml'

    # 第一阶段：微调模型
    print("开始第一阶段训练...")
    results_1 = model.train(
        data=str(config_path),
        epochs=2,              # 训练轮数
        imgsz=640,             # 图像尺寸
        batch=32,              # 减小批大小避免内存溢出
        device='cuda',         # 使用GPU
        project='results/models', # 保存目录
        name='fall_detection', # 实验名称
        exist_ok=True,
        patience=5,           # 早停耐心值
        save=True,            # 保存模型
        pretrained=True,      # 使用预训练权重
        lr0=1e-3,             # 初始学习率
        workers=4,            # 减少数据加载线程
        cache=False,          # 不缓存图像到内存
        amp=True,             # 使用混合精度训练节省内存
        val=True,             # 每轮验证
        plots=True,          # 关闭绘图节省内存
        optimizer='AdamW',     # 使用AdamW优化器
        cos_lr=True,          # 余弦退火学习率
        translate=0.1,      # 平移增强
        scale=0.1,          # 缩放增强
        fliplr=0.5,          # 水平翻转概率
        perspective=0.001,   # 透视变换
        freeze=20,             # 冻结前20层，只训练分类头
        box=7.5,              # 边界框损失权重
        cls=1,              # 分类损失权重
        dfl=1.5,               # 分布式边界框
    )

    # 复制最佳模型到多个位置
    import shutil
    source_path = PROJECT_ROOT / 'results' / 'models' / 'fall_detection' / 'weights' / 'best.pt'

    if source_path.exists():
        # 1. 复制到 models/yolo_fall/
        model_path1 = PROJECT_ROOT / 'models' / 'yolo_fall' / 'best.pt'
        model_path1.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, model_path1)
        print(f"\n✅ 模型已复制到: {model_path1}")

        # 2. 复制到 fall_detection/models/ (用于打包)
        model_path2 = PROJECT_ROOT / 'fall_detection' / 'models' / 'best.pt'
        model_path2.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_path, model_path2)
        print(f"✅ 模型已复制到: {model_path2} (打包用)")

        print(f"\n第一阶段训练完成！模型保存在:")
        print(f"  原始位置: {source_path}")
        print(f"  备份位置: {model_path1}")
        print(f"  打包位置: {model_path2}")
    else:
        print(f"\n注意: 训练已完成，模型保存在: {source_path.parent}")

    # 第二阶段：加载第一阶段模型进行深度微调
    print("\n开始第二阶段训练...")
    stage1_model_path = PROJECT_ROOT / 'models' / 'yolo_fall' / 'best.pt'
    model = YOLO(str(stage1_model_path))
    
    results_2 = model.train(
        data=str(config_path),
        epochs=50,              # 增加训练轮数进行深度微调
        imgsz=640,              # 图像尺寸
        batch=16,               # 进一步减小批大小以适应更复杂的训练
        device='cuda',
        project='results/models',   # 保存目录
        name='fall_detection_stage2',  # 第二阶段实验名称
        exist_ok=True,
        patience=10,            # 增加早停耐心值
        save=True,              # 保存模型
        pretrained=False,       # 不重新加载预训练权重，使用第一阶段的权重
        lr0=5e-4,              # 降低学习率进行精细调优
        momentum=0.9,           # 动量
        weight_decay=5e-4,      # 权重衰减
        warmup_epochs=3,        # 预热轮数
        warmup_momentum=0.8,    # 预热动量
        workers=4,              # 数据加载线程
        cache=False,            # 不缓存图像到内存
        amp=True,               # 使用混合精度训练
        val=True,               # 每轮验证
        plots=True,             # 开启绘图查看训练过程
        optimizer='AdamW',      # 使用AdamW优化器
        cos_lr=True,           # 余弦退火学习率
        
        # 数据增强参数 - 第二阶段可以使用更强的增强
        translate=0.15,         # 平移增强
        scale=0.2,             # 缩放增强
        fliplr=0.5,            # 水平翻转概率
        perspective=0.002,      # 透视变换
        
        # 损失函数权重
        freeze=21,             # 🔥 关键参数：冻结前21层
        box=7.5,               # 边界框损失权重
        cls=1,               # 分类损失权重
        dfl=1.5,               # 分布式边界框损失权重
        
        # 其他训练参数
        dropout=0.1,            # Dropout率
        label_smoothing=0.1,    # 标签平滑
    )

    # 直接覆盖原模型权重
    source_path = PROJECT_ROOT / 'results' / 'models' / 'fall_detection_stage2' / 'weights' / 'best.pt'
    
    if source_path.exists():
        print(f"\n📁 覆盖原模型权重...")
        
        # 直接覆盖原模型位置
        target_paths = [
            PROJECT_ROOT / 'models' / 'yolo_fall' / 'best.pt',
            PROJECT_ROOT / 'fall_detection' / 'models' / 'best.pt'
        ]
        
        for target_path in target_paths:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_path, target_path)
            print(f"✅ 模型已覆盖到: {target_path}")
        
        print(f"\n🎉 第二阶段训练完成 - 原模型已更新！")
        
    else:
        print(f"\n⚠️  注意: 模型路径不存在: {source_path}")
        print(f"训练可能未成功完成，请检查错误信息")

if __name__ == "__main__":
    main()