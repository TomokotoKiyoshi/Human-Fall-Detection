#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8n模型层结构检查脚本

打印模型每一层的详细信息
"""

from ultralytics import YOLO
from ultralytics.nn.modules import *
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def inspect_model_layers():
    """检查YOLOv8n模型的所有层"""

    # 加载YOLOv8n模型
    model = YOLO('yolo_models/yolov8n.pt')

    print("=" * 80)
    print("YOLOv8n 模型结构分析")
    print("=" * 80)

    # 获取PyTorch模型
    pytorch_model = model.model

    # 打印模型基本信息
    print(f"\n模型类型: {type(pytorch_model)}")
    print(f"总参数量: {sum(p.numel() for p in pytorch_model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad):,}")

    print("\n" + "=" * 80)
    print("详细层结构:")
    print("=" * 80)

    # 遍历模型的所有模块
    for i, (name, module) in enumerate(pytorch_model.named_modules()):
        if len(name) == 0:  # 跳过根模块
            continue

        # 获取层类型
        module_type = module.__class__.__name__

        # 打印层信息
        print(f"\n层 {i}: {name}")
        print(f"  类型: {module_type}")

        # 根据不同层类型打印详细信息
        if isinstance(module, nn.Conv2d):
            print(f"  卷积层 Conv2d:")
            print(f"    输入通道: {module.in_channels}")
            print(f"    输出通道: {module.out_channels}")
            print(f"    卷积核大小: {module.kernel_size}")
            print(f"    步长: {module.stride}")
            print(f"    填充: {module.padding}")
            print(f"    参数量: {sum(p.numel() for p in module.parameters()):,}")

        elif isinstance(module, nn.BatchNorm2d):
            print(f"  批归一化 BatchNorm2d:")
            print(f"    特征数: {module.num_features}")
            print(f"    momentum: {module.momentum}")
            print(f"    eps: {module.eps}")

        elif isinstance(module, nn.SiLU):
            print(f"  激活函数 SiLU (Swish)")

        elif isinstance(module, nn.Upsample):
            print(f"  上采样 Upsample:")
            print(f"    缩放因子: {module.scale_factor}")
            print(f"    模式: {module.mode}")

        elif isinstance(module, nn.MaxPool2d):
            print(f"  最大池化 MaxPool2d:")
            print(f"    池化核大小: {module.kernel_size}")
            print(f"    步长: {module.stride}")

        elif module_type == "C2f":
            print(f"  C2f模块 (Cross Stage Partial):")
            if hasattr(module, 'cv1'):
                print(f"    cv1输出通道: {module.cv1.conv.out_channels if hasattr(module.cv1, 'conv') else 'N/A'}")
            if hasattr(module, 'cv2'):
                print(f"    cv2输出通道: {module.cv2.conv.out_channels if hasattr(module.cv2, 'conv') else 'N/A'}")

        elif module_type == "SPPF":
            print(f"  SPPF模块 (Spatial Pyramid Pooling Fast):")
            print(f"    池化核大小: {module.k if hasattr(module, 'k') else 'N/A'}")

        elif module_type == "Detect":
            print(f"  检测头 Detect:")
            print(f"    类别数: {module.nc if hasattr(module, 'nc') else 'N/A'}")
            print(f"    锚框数: {module.na if hasattr(module, 'na') else 'N/A'}")

        elif module_type == "DFL":
            print(f"  DFL模块 (Distribution Focal Loss)")

        elif module_type == "Conv":
            print(f"  Conv模块 (卷积+BN+激活):")
            if hasattr(module, 'conv'):
                print(f"    输入通道: {module.conv.in_channels}")
                print(f"    输出通道: {module.conv.out_channels}")
                print(f"    卷积核: {module.conv.kernel_size}")

        elif module_type == "Bottleneck":
            print(f"  Bottleneck模块:")
            if hasattr(module, 'cv1') and hasattr(module.cv1, 'conv'):
                print(f"    cv1输出: {module.cv1.conv.out_channels}")
            if hasattr(module, 'cv2') and hasattr(module.cv2, 'conv'):
                print(f"    cv2输出: {module.cv2.conv.out_channels}")

        elif module_type == "Concat":
            print(f"  Concat模块 (特征拼接):")
            print(f"    拼接维度: {module.d if hasattr(module, 'd') else 1}")


def inspect_frozen_layers():
    """检查冻结层的情况"""

    print("\n" + "=" * 80)
    print("冻结层分析 (freeze=10时):")
    print("=" * 80)

    model = YOLO('yolov8n.pt')
    pytorch_model = model.model

    # 模拟冻结前10层
    freeze_layers = 10

    layer_count = 0
    frozen_params = 0
    trainable_params = 0

    for i, (name, param) in enumerate(pytorch_model.named_parameters()):
        layer_count += 1
        param_count = param.numel()

        if i < freeze_layers:
            frozen_params += param_count
            status = "冻结"
        else:
            trainable_params += param_count
            status = "可训练"

        if i < 15 or i >= layer_count - 5:  # 只打印前15层和最后5层
            print(f"参数 {i:3d}: {status:8s} | {name:50s} | 形状: {list(param.shape)}")

    print(f"\n总结:")
    print(f"  总参数层数: {layer_count}")
    print(f"  冻结参数量: {frozen_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params/(frozen_params+trainable_params)*100:.2f}%")


def inspect_detection_head():
    """专门检查检测头的结构"""

    print("\n" + "=" * 80)
    print("检测头 (Detection Head) 详细分析:")
    print("=" * 80)

    model = YOLO('yolov8n.pt')
    pytorch_model = model.model

    # 找到Detect模块
    for name, module in pytorch_model.named_modules():
        if module.__class__.__name__ == "Detect":
            print(f"\n检测头位置: {name}")
            print(f"  类别数 (nc): {module.nc if hasattr(module, 'nc') else 'N/A'}")
            print(f"  每个位置的锚框数 (na): {module.na if hasattr(module, 'na') else 'N/A'}")
            print(f"  输出通道数 (no): {module.no if hasattr(module, 'no') else 'N/A'}")
            print(f"  步长 (stride): {module.stride if hasattr(module, 'stride') else 'N/A'}")

            # 检查cv2和cv3层（分类和回归头）
            if hasattr(module, 'cv2'):
                print("\n  分类头 (cv2):")
                for i, cv2_module in enumerate(module.cv2):
                    print(f"    尺度 {i}: {cv2_module}")

            if hasattr(module, 'cv3'):
                print("\n  回归头 (cv3):")
                for i, cv3_module in enumerate(module.cv3):
                    print(f"    尺度 {i}: {cv3_module}")


def main():
    """主函数"""

    # 1. 检查模型所有层
    inspect_model_layers()

    # 2. 检查冻结层情况
    inspect_frozen_layers()

    # 3. 专门检查检测头
    inspect_detection_head()

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()