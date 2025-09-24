#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv8-Pose 特征提取脚本

该脚本使用YOLOv8-Pose模型从图像中提取人体姿态关键点特征。
输出格式遵循 docs/feature_format.md 中定义的规范。
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入数据加载模块
from src.utils.data_loader import load_image_and_meta

# ============ 加载配置 ============
# 加载姿态配置
POSE_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'pose_config.yaml'
with open(POSE_CONFIG_PATH, 'r', encoding='utf-8') as f:
    pose_config = yaml.safe_load(f)

# 从配置文件获取关键点名称
KEYPOINT_NAMES: List[str] = pose_config['keypoints']['names']

# 模型路径
MODEL_DIR = PROJECT_ROOT / 'yolo_models'
MODEL_NAME = 'yolov8s-pose.pt'
MODEL_PATH = MODEL_DIR / MODEL_NAME

# 输出路径
OUTPUT_DIR = PROJECT_ROOT / 'features' / 'pose_features'

# 数据集划分
SPLITS = ['train', 'val', 'test']
# ===================================


def extract_features(model: YOLO, image: np.ndarray, meta: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    从图像中提取姿态关键点特征。

    参数:
        model: YOLOv8-Pose模型实例
        image: numpy array格式的图像
        meta: 标注信息列表

    返回:
        特征字典，包含:
        - keypoints: 17个关键点的坐标字典或None
        - meta: 标注信息列表
    """
    if image is None:
        return None

    # 使用YOLOv8-Pose进行推理
    try:
        results = model(image, verbose=False, device='cuda')
    except Exception as e:
        print(f"错误: 模型推理失败: {e}")
        return None

    feature_dict = {}

    # 提取第一个检测到的人的关键点
    if (len(results[0].keypoints.xy) > 0 and
        results[0].keypoints.xy[0].shape[0] > 0):

        keypoints_xy = results[0].keypoints.xy[0].cpu().numpy()

        keypoints = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            if i < len(keypoints_xy):
                x, y = keypoints_xy[i]
                # 保存为float类型的像素坐标
                keypoints[name] = (float(x), float(y))
            else:
                keypoints[name] = (None, None)

        feature_dict['keypoints'] = keypoints
    else:
        # 未检测到人时，keypoints为None
        feature_dict['keypoints'] = None

    # 直接使用传入的meta信息
    feature_dict['meta'] = meta

    return feature_dict


def process_split(split: str, model: YOLO) -> Dict[str, Any]:
    """
    处理数据集的一个划分（train/val/test）。

    参数:
        split: 数据集划分名称 ('train', 'val', 'test')
        model: YOLOv8-Pose模型实例

    返回:
        特征字典，键为图像文件名，值为特征数据
    """
    print(f"加载 {split} 数据集...")

    # 使用data_loader加载数据
    try:
        data = load_image_and_meta(split, root_dir=PROJECT_ROOT)
    except Exception as e:
        print(f"错误: 加载数据失败: {e}")
        return {}

    if not data:
        print(f"警告: {split} 数据集为空")
        return {}

    print(f"找到 {len(data)} 张图像待处理")

    features = {}
    failed_count = 0

    # 处理每张图像
    for filename, item in tqdm(data.items(), desc=f"处理 {split} 集"):
        image = item['image']
        meta = item['meta']

        feature = extract_features(model, image, meta)

        if feature is not None:
            features[filename] = feature
        else:
            failed_count += 1

    if failed_count > 0:
        print(f"警告: 处理失败 {failed_count} 张图像")

    return features


def main():
    """主函数：初始化模型并处理所有数据集划分。"""

    print("=" * 50)
    print("YOLOv8-Pose 特征提取程序启动")
    print("=" * 50)

    # 检查GPU可用性
    if not torch.cuda.is_available():
        print("错误: CUDA不可用！请检查GPU驱动和PyTorch CUDA版本")
        return

    print(f"检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

    # 初始化YOLOv8-Pose模型
    print("加载YOLOv8-Pose模型...")

    # 如果模型不存在，自动下载到指定目录
    if not MODEL_PATH.exists():
        print(f"模型不存在，将下载到: {MODEL_PATH}")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        model = YOLO(str(MODEL_PATH))
        model.to('cuda')  # 强制使用GPU
        print(f"模型加载成功 (使用GPU) - 位置: {MODEL_PATH}")
    except Exception as e:
        print(f"错误: 模型加载失败: {e}")
        return

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

    # 处理各个数据集划分
    for split in SPLITS:
        print(f"\n开始处理 {split} 数据集...")

        # 提取特征
        features = process_split(split, model)

        if features:
            # 保存特征文件
            output_path = OUTPUT_DIR / f'{split}.npy'
            np.save(output_path, features, allow_pickle=True)
            print(f"成功保存 {len(features)} 个样本的特征到 {output_path}")
        else:
            print(f"警告: {split} 数据集未生成任何特征")

    print("\n" + "=" * 50)
    print("特征提取完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()