#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载工具函数

提供简单的接口用于加载图像和标签数据
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np


def load_image_and_meta(
    split: str,
    root_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    加载指定数据集划分的所有图像和标签

    参数:
        split: 数据集划分 ('train', 'val', 'test')
        root_dir: 项目根目录，默认自动检测

    返回:
        Dict[str, Dict]: 字典，键为文件名，值包含:
            - 'image': numpy array格式的图像 (H, W, 3)
            - 'meta': List[Dict] YOLO格式标签列表，每个Dict包含:
                - cls: 类别 (0: 未摔倒, 1: 摔倒)
                - x_center: 中心x坐标（归一化）
                - y_center: 中心y坐标（归一化）
                - width: 宽度（归一化）
                - height: 高度（归一化）

    示例:
        >>> data = load_image_and_meta('train')
        >>> for filename, item in data.items():
        ...     img = item['image']  # numpy array
        ...     meta = item['meta']   # 标签列表
    """

    # 确定项目根目录
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent.parent.parent
    else:
        root_dir = Path(root_dir)

    # 定义路径
    img_dir = root_dir / 'data' / 'images' / split
    label_dir = root_dir / 'data' / 'labels' / split

    # 检查目录是否存在
    if not img_dir.exists():
        raise ValueError(f"图像目录不存在: {img_dir}")

    # 获取所有图像文件
    image_files = sorted(
        list(img_dir.glob('*.png')) +
        list(img_dir.glob('*.jpg')) +
        list(img_dir.glob('*.jpeg'))
    )

    if not image_files:
        print(f"警告: 未找到图像文件: {img_dir}")
        return {}

    # 加载数据
    data = {}

    for img_path in image_files:
        # 加载图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告: 无法读取图像: {img_path}")
            continue

        # 加载标签
        label_path = label_dir / (img_path.stem + '.txt')
        meta = parse_yolo_label(label_path)

        # 存储数据
        data[img_path.name] = {
            'image': img,
            'meta': meta
        }

    return data


def parse_yolo_label(label_path: Path) -> List[Dict[str, Any]]:
    """
    解析YOLO格式标签文件

    参数:
        label_path: 标签文件路径

    返回:
        List[Dict]: 标签列表，每个Dict包含:
            - cls: 类别ID (0: 未摔倒, 1: 摔倒)
            - x_center: 边界框中心x坐标（归一化）
            - y_center: 边界框中心y坐标（归一化）
            - width: 边界框宽度（归一化）
            - height: 边界框高度（归一化）
    """
    meta = []

    if not label_path.exists():
        return meta

    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    meta.append({
                        'cls': cls,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
    except Exception as e:
        print(f"解析标签文件失败 {label_path}: {e}")

    return meta
