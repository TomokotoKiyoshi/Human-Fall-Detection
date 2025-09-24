#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
形状特征提取脚本

该脚本从已有的姿态关键点数据中提取形状/分布特征。
输入：features/pose_features/下的关键点数据
输出：features/shape_features/下的形状特征数据
"""

import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from tqdm import tqdm

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入特征提取模块
from src.feature_extraction import (
    calculate_flat_ratio,
    calculate_span_ratio,
    calculate_torso_ankle_distance,
    calculate_torso_head_distance,
    calculate_trunk_angle,
    calculate_thigh_angles,
    calculate_shin_angles
)

# 输入输出路径
INPUT_DIR = PROJECT_ROOT / 'features' / 'pose_features'
OUTPUT_DIR = PROJECT_ROOT / 'features' / 'shape_features'

# 数据集划分
SPLITS = ['train', 'val', 'test']


def extract_shape_features_from_keypoints(keypoints: Dict) -> Dict[str, Any]:
    """
    从关键点数据中提取形状特征

    参数:
        keypoints: 关键点字典

    返回:
        包含所有形状特征的字典
    """

    features = {}

    # 1. 扁平率
    flat_ratio = calculate_flat_ratio(keypoints)
    features['flat_ratio'] = flat_ratio

    # 2. 跨度比
    span_ratio = calculate_span_ratio(keypoints)
    features['span_ratio'] = span_ratio

    # 3. 躯干与脚踝距离
    torso_ankle = calculate_torso_ankle_distance(keypoints)
    features['torso_ankle_distance'] = torso_ankle.get('euclidean')
    features['torso_ankle_distance_x'] = torso_ankle.get('x_distance')
    features['torso_ankle_distance_y'] = torso_ankle.get('y_distance')

    # 4. 躯干与头部距离
    torso_head = calculate_torso_head_distance(keypoints)
    features['torso_head_distance_x'] = torso_head.get('x_distance')
    features['torso_head_distance_y'] = torso_head.get('y_distance')

    # 5. 躯干角度
    trunk_angle = calculate_trunk_angle(keypoints)
    features['trunk_angle'] = trunk_angle

    # 6. 大腿角度
    thigh_angles = calculate_thigh_angles(keypoints)
    features['left_thigh_angle'] = thigh_angles.get('left')
    features['right_thigh_angle'] = thigh_angles.get('right')

    # 7. 小腿角度
    shin_angles = calculate_shin_angles(keypoints)
    features['left_shin_angle'] = shin_angles.get('left')
    features['right_shin_angle'] = shin_angles.get('right')

    return features


def process_split(split: str) -> Dict[str, Any]:
    """
    处理一个数据集划分

    参数:
        split: 数据集划分名称 ('train', 'val', 'test')

    返回:
        特征字典
    """

    input_file = INPUT_DIR / f'{split}.npy'

    if not input_file.exists():
        print(f"警告: 输入文件不存在: {input_file}")
        return {}

    print(f"加载 {split} 数据集的关键点数据...")
    pose_data = np.load(input_file, allow_pickle=True).item()

    print(f"找到 {len(pose_data)} 个样本")

    shape_features = {}
    failed_count = 0

    # 处理每个样本
    for filename, data in tqdm(pose_data.items(), desc=f"提取 {split} 集特征"):
        keypoints = data.get('keypoints')
        meta = data.get('meta', [])

        if keypoints is not None:
            # 提取形状特征
            features = extract_shape_features_from_keypoints(keypoints)

            # 组合结果
            shape_features[filename] = {
                'shape_features': features,
                'meta': meta
            }
        else:
            # 无关键点时，所有特征都为None
            shape_features[filename] = {
                'shape_features': {
                    'flat_ratio': None,
                    'span_ratio': None,
                    'torso_ankle_distance': None,
                    'torso_ankle_distance_x': None,
                    'torso_ankle_distance_y': None,
                    'torso_head_distance_x': None,
                    'torso_head_distance_y': None,
                    'trunk_angle': None,
                    'left_thigh_angle': None,
                    'right_thigh_angle': None,
                    'left_shin_angle': None,
                    'right_shin_angle': None
                },
                'meta': meta
            }
            failed_count += 1

    if failed_count > 0:
        print(f"警告: {failed_count} 个样本无关键点数据")

    return shape_features


def main():
    """主函数：处理所有数据集划分"""

    print("=" * 50)
    print("形状特征提取程序启动")
    print("=" * 50)

    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {OUTPUT_DIR}")

    # 处理各个数据集划分
    for split in SPLITS:
        print(f"\n开始处理 {split} 数据集...")

        # 提取特征
        features = process_split(split)

        if features:
            # 保存特征文件
            output_path = OUTPUT_DIR / f'{split}.npy'
            np.save(output_path, features, allow_pickle=True)
            print(f"成功保存 {len(features)} 个样本的特征到 {output_path}")
        else:
            print(f"警告: {split} 数据集未生成任何特征")

    print("\n" + "=" * 50)
    print("形状特征提取完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()