#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
躯干角度特征提取模块

计算躯干主轴与图像y轴之间的夹角
用于判断人体是站立还是躺倒
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_trunk_angle(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """
    计算躯干与图像y轴的夹角

    夹角范围：0° ~ 90°
    - 站立时：躯干垂直，夹角接近0°
    - 躺倒时：躯干水平，夹角接近90°

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        float: 夹角（度数），如果计算失败返回None
    """

    if keypoints is None:
        return None

    # 获取肩部点
    shoulder_points = []
    for name in ['left_shoulder', 'right_shoulder']:
        if name in keypoints and keypoints[name] is not None:
            x, y = keypoints[name]
            if x is not None and y is not None:
                shoulder_points.append([x, y])

    # 获取髋部点
    hip_points = []
    for name in ['left_hip', 'right_hip']:
        if name in keypoints and keypoints[name] is not None:
            x, y = keypoints[name]
            if x is not None and y is not None:
                hip_points.append([x, y])

    # 需要至少各有一个肩部和髋部点
    if len(shoulder_points) == 0 or len(hip_points) == 0:
        return None

    # 计算肩部中心和髋部中心
    shoulder_center = np.mean(shoulder_points, axis=0)
    hip_center = np.mean(hip_points, axis=0)

    # 计算躯干向量（从髋部指向肩部）
    trunk_vector = shoulder_center - hip_center
    trunk_x, trunk_y = trunk_vector

    # 计算向量长度
    trunk_length = np.sqrt(trunk_x**2 + trunk_y**2)

    # 避免零向量
    if trunk_length < 1e-6:
        return None

    # 计算与y轴的夹角
    # y轴向量为(0, 1)，但在图像坐标系中y向下，所以使用(0, -1)
    # 这里使用绝对值确保角度在0-90度之间
    cos_angle = abs(trunk_y) / trunk_length

    # 确保cos值在有效范围内
    cos_angle = np.clip(cos_angle, 0, 1)

    # 计算角度（弧度转度数）
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)