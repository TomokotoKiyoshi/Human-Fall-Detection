#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
躯干与头部距离特征提取模块

计算躯干中心与头部中心在x和y方向上的距离
用于判断身体倾斜状态
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_torso_head_distance(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """
    计算躯干中心与头部中心之间的距离

    返回两个值：
    1. x方向距离（水平偏移）
    2. y方向距离（垂直偏移）

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        Dict: 包含两个距离值的字典
            - 'x_distance': x方向距离
            - 'y_distance': y方向距离
    """

    result = {
        'x_distance': None,
        'y_distance': None
    }

    if keypoints is None:
        return result

    # 获取躯干的四个关键点
    required_torso_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    torso_points = []

    for name in required_torso_points:
        if name in keypoints and keypoints[name] is not None:
            x, y = keypoints[name]
            if x is not None and y is not None:
                torso_points.append([x, y])

    # 需要至少3个躯干点才能计算有意义的中心
    if len(torso_points) < 3:
        return result

    # 计算躯干中心
    torso_center = np.mean(torso_points, axis=0)
    center_x, center_y = torso_center

    # 获取头部关键点（鼻子、眼睛、耳朵）
    head_point_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']
    head_points = []

    for name in head_point_names:
        if name in keypoints and keypoints[name] is not None:
            x, y = keypoints[name]
            if x is not None and y is not None:
                head_points.append([x, y])

    # 至少需要2个头部点才能计算有意义的中心
    if len(head_points) < 2:
        return result

    # 计算头部中心
    head_center = np.mean(head_points, axis=0)
    head_x, head_y = head_center

    # 计算x方向距离（水平偏移）
    x_distance = abs(center_x - head_x)
    result['x_distance'] = float(x_distance)

    # 计算y方向距离（垂直偏移）
    y_distance = abs(center_y - head_y)
    result['y_distance'] = float(y_distance)

    return result