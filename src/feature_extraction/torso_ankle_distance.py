#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
躯干与脚踝距离特征提取模块

计算躯干中心与脚踝中心之间的距离
包括欧氏距离和分方向距离
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_torso_ankle_distance(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """
    计算躯干中心与脚踝中心之间的距离

    返回三个值：
    1. 欧氏距离
    2. x方向距离（水平分离度）
    3. y方向距离（垂直分离度）

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        Dict: 包含三个距离值的字典
            - 'euclidean': 欧氏距离
            - 'x_distance': x方向距离
            - 'y_distance': y方向距离
    """

    result = {
        'euclidean': None,
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

    # 获取脚踝点
    ankle_points = []
    for name in ['left_ankle', 'right_ankle']:
        if name in keypoints and keypoints[name] is not None:
            x, y = keypoints[name]
            if x is not None and y is not None:
                ankle_points.append([x, y])

    # 至少需要一个脚踝点
    if len(ankle_points) == 0:
        return result

    # 计算脚踝中心
    ankle_center = np.mean(ankle_points, axis=0)
    ankle_x, ankle_y = ankle_center

    # 计算欧氏距离
    euclidean = np.sqrt((center_x - ankle_x)**2 + (center_y - ankle_y)**2)
    result['euclidean'] = float(euclidean)

    # 计算x方向距离（水平分离度）
    x_distance = abs(center_x - ankle_x)
    result['x_distance'] = float(x_distance)

    # 计算y方向距离（垂直分离度）
    y_distance = abs(center_y - ankle_y)
    result['y_distance'] = float(y_distance)

    return result