#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扁平率特征提取模块

计算人体关键点的扁平率 (flat_ratio = σy/σx)
用于判断人体姿态是站立还是躺倒
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_flat_ratio(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """
    计算人体关键点的扁平率

    扁平率 = σy / σx
    - 站立时：y方向离散度大，σy较大，flat_ratio > 1
    - 躺倒时：x方向离散度大，σy较小，flat_ratio < 1

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        float: 扁平率值，如果计算失败返回None
    """

    if keypoints is None:
        return None

    # 收集所有有效的关键点坐标
    valid_points = []
    for name, point in keypoints.items():
        if point is not None and len(point) == 2:
            x, y = point
            if x is not None and y is not None:
                valid_points.append([x, y])

    # 至少需要3个点才能计算有意义的标准差
    if len(valid_points) < 3:
        return None

    # 转换为numpy数组
    points = np.array(valid_points)

    # 计算x和y方向的标准差
    sigma_x = np.std(points[:, 0])
    sigma_y = np.std(points[:, 1])

    # 避免除零错误
    if sigma_x < 1e-6:
        return None

    # 计算扁平率
    flat_ratio = sigma_y / sigma_x

    return float(flat_ratio)