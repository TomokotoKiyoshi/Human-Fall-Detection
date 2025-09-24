#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨度比特征提取模块

计算人体关键点的跨度比 (span_ratio = y_span/x_span)
用于判断人体姿态的延展方向
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_span_ratio(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """
    计算人体关键点的跨度比

    跨度比 = y_span / x_span
    - 站立时：竖直跨度大，水平跨度小，比值大（通常 > 2）
    - 躺倒时：水平跨度大，竖直跨度小，比值小（通常 < 0.5）

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        float: 跨度比值，如果计算失败返回None
    """

    if keypoints is None:
        return None

    # 收集所有有效的关键点坐标
    x_coords = []
    y_coords = []

    for name, point in keypoints.items():
        if point is not None and len(point) == 2:
            x, y = point
            if x is not None and y is not None:
                x_coords.append(x)
                y_coords.append(y)

    # 至少需要2个点才能计算跨度
    if len(x_coords) < 2 or len(y_coords) < 2:
        return None

    # 计算x和y方向的跨度
    x_span = max(x_coords) - min(x_coords)
    y_span = max(y_coords) - min(y_coords)

    # 避免除零错误
    if x_span < 1e-6:
        return None

    # 计算跨度比
    span_ratio = y_span / x_span

    return float(span_ratio)