#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小腿角度特征提取模块

计算左右小腿分别与图像y轴之间的夹角
用于判断膝盖弯曲程度和姿态
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_shin_angles(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """
    计算左右小腿与图像y轴的夹角

    夹角范围：0° ~ 90°
    - 站立时：小腿垂直，夹角接近0°
    - 坐姿时：小腿通常垂直或稍微倾斜，夹角0°~30°
    - 躺倒时：小腿水平，夹角接近90°

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        Dict: 包含左右小腿角度的字典
            - 'left': 左小腿角度
            - 'right': 右小腿角度
    """

    result = {
        'left': None,
        'right': None
    }

    if keypoints is None:
        return result

    # 计算左小腿角度
    if ('left_knee' in keypoints and keypoints['left_knee'] is not None and
        'left_ankle' in keypoints and keypoints['left_ankle'] is not None):

        knee_x, knee_y = keypoints['left_knee']
        ankle_x, ankle_y = keypoints['left_ankle']

        if all(v is not None for v in [knee_x, knee_y, ankle_x, ankle_y]):
            # 小腿向量（从膝盖指向脚踝）
            shin_vector = np.array([ankle_x - knee_x, ankle_y - knee_y])
            shin_length = np.linalg.norm(shin_vector)

            if shin_length > 1e-6:
                # 计算与y轴的夹角
                cos_angle = abs(shin_vector[1]) / shin_length
                cos_angle = np.clip(cos_angle, 0, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                result['left'] = float(angle_deg)

    # 计算右小腿角度
    if ('right_knee' in keypoints and keypoints['right_knee'] is not None and
        'right_ankle' in keypoints and keypoints['right_ankle'] is not None):

        knee_x, knee_y = keypoints['right_knee']
        ankle_x, ankle_y = keypoints['right_ankle']

        if all(v is not None for v in [knee_x, knee_y, ankle_x, ankle_y]):
            # 小腿向量（从膝盖指向脚踝）
            shin_vector = np.array([ankle_x - knee_x, ankle_y - knee_y])
            shin_length = np.linalg.norm(shin_vector)

            if shin_length > 1e-6:
                # 计算与y轴的夹角
                cos_angle = abs(shin_vector[1]) / shin_length
                cos_angle = np.clip(cos_angle, 0, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                result['right'] = float(angle_deg)

    return result