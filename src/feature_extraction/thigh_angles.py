#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
大腿角度特征提取模块

计算左右大腿分别与图像y轴之间的夹角
用于判断腿部姿态和动作状态
"""

import numpy as np
from typing import Dict, Optional, Tuple


def calculate_thigh_angles(keypoints: Dict[str, Tuple[float, float]]) -> Dict[str, Optional[float]]:
    """
    计算左右大腿与图像y轴的夹角

    夹角范围：0° ~ 90°
    - 站立时：大腿垂直，夹角接近0°
    - 坐姿时：大腿水平，夹角接近90°

    参数:
        keypoints: 关键点字典，格式为 {关键点名称: (x, y)}

    返回:
        Dict: 包含左右大腿角度的字典
            - 'left': 左大腿角度
            - 'right': 右大腿角度
    """

    result = {
        'left': None,
        'right': None
    }

    if keypoints is None:
        return result

    # 计算左大腿角度
    if ('left_hip' in keypoints and keypoints['left_hip'] is not None and
        'left_knee' in keypoints and keypoints['left_knee'] is not None):

        hip_x, hip_y = keypoints['left_hip']
        knee_x, knee_y = keypoints['left_knee']

        if all(v is not None for v in [hip_x, hip_y, knee_x, knee_y]):
            # 大腿向量（从髋部指向膝盖）
            thigh_vector = np.array([knee_x - hip_x, knee_y - hip_y])
            thigh_length = np.linalg.norm(thigh_vector)

            if thigh_length > 1e-6:
                # 计算与y轴的夹角
                cos_angle = abs(thigh_vector[1]) / thigh_length
                cos_angle = np.clip(cos_angle, 0, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                result['left'] = float(angle_deg)

    # 计算右大腿角度
    if ('right_hip' in keypoints and keypoints['right_hip'] is not None and
        'right_knee' in keypoints and keypoints['right_knee'] is not None):

        hip_x, hip_y = keypoints['right_hip']
        knee_x, knee_y = keypoints['right_knee']

        if all(v is not None for v in [hip_x, hip_y, knee_x, knee_y]):
            # 大腿向量（从髋部指向膝盖）
            thigh_vector = np.array([knee_x - hip_x, knee_y - hip_y])
            thigh_length = np.linalg.norm(thigh_vector)

            if thigh_length > 1e-6:
                # 计算与y轴的夹角
                cos_angle = abs(thigh_vector[1]) / thigh_length
                cos_angle = np.clip(cos_angle, 0, 1)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)
                result['right'] = float(angle_deg)

    return result