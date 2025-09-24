#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取模块包

该包包含用于从人体姿态关键点提取语义特征的各个模块。
每个模块实现一个特定的特征计算。
"""

from .flat_ratio import calculate_flat_ratio
from .span_ratio import calculate_span_ratio
from .torso_ankle_distance import calculate_torso_ankle_distance
from .torso_head_distance import calculate_torso_head_distance
from .trunk_angle import calculate_trunk_angle
from .thigh_angles import calculate_thigh_angles
from .shin_angles import calculate_shin_angles

__all__ = [
    'calculate_flat_ratio',
    'calculate_span_ratio',
    'calculate_torso_ankle_distance',
    'calculate_torso_head_distance',
    'calculate_trunk_angle',
    'calculate_thigh_angles',
    'calculate_shin_angles'
]