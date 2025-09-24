#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human Fall Detection Package

基于YOLOv8的人体摔倒检测系统

Example:
    简单使用::

        import fall_detection

        # 检测视频
        result = fall_detection.process_video("test.mp4", "output.mp4")
        print(f"检测到摔倒: {result['statistics']['fall_count']}次")

        # 检测图片
        result = fall_detection.process_image("test.jpg")
        if result["fall_detected"]:
            print("警告：检测到摔倒！")

    高级使用::

        from fall_detection import FallDetector

        # 创建检测器
        detector = FallDetector(confidence=0.6)

        # 检测单张图片
        import cv2
        img = cv2.imread("test.jpg")
        result = detector.detect_image(img)
"""

from .detector import FallDetector
from .api import (
    process_video,
    process_image,
    detect_from_array,
)

__version__ = "0.2.0"
__author__ = "Human Fall Detection Team"

# 导出的API
__all__ = [
    # 类
    "FallDetector",
    # 函数
    "process_video",
    "process_image",
    "detect_from_array",
]

# 包信息
def get_info():
    """获取包信息"""
    return {
        "version": __version__,
        "author": __author__,
    }