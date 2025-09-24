#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
摔倒检测推理脚本

使用微调后的YOLOv8n模型检测正常/摔倒状态
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置路径
INPUT_VIDEO_PATH = PROJECT_ROOT / 'test_src' / 'test.mp4'  # 输入视频路径
OUTPUT_VIDEO_PATH = PROJECT_ROOT / 'results' / 'fall_detection' / 'output.mp4'  # 输出视频路径

# 模型路径（按优先级尝试）
MODEL_PATHS = [
    PROJECT_ROOT / 'fall_detection' / 'models' / 'best.pt',  # 打包位置（优先）
    PROJECT_ROOT / 'results' / 'models' / 'fall_detection' / 'weights' / 'best.pt',  # 原始位置
]

import cv2
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

def detect_video(video_path, model_path):
    """检测视频中的摔倒"""

    # 加载模型
    model = YOLO(model_path)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出视频设置
    output_path = OUTPUT_VIDEO_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # 统计
    frame_count = 0
    fall_count = 0
    normal_count = 0

    print(f"开始处理视频: {video_path.name}")
    print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")

    # 使用tqdm显示进度条
    pbar = tqdm(total=total_frames, desc="处理进度", unit="帧")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pbar.update(1)

        # 检测
        results = model(frame, verbose=False, device='cuda', conf=0.25)

        # 绘制结果
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # 根据类别选择颜色和标签
                    if cls == 0:  # 正常
                        color = (0, 255, 0)  # 绿色
                        label = f"Normal: {conf:.2f}"
                        thickness = 2
                        frame_normal = True
                        normal_count += 1
                    else:  # 摔倒 (cls == 1)
                        color = (0, 0, 255)  # 红色
                        label = f"FALL: {conf:.2f}"
                        thickness = 3
                        frame_fall = True
                        fall_count += 1

                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    # 添加标签背景
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame,
                                 (x1, y1 - label_size[1] - 10),
                                 (x1 + label_size[0], y1),
                                 color, -1)

                    # 添加标签文字
                    cv2.putText(frame, label,
                               (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)

    pbar.close()
    cap.release()
    out.release()

    print("\n检测完成！")
    print(f"输出视频: {output_path}")

def main():
    # 查找可用的模型
    model_path = None
    for path in MODEL_PATHS:
        if path.exists():
            model_path = path
            print(f"使用模型: {model_path}")
            break

    if model_path is None:
        print("模型文件未找到！尝试过以下位置:")
        for path in MODEL_PATHS:
            print(f"  - {path}")
        print("\n请先运行 scripts/training/train_yolo_fall.py 训练模型")
        return

    # 测试视频
    if INPUT_VIDEO_PATH.exists():
        detect_video(INPUT_VIDEO_PATH, model_path)
    else:
        print(f"测试视频不存在: {INPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()