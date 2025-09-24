#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频目标检测脚本

使用YOLOv8n模型处理视频
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


def process_video(video_path: Path, output_path: Path):
    """
    处理视频文件

    参数:
        video_path: 输入视频路径
        output_path: 输出视频路径
    """
    # 检查输入视频
    if not video_path.exists():
        print(f"错误: 视频文件不存在: {video_path}")
        return

    # 加载YOLOv8n模型
    model_path = PROJECT_ROOT / 'yolo_models' / 'yolov8n.pt'
    if not model_path.exists():
        print(f"模型不存在，正在下载到: {model_path}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO('yolov8n.pt')
        # 保存模型到指定位置
        import shutil
        if Path('yolov8n.pt').exists():
            shutil.move('yolov8n.pt', str(model_path))
    else:
        model = YOLO(str(model_path))

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))

    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")

    # 创建输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # COCO类别名称
    class_names = model.names

    # 处理视频
    frame_count = 0
    total_detections = 0
    person_detections = 0

    print("\n开始处理视频...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 使用YOLO进行目标检测，只检测人（class 0）
        results = model(frame, verbose=False, device='cuda', classes=[0])

        # 处理检测结果
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 获取类别和置信度
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = class_names[cls]

                    # 只处理人体检测（实际上现在只会检测到人）
                    if label == 'person':
                        person_detections += 1
                        total_detections += 1
                        color = (0, 255, 0)  # 绿色框表示人
                        thickness = 3

                        # 绘制边界框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                        # 添加标签
                        label_text = f"Person: {conf:.2f}"
                        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20

                        # 标签背景
                        cv2.rectangle(frame,
                                     (x1, label_y - label_size[1] - 4),
                                     (x1 + label_size[0], label_y + 4),
                                     color, -1)

                        # 标签文字
                        cv2.putText(frame, label_text,
                                   (x1, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 添加统计信息
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons detected: {person_detections}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 写入输出视频
        out.write(frame)

        # 显示进度
        if frame_count % 30 == 0:  # 每30帧显示一次
            progress = frame_count / total_frames * 100
            print(f"  进度: {progress:.1f}% ({frame_count}/{total_frames})")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n处理完成!")
    print(f"  总帧数: {frame_count}")
    print(f"  检测到的人体次数: {person_detections}")
    print(f"  输出视频: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("YOLOv8n 视频目标检测")
    print("=" * 60)

    # 设置路径
    video_path = Path("D:/Human-Fall-Detection/test.mp4")
    output_dir = PROJECT_ROOT / 'results' / 'video_detection'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'test_detection.mp4'

    # 处理视频
    process_video(video_path, output_path)


if __name__ == "__main__":
    main()