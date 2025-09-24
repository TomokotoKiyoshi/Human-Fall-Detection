#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
姿态关键点可视化GUI

该脚本提供了一个交互式GUI界面，用于可视化YOLOv8-Pose模型提取的人体姿态关键点特征。

主要功能：
1. 加载features/pose_features/下的train/val/test特征文件（.npy格式）
2. 显示原始图像并叠加17个COCO格式关键点
3. 根据pose_config.yaml中定义的连接关系绘制骨架
4. 显示边界框和类别标签（0:未摔倒, 1:摔倒）
5. 提供数据集切换功能（train/val/test）
6. 提供样本导航功能（上一张/下一张/跳转）
7. 提供显示选项控制（关键点/骨架/边界框/标签开关）
8. 显示关键点置信度和坐标信息
9. 支持键盘快捷键操作

使用说明：
- 左侧面板：显示选项和控制按钮
- 右侧画布：显示图像和可视化结果
- 键盘快捷键：
  - 左箭头/A：上一张
  - 右箭头/D：下一张
  - Space：切换骨架显示
  - K：切换关键点显示
  - B：切换边界框显示
  - L：切换标签显示

依赖：
- tkinter: GUI界面
- numpy: 特征文件加载
- cv2: 图像处理
- yaml: 配置文件读取
- PIL: 图像显示
- data_loader: 原始图像加载
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import yaml
from PIL import Image, ImageTk

# 添加项目根目录到系统路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# 导入数据加载模块
from src.utils.data_loader import load_image_and_meta


class PoseKeypointsGUI:
    """姿态关键点可视化GUI类"""

    def __init__(self, root: tk.Tk):
        """初始化GUI"""
        self.root = root
        self.root.title("人体姿态关键点可视化系统")
        self.root.geometry("1400x900")

        # 加载配置
        self.load_config()

        # 数据相关变量
        self.current_split = "train"
        self.current_index = 0
        self.features_data = {}
        self.image_names = []
        self.original_images = {}

        # 显示控制变量
        self.show_keypoints = tk.BooleanVar(value=True)
        self.show_skeleton = tk.BooleanVar(value=True)
        self.show_bbox = tk.BooleanVar(value=True)
        self.show_label = tk.BooleanVar(value=True)

        # 初始化GUI组件
        self.setup_gui()

        # 加载初始数据
        self.load_split_data(self.current_split)

        # 绑定键盘事件
        self.bind_keyboard_events()

    def load_config(self):
        """加载姿态配置文件"""
        config_path = PROJECT_ROOT / 'configs' / 'pose_config.yaml'
        with open(config_path, 'r', encoding='utf-8') as f:
            pose_config = yaml.safe_load(f)

        self.keypoint_names = pose_config['keypoints']['names']
        self.skeleton_connections = pose_config['keypoints']['all_connections']

        # 定义关键点颜色（按身体部位分组）
        self.keypoint_colors = {
            # 头部 - 红色系
            'nose': (255, 0, 0),
            'left_eye': (255, 50, 50),
            'right_eye': (255, 50, 50),
            'left_ear': (255, 100, 100),
            'right_ear': (255, 100, 100),
            # 躯干 - 绿色系
            'left_shoulder': (0, 255, 0),
            'right_shoulder': (0, 255, 0),
            'left_hip': (0, 200, 0),
            'right_hip': (0, 200, 0),
            # 手臂 - 蓝色系
            'left_elbow': (0, 0, 255),
            'right_elbow': (0, 0, 255),
            'left_wrist': (100, 100, 255),
            'right_wrist': (100, 100, 255),
            # 腿部 - 黄色系
            'left_knee': (255, 255, 0),
            'right_knee': (255, 255, 0),
            'left_ankle': (255, 200, 0),
            'right_ankle': (255, 200, 0),
        }

        # 骨架颜色
        self.skeleton_color = (0, 255, 255)  # 青色

    def setup_gui(self):
        """设置GUI组件"""
        # 左侧控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="nsew")

        # 数据集选择
        ttk.Label(control_frame, text="数据集选择：", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=5, sticky="w")

        dataset_frame = ttk.Frame(control_frame)
        dataset_frame.grid(row=1, column=0, pady=5)

        self.split_var = tk.StringVar(value="train")
        for i, split in enumerate(['train', 'val', 'test']):
            ttk.Radiobutton(dataset_frame, text=split, variable=self.split_var,
                          value=split, command=self.on_split_change).grid(row=0, column=i, padx=5)

        # 样本导航
        ttk.Label(control_frame, text="样本导航：", font=("Arial", 12, "bold")).grid(row=2, column=0, pady=(20, 5), sticky="w")

        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=3, column=0, pady=5)

        ttk.Button(nav_frame, text="上一张", command=self.prev_image).grid(row=0, column=0, padx=5)
        ttk.Button(nav_frame, text="下一张", command=self.next_image).grid(row=0, column=1, padx=5)

        # 样本索引输入
        index_frame = ttk.Frame(control_frame)
        index_frame.grid(row=4, column=0, pady=5)

        ttk.Label(index_frame, text="跳转到:").grid(row=0, column=0)
        self.index_entry = ttk.Entry(index_frame, width=10)
        self.index_entry.grid(row=0, column=1, padx=5)
        ttk.Button(index_frame, text="跳转", command=self.jump_to_index).grid(row=0, column=2)

        # 当前样本信息
        self.info_label = ttk.Label(control_frame, text="", font=("Arial", 10))
        self.info_label.grid(row=5, column=0, pady=10)

        # 显示选项
        ttk.Label(control_frame, text="显示选项：", font=("Arial", 12, "bold")).grid(row=6, column=0, pady=(20, 5), sticky="w")

        ttk.Checkbutton(control_frame, text="显示关键点", variable=self.show_keypoints,
                       command=self.update_display).grid(row=7, column=0, pady=2, sticky="w")
        ttk.Checkbutton(control_frame, text="显示骨架", variable=self.show_skeleton,
                       command=self.update_display).grid(row=8, column=0, pady=2, sticky="w")
        ttk.Checkbutton(control_frame, text="显示边界框", variable=self.show_bbox,
                       command=self.update_display).grid(row=9, column=0, pady=2, sticky="w")
        ttk.Checkbutton(control_frame, text="显示标签", variable=self.show_label,
                       command=self.update_display).grid(row=10, column=0, pady=2, sticky="w")

        # 关键点信息显示
        ttk.Label(control_frame, text="关键点信息：", font=("Arial", 12, "bold")).grid(row=11, column=0, pady=(20, 5), sticky="w")

        # 创建关键点信息文本框
        self.keypoint_info = tk.Text(control_frame, width=35, height=20, font=("Courier", 9))
        self.keypoint_info.grid(row=12, column=0, pady=5)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(control_frame, command=self.keypoint_info.yview)
        scrollbar.grid(row=12, column=1, sticky="ns")
        self.keypoint_info.config(yscrollcommand=scrollbar.set)

        # 右侧图像显示区域
        image_frame = ttk.Frame(self.root, padding="10")
        image_frame.grid(row=0, column=1, sticky="nsew")

        # 创建画布
        self.canvas = tk.Canvas(image_frame, bg="gray", width=1000, height=800)
        self.canvas.pack()

        # 配置网格权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def bind_keyboard_events(self):
        """绑定键盘快捷键"""
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('a', lambda e: self.prev_image())
        self.root.bind('d', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.toggle_skeleton())
        self.root.bind('k', lambda e: self.toggle_keypoints())
        self.root.bind('b', lambda e: self.toggle_bbox())
        self.root.bind('l', lambda e: self.toggle_label())

    def toggle_skeleton(self):
        """切换骨架显示"""
        self.show_skeleton.set(not self.show_skeleton.get())
        self.update_display()

    def toggle_keypoints(self):
        """切换关键点显示"""
        self.show_keypoints.set(not self.show_keypoints.get())
        self.update_display()

    def toggle_bbox(self):
        """切换边界框显示"""
        self.show_bbox.set(not self.show_bbox.get())
        self.update_display()

    def toggle_label(self):
        """切换标签显示"""
        self.show_label.set(not self.show_label.get())
        self.update_display()

    def load_split_data(self, split: str):
        """加载指定数据集划分的数据"""
        # 加载特征文件
        features_path = PROJECT_ROOT / 'features' / 'pose_features' / f'{split}.npy'

        if not features_path.exists():
            messagebox.showerror("错误", f"特征文件不存在: {features_path}")
            return

        try:
            # 加载特征数据
            self.features_data = np.load(features_path, allow_pickle=True).item()
            self.image_names = list(self.features_data.keys())

            # 加载原始图像
            self.original_images = load_image_and_meta(split, root_dir=PROJECT_ROOT)

            # 重置索引
            self.current_index = 0
            self.current_split = split

            # 更新显示
            self.update_display()

            # 更新信息标签
            self.update_info_label()

        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {e}")

    def on_split_change(self):
        """处理数据集切换事件"""
        new_split = self.split_var.get()
        if new_split != self.current_split:
            self.load_split_data(new_split)

    def prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            self.update_info_label()

    def next_image(self):
        """显示下一张图像"""
        if self.current_index < len(self.image_names) - 1:
            self.current_index += 1
            self.update_display()
            self.update_info_label()

    def jump_to_index(self):
        """跳转到指定索引"""
        try:
            index = int(self.index_entry.get())
            if 0 <= index < len(self.image_names):
                self.current_index = index
                self.update_display()
                self.update_info_label()
            else:
                messagebox.showwarning("警告", f"索引必须在0到{len(self.image_names)-1}之间")
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的整数")

    def update_info_label(self):
        """更新样本信息标签"""
        if self.image_names:
            info_text = f"当前样本: {self.current_index + 1}/{len(self.image_names)}\n"
            info_text += f"文件名: {self.image_names[self.current_index]}"
            self.info_label.config(text=info_text)

    def draw_keypoints(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """在图像上绘制关键点"""
        if keypoints is None:
            return image

        for name in self.keypoint_names:
            if name in keypoints:
                x, y = keypoints[name]
                if x is not None and y is not None:
                    # 获取关键点颜色
                    color = self.keypoint_colors.get(name, (255, 255, 255))
                    # 绘制关键点
                    cv2.circle(image, (int(x), int(y)), 5, color, -1)
                    cv2.circle(image, (int(x), int(y)), 7, (0, 0, 0), 2)  # 黑色边框

        return image

    def draw_skeleton(self, image: np.ndarray, keypoints: Dict) -> np.ndarray:
        """在图像上绘制骨架"""
        if keypoints is None:
            return image

        for connection in self.skeleton_connections:
            if len(connection) == 2:
                idx1, idx2 = connection
                if idx1 < len(self.keypoint_names) and idx2 < len(self.keypoint_names):
                    name1 = self.keypoint_names[idx1]
                    name2 = self.keypoint_names[idx2]

                    if name1 in keypoints and name2 in keypoints:
                        pt1 = keypoints[name1]
                        pt2 = keypoints[name2]

                        if pt1[0] is not None and pt2[0] is not None:
                            cv2.line(image,
                                   (int(pt1[0]), int(pt1[1])),
                                   (int(pt2[0]), int(pt2[1])),
                                   self.skeleton_color, 2)

        return image

    def draw_bbox_and_label(self, image: np.ndarray, meta: List[Dict]) -> np.ndarray:
        """在图像上绘制边界框和标签"""
        if not meta:
            return image

        h, w = image.shape[:2]

        for obj in meta:
            # 转换归一化坐标到像素坐标
            cx = obj['x_center'] * w
            cy = obj['y_center'] * h
            bbox_w = obj['width'] * w
            bbox_h = obj['height'] * h

            # 计算边界框角点
            x1 = int(cx - bbox_w / 2)
            y1 = int(cy - bbox_h / 2)
            x2 = int(cx + bbox_w / 2)
            y2 = int(cy + bbox_h / 2)

            # 确定颜色和标签文本
            if obj['cls'] == 0:
                color = (0, 255, 0)  # 绿色：未摔倒
                label = "Normal"
            else:
                color = (0, 0, 255)  # 红色：摔倒
                label = "Fall"

            if self.show_bbox.get():
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            if self.show_label.get():
                # 绘制标签背景
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(image, (x1, y1 - 25), (x1 + label_size[0] + 5, y1), color, -1)
                # 绘制标签文本
                cv2.putText(image, label, (x1 + 2, y1 - 8),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return image

    def update_keypoint_info(self, keypoints: Dict):
        """更新关键点信息显示"""
        self.keypoint_info.delete(1.0, tk.END)

        if keypoints is None:
            self.keypoint_info.insert(tk.END, "未检测到关键点\n")
            return

        self.keypoint_info.insert(tk.END, "关键点坐标:\n")
        self.keypoint_info.insert(tk.END, "-" * 30 + "\n")

        for i, name in enumerate(self.keypoint_names):
            if name in keypoints:
                x, y = keypoints[name]
                if x is not None and y is not None:
                    info = f"{i:2d}. {name:15s}: ({x:6.1f}, {y:6.1f})\n"
                else:
                    info = f"{i:2d}. {name:15s}: (None, None)\n"
                self.keypoint_info.insert(tk.END, info)

    def update_display(self):
        """更新显示"""
        if not self.image_names:
            return

        # 获取当前样本
        image_name = self.image_names[self.current_index]
        feature_data = self.features_data[image_name]

        # 获取原始图像
        if image_name in self.original_images:
            image = self.original_images[image_name]['image'].copy()
        else:
            # 如果找不到原始图像，创建空白图像
            image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        # 获取关键点和标注信息
        keypoints = feature_data.get('keypoints', None)
        meta = feature_data.get('meta', [])

        # 绘制可视化元素
        if self.show_skeleton.get():
            image = self.draw_skeleton(image, keypoints)
        if self.show_keypoints.get():
            image = self.draw_keypoints(image, keypoints)

        image = self.draw_bbox_and_label(image, meta)

        # 转换为RGB格式（OpenCV使用BGR）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整图像大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 1 and canvas_height > 1:
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h))

        # 转换为PIL图像并显示
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                anchor="center", image=photo)
        self.canvas.image = photo  # 保持引用

        # 更新关键点信息
        self.update_keypoint_info(keypoints)


def main():
    """主函数"""
    root = tk.Tk()
    app = PoseKeypointsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()