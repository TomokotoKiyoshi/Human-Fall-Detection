"""
标签框可视化GUI - YOLO格式标注可视化工具

功能描述：
    这个GUI工具用于可视化跌倒检测数据集的YOLO格式标注框，帮助验证标注质量和数据分布。

主要功能：
    1. 数据集切换：支持在train/val/test三个数据集之间快速切换
    2. 图像导航：
       - 使用Previous/Next按钮或键盘快捷键（A/←、D/→）浏览图像
       - 支持直接跳转到指定索引的图像
    3. 标注框可视化：
       - 绿色框：Normal（未摔倒，类别=0）
       - 红色框：Fall（摔倒，类别=1）
       - 显示YOLO格式的归一化坐标并转换为像素坐标绘制
    4. 显示选项：
       - Show Labels：显示/隐藏类别标签文字
       - Show Coordinates：显示/隐藏边界框坐标
       - Show All Boxes：显示所有框或仅显示第一个框
    5. 统计信息：
       - 实时显示当前图像的标注框总数
       - 分别统计Normal和Fall类别的数量
    6. 文件信息：显示当前图像文件名和索引位置

使用方法：
    1. 运行脚本：python labels_gui.py
    2. 选择数据集（train/val/test）
    3. 使用导航按钮或键盘浏览图像
    4. 切换显示选项查看不同信息

数据路径结构：
    - 图像路径：data/images/{split}/*.png
    - 标签路径：data/labels/{split}/*.txt

标签格式：
    YOLO格式：class_id x_center y_center width height（所有值归一化到0-1）
    - class_id: 0=未摔倒(Normal), 1=摔倒(Fall)

键盘快捷键：
    - A/←：上一张图像
    - D/→：下一张图像

依赖库：
    - OpenCV：图像处理和绘制
    - Tkinter：GUI界面
    - Pillow：图像格式转换
    - NumPy：数组处理
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))


class LabelsVisualizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Box Visualizer")
        self.root.geometry("1200x800")

        # 数据相关变量
        self.current_split = 'train'
        self.image_files = []
        self.current_index = 0
        self.current_image = None

        # 颜色配置 (YOLO格式: 0=nonfallen, 1=fallen)
        self.colors = {
            0: (0, 255, 0),     # Green: Normal (not fallen)
            1: (0, 0, 255),     # Red: Fallen
        }

        # 创建GUI组件
        self.setup_gui()

        # 绑定键盘事件
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<a>', lambda e: self.prev_image())
        self.root.bind('<d>', lambda e: self.next_image())

        # 加载初始数据
        self.load_split_data()

    def setup_gui(self):
        """设置GUI组件"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Dataset selection
        ttk.Label(control_frame, text="Dataset:").pack(side=tk.LEFT, padx=5)
        self.split_var = tk.StringVar(value='train')
        split_combo = ttk.Combobox(control_frame, textvariable=self.split_var,
                                   values=['train', 'val', 'test'],
                                   width=10, state='readonly')
        split_combo.pack(side=tk.LEFT, padx=5)
        split_combo.bind('<<ComboboxSelected>>', self.on_split_change)

        # Navigation buttons
        ttk.Button(control_frame, text="← Previous (A/←)",
                  command=self.prev_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next (D/→) →",
                  command=self.next_image).pack(side=tk.LEFT, padx=5)

        # Image index display
        self.index_label = ttk.Label(control_frame, text="0 / 0")
        self.index_label.pack(side=tk.LEFT, padx=20)

        # Jump to image
        ttk.Label(control_frame, text="Jump to:").pack(side=tk.LEFT, padx=5)
        self.jump_entry = ttk.Entry(control_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Go",
                  command=self.jump_to_image).pack(side=tk.LEFT, padx=5)

        # 显示选项
        options_frame = ttk.Frame(self.root)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.show_labels = tk.BooleanVar(value=True)
        self.show_coords = tk.BooleanVar(value=True)
        self.show_all = tk.BooleanVar(value=True)

        ttk.Checkbutton(options_frame, text="Show Labels",
                       variable=self.show_labels,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Show Coordinates",
                       variable=self.show_coords,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(options_frame, text="Show All Boxes",
                       variable=self.show_all,
                       command=self.update_display).pack(side=tk.LEFT, padx=5)

        # Color legend
        ttk.Label(options_frame, text="  |  Color Legend:").pack(side=tk.LEFT, padx=5)
        ttk.Label(options_frame, text="Green=Normal",
                 foreground='green').pack(side=tk.LEFT, padx=2)
        ttk.Label(options_frame, text="Red=Fall",
                 foreground='red').pack(side=tk.LEFT, padx=2)

        # 信息显示框
        info_frame = ttk.Frame(self.root)
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.info_label.pack(side=tk.LEFT, padx=5)

        # 统计信息
        self.stats_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.stats_label.pack(side=tk.LEFT, padx=20)

        # 图像显示区域
        self.canvas = tk.Canvas(self.root, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Status bar
        self.status_label = ttk.Label(self.root, text="Ready",
                                     relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

    def load_split_data(self):
        """加载当前split的数据"""
        data_dir = Path('data') / 'images' / self.current_split
        if not data_dir.exists():
            messagebox.showerror("Error", f"Directory not found: {data_dir}")
            return

        self.image_files = sorted(list(data_dir.glob('*.png')))
        self.current_index = 0

        if self.image_files:
            self.display_image()
        else:
            messagebox.showwarning("Warning", f"No images found in {self.current_split} set")

        self.update_index_label()

    def on_split_change(self, event=None):
        """切换数据集"""
        self.current_split = self.split_var.get()
        self.load_split_data()

    def prev_image(self):
        """显示上一张图片"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_image()
            self.update_index_label()

    def next_image(self):
        """显示下一张图片"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()
            self.update_index_label()

    def jump_to_image(self):
        """跳转到指定索引的图片"""
        try:
            index = int(self.jump_entry.get()) - 1  # 用户输入从1开始
            if 0 <= index < len(self.image_files):
                self.current_index = index
                self.display_image()
                self.update_index_label()
            else:
                messagebox.showwarning("Warning", f"Please enter a number between 1-{len(self.image_files)}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid number")

    def update_index_label(self):
        """更新索引显示"""
        if self.image_files:
            self.index_label.config(
                text=f"{self.current_index + 1} / {len(self.image_files)}"
            )
        else:
            self.index_label.config(text="0 / 0")

    def display_image(self):
        """显示当前图片"""
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]

        # Load original image
        self.current_image = cv2.imread(str(image_path))
        if self.current_image is None:
            self.status_label.config(text=f"Failed to load image: {image_path.name}")
            return

        # Update info label
        self.info_label.config(text=f"File: {image_path.name}")

        # 处理并显示图像
        self.process_and_display()

    def update_display(self):
        """更新显示（当复选框改变时）"""
        if self.current_image is not None:
            self.process_and_display()

    def process_and_display(self):
        """处理并显示图像"""
        if self.current_image is None:
            return

        img = self.current_image.copy()
        image_path = self.image_files[self.current_index]

        # 读取标签文件 (使用新的YOLO格式路径)
        label_path = Path('data') / 'labels' / self.current_split / image_path.name.replace('.png', '.txt')

        normal_count = 0
        fall_count = 0

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()

            # 绘制每个框
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 5:
                    # YOLO格式：class_id x_center y_center width height (归一化)
                    cls = int(parts[0])  # 0: nonfallen, 1: fallen
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 获取图像尺寸
                    img_height, img_width = img.shape[:2]

                    # 转换为像素坐标
                    x_center_px = int(x_center * img_width)
                    y_center_px = int(y_center * img_height)
                    width_px = int(width * img_width)
                    height_px = int(height * img_height)

                    # 计算左上角和右下角坐标
                    x1 = x_center_px - width_px // 2
                    y1 = y_center_px - height_px // 2
                    x2 = x_center_px + width_px // 2
                    y2 = y_center_px + height_px // 2

                    # 统计 (YOLO格式: 0=nonfallen, 1=fallen)
                    if cls == 0:
                        normal_count += 1
                    else:
                        fall_count += 1

                    # 如果不显示所有框，只显示第一个
                    if not self.show_all.get() and i > 0:
                        break

                    # 选择颜色
                    color = self.colors.get(cls, (128, 128, 128))

                    # 绘制框
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    # Show label (YOLO格式: 0=nonfallen, 1=fallen)
                    if self.show_labels.get():
                        label_text = "Fall" if cls == 1 else "Normal"

                        # 计算文本背景
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label_text, font, font_scale, thickness
                        )

                        # 绘制文本背景
                        cv2.rectangle(img,
                                    (x1, y1 - text_height - 10),
                                    (x1 + text_width + 4, y1),
                                    color, -1)

                        # 绘制文本
                        cv2.putText(img, label_text,
                                  (x1 + 2, y1 - 5),
                                  font, font_scale,
                                  (255, 255, 255), thickness)

                    # 显示坐标
                    if self.show_coords.get():
                        coord_text = f"({x1},{y1})-({x2},{y2})"
                        cv2.putText(img, coord_text,
                                  (x1, y2 + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                  color, 1)

        else:
            # No label file
            self.stats_label.config(text="No label file")
            cv2.putText(img, "No Label File",
                       (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 255), 2)

        # Update statistics
        total = normal_count + fall_count
        if total > 0:
            self.stats_label.config(
                text=f"Total Boxes: {total} | Normal: {normal_count} | Fall: {fall_count}"
            )

        # 显示图像
        self.show_image_on_canvas(img)

    def show_image_on_canvas(self, cv_image):
        """在Canvas上显示图像"""
        # 转换颜色空间
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        # 调整大小以适应Canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            h, w = rgb_image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h, 1.0)

            new_w = int(w * scale)
            new_h = int(h * scale)

            rgb_image = cv2.resize(rgb_image, (new_w, new_h))

        # 转换为PhotoImage
        pil_image = Image.fromarray(rgb_image)
        self.photo = ImageTk.PhotoImage(pil_image)

        # 在Canvas上显示
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=self.photo,
            anchor=tk.CENTER
        )

        self.status_label.config(text="Ready")


def main():
    root = tk.Tk()
    app = LabelsVisualizerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()