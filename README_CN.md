# 🚨 人体摔倒检测系统 (Human Fall Detection)

<div align="center">

[English](README.md) | [简体中文](README_CN.md)

</div>

[![PyPI](https://img.shields.io/pypi/v/human-fall-detection?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/human-fall-detection/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

基于YOLOv8的静态图像摔倒检测系统，支持实时视频流和离线视频文件处理。

## 📋 目录

- [🚨 人体摔倒检测系统 (Human Fall Detection)](#-人体摔倒检测系统-human-fall-detection)
  - [📋 目录](#-目录)
  - [🎯 项目简介](#-项目简介)
  - [📊 模型性能](#-模型性能)
    - [分类别性能](#分类别性能)
      - [🟢 正常状态 (Normal)](#-正常状态-normal)
      - [🔴 摔倒状态 (Fall)](#-摔倒状态-fall)
  - [💾 安装方法](#-安装方法)
    - [方法一：pip安装（推荐）](#方法一pip安装推荐)
    - [方法二：从源码安装](#方法二从源码安装)
  - [📖 使用示例](#-使用示例)
    - [Jupyter Notebook 示例](#jupyter-notebook-示例)
  - [📈 评估结果](#-评估结果)
    - [性能指标图表](#性能指标图表)
    - [检测示例](#检测示例)
    - [视频检测效果](#视频检测效果)
  - [🙏 致谢](#-致谢)
  - [📮 联系方式](#-联系方式)

## 🎯 项目简介

本项目是一个基于深度学习的人体摔倒检测系统，使用微调的YOLOv8模型实现高精度的摔倒检测。该系统可以应用于老年人看护、医院监控、智能家居等场景，及时发现摔倒事件并发出警报。

## 📊 模型性能

基于测试集的评估结果：

| 指标             | 数值   | 说明                     |
| ---------------- | ------ | ------------------------ |
| **mAP@0.5**      | 90.61% | 主要性能指标             |
| **mAP@0.5:0.95** | 39.63% | 更严格的评估标准         |
| **F1-Score**     | 88.60% | 精确度和召回率的调和平均 |
| **总体精确度**   | 90.13% | 所有检测的准确率         |
| **总体召回率**   | 87.11% | 检测到的真实目标比例     |

### 分类别性能

#### 🟢 正常状态 (Normal)
- AP@0.5: 86.22%
- 精确度: 90.75%
- 召回率: 80.36%

#### 🔴 摔倒状态 (Fall)
- AP@0.5: 95.00%
- 精确度: 89.51%
- 召回率: 93.86%

#### ✅ 性能评估
- 优秀的摔倒检测敏感度 (93.86% 召回率)
- 较低的误报率 (<10% 误检)
- 整体检测性能优异 (mAP > 90%)


## 💾 安装方法

### 方法一：pip安装（推荐）

```bash
pip install human-fall-detection
```

### 方法二：从源码安装

```bash
git clone https://github.com/TomokotoKiyoshi/Human-Fall-Detection.git
cd Human-Fall-Detection
uv sync
```

## 📖 使用示例

### Jupyter Notebook 示例

参见 `example_usage.ipynb` 获取交互式示例。

## 📈 评估结果

### 性能指标图表

|                                    混淆矩阵                                     |                              PR曲线                              |                            F1曲线                             |
| :-----------------------------------------------------------------------------: | :--------------------------------------------------------------: | :-----------------------------------------------------------: |
| ![混淆矩阵](results/evaluation/test_evaluation/confusion_matrix_normalized.png) |  ![PR曲线](results/evaluation/test_evaluation/BoxPR_curve.png)   | ![F1曲线](results/evaluation/test_evaluation/BoxF1_curve.png) |
|                                 **精确度曲线**                                  |                          **召回率曲线**                          |                                                               |
|        ![精确度曲线](results/evaluation/test_evaluation/BoxP_curve.png)         | ![召回率曲线](results/evaluation/test_evaluation/BoxR_curve.png) |                                                               |

### 检测示例

以下是模型在测试集上的实际检测效果：

| 原始标注                                                          | 模型预测                                                        |
| ----------------------------------------------------------------- | --------------------------------------------------------------- |
| ![标注](results/evaluation/test_evaluation/val_batch0_labels.jpg) | ![预测](results/evaluation/test_evaluation/val_batch0_pred.jpg) |

### 视频检测效果

以下是系统处理视频的实际效果展示：

<div align="center">

![Demo](results/fall_detection/output_demo.gif)

</div>

视频展示了系统的实时检测能力：
- 🟩 绿色框：正常状态
- 🟥 红色框：检测到摔倒
- 实时显示置信度分数


## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 提供优秀的目标检测框架
- [UAH Fall Detection Dataset](https://gram.web.uah.es/data/datasets/fpds/index.html) - 提供高质量的摔倒检测数据集

## 📮 联系方式

- 项目主页: [https://github.com/TomokotoKiyoshi/Human-Fall-Detection](https://github.com/TomokotoKiyoshi/Human-Fall-Detection)
- 问题反馈: [Issues](https://github.com/TomokotoKiyoshi/Human-Fall-Detection/issues)

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个星标支持！**

</div>