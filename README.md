# 🚨 Human Fall Detection System

<div align="center">

[English](README.md) | [简体中文](README_CN.md)

</div>

[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

YOLOv8-based static image fall detection system with support for real-time video stream and offline video file processing.

## 📋 Table of Contents

- [🚨 Human Fall Detection System](#-human-fall-detection-system)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Introduction](#-introduction)
    - [Core Advantages](#core-advantages)
  - [📊 Model Performance](#-model-performance)
    - [Class-wise Performance](#class-wise-performance)
      - [🟢 Normal State](#-normal-state)
      - [🔴 Fall State](#-fall-state)
  - [💾 Installation](#-installation)
    - [Method 1: pip Installation (Recommended)](#method-1-pip-installation-recommended)
    - [Method 2: Install from Source](#method-2-install-from-source)
  - [📖 Usage Examples](#-usage-examples)
    - [Jupyter Notebook Example](#jupyter-notebook-example)
  - [📈 Evaluation Results](#-evaluation-results)
    - [Performance Metrics Charts](#performance-metrics-charts)
    - [Detection Examples](#detection-examples)
    - [Video Detection Demo](#video-detection-demo)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📮 Contact](#-contact)

## 🎯 Introduction

This project is a deep learning-based human fall detection system that achieves high-precision fall detection using a fine-tuned YOLOv8 model. The system can be applied to elderly care, hospital monitoring, smart home scenarios, and other contexts to promptly detect fall events and trigger alerts.

### Core Advantages

- 🔥 **High Accuracy**: mAP@0.5 reaches 94.85%, F1-Score reaches 92.02%
- ⚡ **Real-time Detection**: Supports real-time video stream processing
- 📦 **Easy to Deploy**: Provides Python package with pip installation support
- 🎨 **Visualization**: Automatically annotates detection results and generates marked videos

## 📊 Model Performance

Evaluation results based on test dataset (973 images):

| Metric                | Value  | Description                       |
| --------------------- | ------ | --------------------------------- |
| **mAP@0.5**           | 94.85% | Primary performance indicator     |
| **mAP@0.5:0.95**      | 45.50% | Stricter evaluation standard      |
| **F1-Score**          | 92.02% | Harmonic mean of precision/recall |
| **Overall Precision** | 92.46% | Accuracy of all detections        |
| **Overall Recall**    | 91.58% | Proportion of detected targets    |

### Class-wise Performance

#### 🟢 Normal State
- AP@0.5: 94.46%
- Precision: 91.70%
- Recall: 91.89%

#### 🔴 Fall State
- AP@0.5: 95.23%
- Precision: 93.21%
- Recall: 91.26%


## 💾 Installation

### Method 1: pip Installation (Recommended)

```bash
pip install human-fall-detection
```

### Method 2: Install from Source

```bash
git clone https://github.com/TomokotoKiyoshi/Human-Fall-Detection.git
cd Human-Fall-Detection
uv sync
```

## 📖 Usage Examples

### Jupyter Notebook Example

See `example_usage.ipynb` for interactive examples.

## 📈 Evaluation Results

### Performance Metrics Charts

|                                    Confusion Matrix                                     |                              PR Curve                              |                            F1 Curve                             |
| :-------------------------------------------------------------------------------------: | :----------------------------------------------------------------: | :-------------------------------------------------------------: |
| ![Confusion Matrix](results/evaluation/test_evaluation/confusion_matrix_normalized.png) |  ![PR Curve](results/evaluation/test_evaluation/BoxPR_curve.png)   | ![F1 Curve](results/evaluation/test_evaluation/BoxF1_curve.png) |
|                                   **Precision Curve**                                   |                          **Recall Curve**                          |                                                                 |
|          ![Precision Curve](results/evaluation/test_evaluation/BoxP_curve.png)          | ![Recall Curve](results/evaluation/test_evaluation/BoxR_curve.png) |                                                                 |

### Detection Examples

Actual detection performance on the test dataset:

| Ground Truth Labels                                                 | Model Predictions                                                      |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| ![Labels](results/evaluation/test_evaluation/val_batch0_labels.jpg) | ![Predictions](results/evaluation/test_evaluation/val_batch0_pred.jpg) |

### Video Detection Demo

System's real-time video processing demonstration:

<div align="center">

![Demo](results/fall_detection/output_demo.gif)

</div>

The video demonstrates the system's real-time detection capabilities:
- 🟩 Green box: Normal state
- 🟥 Red box: Fall detected
- Real-time confidence scores displayed


## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Excellent object detection framework
- [UAH Fall Detection Dataset](https://gram.web.uah.es/data/datasets/fpds/index.html) - High-quality fall detection dataset

## 📮 Contact

- Project Homepage: [https://github.com/TomokotoKiyoshi/Human-Fall-Detection](https://github.com/TomokotoKiyoshi/Human-Fall-Detection)
- Issues: [Issues](https://github.com/TomokotoKiyoshi/Human-Fall-Detection/issues)

---

<div align="center">

**⭐ If this project helps you, please give it a star!**

</div>