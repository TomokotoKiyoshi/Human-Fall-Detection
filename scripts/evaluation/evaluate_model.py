#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model evaluation script for fall detection.

Evaluates the trained YOLOv8n model on test dataset and returns key metrics.
"""

import sys
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import json
from typing import Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration paths
TEST_IMAGES_PATH = PROJECT_ROOT / 'data' / 'images' / 'test'
TEST_LABELS_PATH = PROJECT_ROOT / 'data' / 'labels' / 'test'
MODEL_PATH = PROJECT_ROOT / 'results' / 'models' / 'fall_detection' / 'weights' / 'best.pt'
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'fall_detection.yaml'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'evaluation'


def evaluate_model() -> Dict[str, Any]:
    """Evaluate the trained model on test dataset.

    Returns:
        Dict containing evaluation metrics.
    """
    print("=" * 60)
    print("Fall Detection Model Evaluation")
    print("=" * 60)

    # Check if model exists
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using scripts/training/train_yolo_fall.py")
        return {}

    # Check if test data exists
    if not TEST_IMAGES_PATH.exists():
        print(f"Error: Test images not found at {TEST_IMAGES_PATH}")
        return {}

    # Count test samples
    test_images = list(TEST_IMAGES_PATH.glob("*.jpg")) + list(TEST_IMAGES_PATH.glob("*.png"))
    test_labels = list(TEST_LABELS_PATH.glob("*.txt"))

    print(f"\nDataset Info:")
    print(f"  Test images: {len(test_images)}")
    print(f"  Test labels: {len(test_labels)}")
    print(f"  Model path: {MODEL_PATH}")

    # Load model
    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run validation on test set
    print("\nRunning evaluation on test set...")
    print("-" * 40)

    # Validate using YOLO's built-in validation
    results = model.val(
        data=str(CONFIG_PATH),
        split='test',
        imgsz=640,
        batch=16,
        conf=0.25,  # Confidence threshold
        iou=0.5,    # IoU threshold for NMS
        device='cuda',
        save_json=True,
        save_txt=True,
        save_conf=True,
        project=str(OUTPUT_DIR),
        name='test_evaluation',
        exist_ok=True,
        plots=True,
        verbose=True
    )

    # Extract key metrics
    metrics = {}

    # Overall metrics
    metrics['mAP50'] = float(results.box.map50) if results.box.map50 else 0.0
    metrics['mAP50-95'] = float(results.box.map) if results.box.map else 0.0

    # Per-class metrics (Normal=0, Fall=1)
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        ap50_values = results.box.ap50.cpu().numpy() if hasattr(results.box.ap50, 'cpu') else results.box.ap50
        if len(ap50_values) > 0:
            metrics['Normal_AP50'] = float(ap50_values[0])
        if len(ap50_values) > 1:
            metrics['Fall_AP50'] = float(ap50_values[1])

    # Precision and Recall
    if hasattr(results.box, 'all_ap') and results.box.all_ap is not None:
        # Try to get per-class precision and recall
        if hasattr(results.box, 'p') and results.box.p is not None:
            p_values = results.box.p.cpu().numpy() if hasattr(results.box.p, 'cpu') else results.box.p
            if len(p_values) > 0:
                metrics['Normal_precision'] = float(p_values[0])
            if len(p_values) > 1:
                metrics['Fall_precision'] = float(p_values[1])
            metrics['precision'] = float(np.mean(p_values))

        if hasattr(results.box, 'r') and results.box.r is not None:
            r_values = results.box.r.cpu().numpy() if hasattr(results.box.r, 'cpu') else results.box.r
            if len(r_values) > 0:
                metrics['Normal_recall'] = float(r_values[0])
            if len(r_values) > 1:
                metrics['Fall_recall'] = float(r_values[1])
            metrics['recall'] = float(np.mean(r_values))
    else:
        # Fallback to overall metrics
        metrics['precision'] = float(results.box.mp) if hasattr(results.box, 'mp') and results.box.mp else 0.0
        metrics['recall'] = float(results.box.mr) if hasattr(results.box, 'mr') and results.box.mr else 0.0

    # Calculate F1 score
    if metrics.get('precision', 0) > 0 and metrics.get('recall', 0) > 0:
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1_score'] = 0.0

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print("\n📊 Overall Metrics:")
    print(f"  mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"  Precision:    {metrics.get('precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('recall', 0):.4f}")
    print(f"  F1-Score:     {metrics.get('f1_score', 0):.4f}")

    print("\n📈 Class-specific Metrics:")
    print("\n  Normal Class (No Fall):")
    if 'Normal_AP50' in metrics:
        print(f"    AP@0.5:     {metrics['Normal_AP50']:.4f}")
    if 'Normal_precision' in metrics:
        print(f"    Precision:  {metrics['Normal_precision']:.4f}")
    if 'Normal_recall' in metrics:
        print(f"    Recall:     {metrics['Normal_recall']:.4f}")

    print("\n  Fall Class:")
    if 'Fall_AP50' in metrics:
        print(f"    AP@0.5:     {metrics['Fall_AP50']:.4f}")
    if 'Fall_precision' in metrics:
        print(f"    Precision:  {metrics['Fall_precision']:.4f}")
    if 'Fall_recall' in metrics:
        print(f"    Recall:     {metrics['Fall_recall']:.4f}")

    # Save metrics to JSON
    metrics_file = OUTPUT_DIR / 'test_evaluation' / 'metrics.json'
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "=" * 60)
    print("📁 Results saved to:")
    print(f"  - Metrics: {metrics_file}")
    print(f"  - Plots: {OUTPUT_DIR / 'test_evaluation'}")
    print("=" * 60)

    # Performance interpretation
    print("\n🎯 Performance Interpretation:")

    if metrics['mAP50'] >= 0.9:
        print("  ✅ Excellent detection performance (mAP > 90%)")
    elif metrics['mAP50'] >= 0.8:
        print("  ✅ Good detection performance (mAP > 80%)")
    elif metrics['mAP50'] >= 0.7:
        print("  ⚠️  Moderate detection performance (mAP > 70%)")
    else:
        print("  ❌ Poor detection performance (mAP < 70%)")

    if 'Fall_recall' in metrics:
        if metrics['Fall_recall'] >= 0.9:
            print("  ✅ Excellent fall detection sensitivity (>90% falls detected)")
        elif metrics['Fall_recall'] >= 0.8:
            print("  ✅ Good fall detection sensitivity (>80% falls detected)")
        else:
            print("  ⚠️  Low fall detection sensitivity (<80% falls detected)")

    if 'Fall_precision' in metrics:
        if metrics['Fall_precision'] >= 0.9:
            print("  ✅ Very few false alarms (<10% false positives)")
        elif metrics['Fall_precision'] >= 0.8:
            print("  ✅ Acceptable false alarm rate (<20% false positives)")
        else:
            print("  ⚠️  High false alarm rate (>20% false positives)")

    return metrics


def run_inference_examples(num_samples=5):
    """Run inference on a few test images as examples.

    Args:
        num_samples: Number of sample images to test
    """
    print("\n" + "=" * 60)
    print("Running Inference Examples")
    print("=" * 60)

    # Load model
    model = YOLO(str(MODEL_PATH))

    # Get sample test images
    test_images = list(TEST_IMAGES_PATH.glob("*.jpg"))[:num_samples]

    if not test_images:
        test_images = list(TEST_IMAGES_PATH.glob("*.png"))[:num_samples]

    if not test_images:
        print("No test images found for inference examples")
        return

    print(f"\nRunning inference on {len(test_images)} sample images...")

    correct_predictions = 0
    total_predictions = 0

    for img_path in test_images:
        # Run inference
        results = model(str(img_path), conf=0.25, device='cuda', verbose=False)

        # Get corresponding label file
        label_path = TEST_LABELS_PATH / f"{img_path.stem}.txt"

        print(f"\n📷 Image: {img_path.name}")

        # Read ground truth
        gt_classes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                gt_classes = [int(line.split()[0]) for line in lines if line.strip()]
                gt_falls = sum(1 for c in gt_classes if c == 1)
                gt_normal = sum(1 for c in gt_classes if c == 0)
                print(f"  Ground Truth: {gt_falls} falls, {gt_normal} normal")
        else:
            print(f"  Ground Truth: No label file found")

        # Get predictions
        for r in results:
            if r.boxes is not None:
                classes = r.boxes.cls.cpu().numpy()
                pred_falls = sum(1 for c in classes if c == 1)
                pred_normal = sum(1 for c in classes if c == 0)
                print(f"  Predictions:  {pred_falls} falls, {pred_normal} normal")

                # Compare predictions with ground truth
                if label_path.exists():
                    pred_classes = sorted([int(c) for c in classes])
                    gt_classes_sorted = sorted(gt_classes)

                    if pred_classes == gt_classes_sorted:
                        print("  ✅ Correct prediction")
                        correct_predictions += 1
                    else:
                        print("  ❌ Incorrect prediction")
                    total_predictions += 1

                # Show confidence scores
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = "Fall" if cls == 1 else "Normal"
                    print(f"    - {class_name}: {conf:.2%} confidence")
            else:
                print(f"  Predictions:  No detections")
                if label_path.exists() and len(gt_classes) == 0:
                    print("  ✅ Correct (no objects)")
                    correct_predictions += 1
                    total_predictions += 1
                elif label_path.exists():
                    print("  ❌ Missed detection")
                    total_predictions += 1

    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions * 100
        print(f"\n📊 Sample Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")


def main():
    """Main evaluation function."""
    # Run evaluation
    metrics = evaluate_model()

    if metrics:
        # Run inference examples
        run_inference_examples(num_samples=10)

        print("\n" + "=" * 60)
        print("✅ Evaluation Complete!")
        print("=" * 60)

        # Summary
        print("\n📋 Summary:")
        print(f"  - Model mAP@0.5: {metrics.get('mAP50', 0):.2%}")
        if 'Fall_recall' in metrics:
            print(f"  - Fall Detection Rate: {metrics['Fall_recall']:.2%}")
        if 'Fall_precision' in metrics:
            print(f"  - Fall Precision: {metrics['Fall_precision']:.2%}")
        print(f"\n  Check {OUTPUT_DIR / 'test_evaluation'} for detailed results and plots.")
    else:
        print("\n❌ Evaluation failed. Please check the error messages above.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())