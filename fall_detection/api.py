#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple functional API for fall detection.

Provides easy-to-use functions for detecting falls in images and videos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
from .detector import FallDetector


def process_video(input_path: str,
                 output_path: Optional[str] = None,
                 model_type: str = "onnx",
                 confidence: float = 0.5,
                 device: str = "cuda",
                 show_progress: bool = True) -> Dict[str, Any]:
    """Process a video file and detect falls.

    Args:
        input_path (str): Path to input video file.
        output_path (Optional[str], optional): Path to save annotated video. If None, no output is saved. Defaults to None.
        model_type (str, optional): Model type - 'pt' for PyTorch or 'onnx' for ONNX FP16. Defaults to "onnx".
        confidence (float, optional): Detection confidence threshold. Defaults to 0.5.
        device (str, optional): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".
        show_progress (bool, optional): Whether to print progress updates. Defaults to True.

    Returns:
        Dict[str, Any]: Detection results containing:
            - video_saved (Optional[str]): Path to saved video or None.
            - total_frames (int): Total number of frames processed.
            - fall_frames (int): Number of frames containing falls.
            - statistics (Dict[str, int]):
                - total_detections (int): Total detections across all frames.
                - fall_count (int): Total fall detections.
                - normal_count (int): Total normal detections.

    Raises:
        FileNotFoundError: If input video file does not exist.

    Example:
        >>> result = process_video("test.mp4", "output.mp4")
        >>> print(f"Detected {result['statistics']['fall_count']} falls")
    """
    # Check input file
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")

    # Create detector
    detector = FallDetector(model_type=model_type, confidence=confidence, device=device)

    # Process video
    result = detector.process_video_file(
        input_path=input_path,
        output_path=output_path,
        show_progress=show_progress
    )

    if show_progress:
        print(f"\nProcessing complete!")
        print(f"Total frames: {result['total_frames']}")
        print(f"Frames with falls: {result['fall_frames']}")
        print(f"Total fall detections: {result['statistics']['fall_count']}")
        print(f"Total normal detections: {result['statistics']['normal_count']}")
        if output_path:
            print(f"Output video: {output_path}")

    return result


def process_image(input_path: str,
                 output_path: Optional[str] = None,
                 model_type: str = "onnx",
                 confidence: float = 0.5,
                 device: str = "cuda") -> Dict[str, Any]:
    """Process a single image and detect falls.

    Args:
        input_path (str): Path to input image file.
        output_path (Optional[str], optional): Path to save annotated image. Defaults to None.
        model_type (str, optional): Model type - 'pt' for PyTorch or 'onnx' for ONNX FP16. Defaults to "onnx".
        confidence (float, optional): Detection confidence threshold. Defaults to 0.5.
        device (str, optional): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        Dict[str, Any]: Detection results containing:
            - fall_detected (bool): True if at least one fall is detected.
            - detections (List[Dict]): List of detection dictionaries.
            - fall_count (int): Number of falls detected.
            - normal_count (int): Number of normal poses detected.
            - annotated_image (np.ndarray): Image with drawn annotations.
            - image_saved (Optional[str]): Path to saved image or None.

    Raises:
        FileNotFoundError: If input image file does not exist.
        ValueError: If image cannot be read.

    Example:
        >>> result = process_image("test.jpg")
        >>> if result["fall_detected"]:
        ...     print("Warning: Fall detected!")
        ...     for det in result["detections"]:
        ...         if det["class"] == "Fall":
        ...             print(f"Confidence: {det['confidence']:.2%}")
    """
    # Check input file
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Image file not found: {input_path}")

    # Read image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # Create detector
    detector = FallDetector(model_type=model_type, confidence=confidence, device=device)

    # Detect in image
    result = detector.detect_image(image)

    # Save result if needed
    if output_path:
        cv2.imwrite(output_path, result["annotated_image"])
        result["image_saved"] = output_path
    else:
        result["image_saved"] = None

    return result


def detect_from_array(image: np.ndarray,
                     model_type: str = "onnx",
                     confidence: float = 0.5,
                     device: str = "cuda") -> Dict[str, Any]:
    """Detect falls from a numpy array image.

    Useful when the image is already loaded in memory.

    Args:
        image (np.ndarray): Input image in BGR format with shape (H, W, 3).
        model_type (str, optional): Model type - 'pt' for PyTorch or 'onnx' for ONNX FP16. Defaults to "onnx".
        confidence (float, optional): Detection confidence threshold. Defaults to 0.5.
        device (str, optional): Device to run inference on ("cuda" or "cpu"). Defaults to "cuda".

    Returns:
        Dict[str, Any]: Same as process_image() output:
            - fall_detected (bool): True if at least one fall is detected.
            - detections (List[Dict]): List of detection dictionaries.
            - fall_count (int): Number of falls detected.
            - normal_count (int): Number of normal poses detected.
            - annotated_image (np.ndarray): Image with drawn annotations.

    Example:
        >>> import cv2
        >>> img = cv2.imread("test.jpg")
        >>> result = detect_from_array(img)
        >>> cv2.imshow("Result", result["annotated_image"])
    """
    # Create detector
    detector = FallDetector(model_type=model_type, confidence=confidence, device=device)

    # Detect in image
    return detector.detect_image(image)

