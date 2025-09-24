#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core detector module for fall detection.

Provides FallDetector class using YOLOv8 model.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator, Any
from ultralytics import YOLO

__all__ = ['FallDetector']


class FallDetector:
    """
    Fall detection class using YOLOv8 model.

    Detects human falls in images and videos using a fine-tuned YOLOv8 model.
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda",
                 confidence: float = 0.5):
        """Initialize the fall detector.

        Args:
            model_path (Optional[str], optional): Path to model file. If None, auto-search for model. Defaults to None.
            device (str, optional): Device to run inference on. Defaults to "cuda".
            confidence (float, optional): Detection confidence threshold. Defaults to 0.5.

        Raises:
            FileNotFoundError: If model file cannot be found when auto-searching.
        """
        self.device = device
        self.confidence = confidence
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None) -> YOLO:
        """Load YOLO model from file.

        Args:
            model_path (Optional[str], optional): Path to the model file. If None, searches in default locations. Defaults to None.

        Returns:
            YOLO: Loaded YOLO model instance ready for inference.

        Raises:
            FileNotFoundError: If model file cannot be found in specified path or default locations.
        """
        if model_path is None:
            # Auto-search for model
            # First try package directory (for pip installed version)
            package_model = Path(__file__).resolve().parent / 'models' / 'best.pt'
            if package_model.exists():
                model_path = str(package_model)
            else:
                # Fallback to project structure (for development)
                project_root = Path(__file__).resolve().parent.parent
                possible_paths = [
                    project_root / 'results' / 'models' / 'fall_detection' / 'weights' / 'best.pt',
                    project_root / 'models' / 'yolo_fall' / 'best.pt',
                ]

                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break

            if model_path is None:
                raise FileNotFoundError(
                    "Model file not found. Please run training script first or specify model_path."
                )

        return YOLO(model_path)

    def _draw_annotation(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                        class_name: str, confidence: float, color: tuple, thickness: int) -> None:
        """Draw bounding box and label on image (internal use).

        Args:
            image (np.ndarray): Image to draw on (modified in-place).
            x1, y1, x2, y2 (int): Bounding box coordinates.
            class_name (str): Class name to display.
            confidence (float): Confidence score to display.
            color (tuple): BGR color for the box.
            thickness (int): Line thickness for the box.
        """
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        # Draw label background
        cv2.rectangle(image,
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)

        # Draw label text
        cv2.putText(image, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def detect_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect falls in a single image.

        Args:
            image (np.ndarray): Input image in BGR format with shape (height, width, 3).

        Returns:
            Dict[str, Any]: Detection results containing:
                - detections (List[Dict]): List of detection dictionaries with keys:
                    - "class" (str): "Fall" or "Normal"
                    - "confidence" (float): Detection confidence score
                    - "bbox" (List[int]): Bounding box [x1, y1, x2, y2]
                - fall_detected (bool): True if at least one fall is detected.
                - fall_count (int): Total number of falls detected.
                - normal_count (int): Total number of normal poses detected.
                - annotated_image (np.ndarray): Image with drawn bounding boxes and labels.
                - scene_status (str): Overall scene status based on highest confidence detection.
                - primary_detection (Dict): The detection with highest confidence.

        Example:
            >>> detector = FallDetector()
            >>> image = cv2.imread("test.jpg")
            >>> result = detector.detect_image(image)
            >>> if result["fall_detected"]:
            ...     print(f"Scene status: {result['scene_status']}")
        """
        # Run detection
        results = self.model(image,
                            verbose=False,
                            device=self.device,
                            conf=self.confidence)

        # Parse results
        detections = []
        fall_count = 0
        normal_count = 0
        annotated_image = image.copy()

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Extract coordinates and class
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Determine class
                    if cls == 0:  # Normal
                        class_name = "Normal"
                        color = (0, 255, 0)  # Green
                        thickness = 2
                        normal_count += 1
                    else:  # Fall
                        class_name = "Fall"
                        color = (0, 0, 255)  # Red
                        thickness = 3
                        fall_count += 1

                    # Save detection result
                    detections.append({
                        "class": class_name,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Draw annotation - only draw highest confidence detection
                    # We'll draw after sorting

        # Sort detections by confidence (highest first)
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        # Get primary detection (highest confidence)
        primary_detection = detections[0] if detections else None
        scene_status = "Normal"  # Default

        # Draw only the highest confidence detection
        if primary_detection:
            x1, y1, x2, y2 = primary_detection["bbox"]
            class_name = primary_detection["class"]
            conf = primary_detection["confidence"]

            if class_name == "Fall":
                color = (0, 0, 255)  # Red
                thickness = 3
                scene_status = "Fall"
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
                scene_status = "Normal"

            # Draw annotation for primary detection only
            self._draw_annotation(annotated_image, x1, y1, x2, y2,
                                 class_name, conf, color, thickness)

        return {
            "detections": detections,
            "primary_detection": primary_detection,
            "scene_status": scene_status,
            "fall_detected": scene_status == "Fall",
            "fall_count": fall_count,
            "normal_count": normal_count,
            "annotated_image": annotated_image
        }

    def detect_video(self, video_path: str) -> Generator[Tuple[np.ndarray, Dict], None, None]:
        """Detect falls in video frame by frame using generator pattern.

        Args:
            video_path (str): Path to the input video file.

        Yields:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
                - Frame (np.ndarray): Current video frame.
                - Detection result (Dict): Same format as detect_image() output.

        Raises:
            ValueError: If video file cannot be opened.

        Example:
            >>> detector = FallDetector()
            >>> for frame, result in detector.detect_video("video.mp4"):
            ...     if result["fall_detected"]:
            ...         cv2.imwrite(f"fall_frame_{i}.jpg", result["annotated_image"])
        """
        cap = cv2.VideoCapture(video_path)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect current frame
                result = self.detect_image(frame)

                yield frame, result
        finally:
            cap.release()

    def process_video_file(self,
                          input_path: str,
                          output_path: Optional[str] = None,
                          show_progress: bool = True) -> Dict[str, Any]:
        """Process entire video file and optionally save annotated output.

        Args:
            input_path (str): Path to the input video file.
            output_path (Optional[str], optional): Path to save annotated video. If None, no output is saved. Defaults to None.
            show_progress (bool, optional): Whether to print progress updates. Defaults to True.

        Returns:
            Dict[str, Any]: Processing results containing:
                - video_saved (Optional[str]): Path to saved video or None.
                - total_frames (int): Total number of frames processed.
                - fall_frames (int): Number of frames containing falls.
                - statistics (Dict[str, int]): Detection statistics:
                    - total_detections (int): Total number of detections.
                    - fall_count (int): Total fall detections across all frames.
                    - normal_count (int): Total normal detections across all frames.

        Raises:
            IOError: If input video cannot be opened or output video cannot be written.

        Example:
            >>> detector = FallDetector()
            >>> results = detector.process_video_file(
            ...     "input.mp4",
            ...     "output_annotated.mp4"
            ... )
            >>> print(f"Found falls in {results['fall_frames']} frames")
        """
        cap = cv2.VideoCapture(input_path)

        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize output video (if needed)
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Statistics
        frame_count = 0
        total_fall_count = 0
        total_normal_count = 0
        fall_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect current frame
                result = self.detect_image(frame)

                # Update statistics
                total_fall_count += result["fall_count"]
                total_normal_count += result["normal_count"]
                if result["fall_detected"]:
                    fall_frames += 1

                # Save annotated frame
                if out:
                    out.write(result["annotated_image"])

                # Show progress
                if show_progress and frame_count % 30 == 0:
                    progress = frame_count / total_frames * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            cap.release()
            if out:
                out.release()

        return {
            "video_saved": output_path if output_path else None,
            "total_frames": frame_count,
            "fall_frames": fall_frames,
            "statistics": {
                "total_detections": total_fall_count + total_normal_count,
                "fall_count": total_fall_count,
                "normal_count": total_normal_count
            }
        }