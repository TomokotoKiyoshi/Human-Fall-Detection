#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Core detector module for fall detection.

Provides FallDetector class using YOLOv8 model.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Any, Optional
from ultralytics import YOLO
import onnxruntime as ort

__all__ = ['FallDetector']


class FallDetector:
    """
    Fall detection class using YOLOv8 model.

    Detects human falls in images and videos using a fine-tuned YOLOv8 model.
    """

    def __init__(self,
                 model_type: str = "onnx",
                 device: str = "cuda",
                 confidence: float = 0.5):
        """Initialize the fall detector.

        Args:
            model_type (str): Model type - 'pt' for PyTorch or 'onnx' for ONNX FP16. Defaults to "onnx".
            device (str): Device to run inference on. Defaults to "cuda".
            confidence (float): Detection confidence threshold. Defaults to 0.5.
        """
        self.device = device
        self.confidence = confidence
        self.model_type = model_type
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.input_shape = None

        # Get model path relative to this file
        base_dir = Path(__file__).parent
        if model_type == "pt":
            self.model_path = base_dir / 'models' / 'best.pt'
            self.model = YOLO(str(self.model_path))
        elif model_type == "onnx":
            self.model_path = base_dir / 'models' / 'best_fp16.onnx'
            self._load_onnx_model()
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'pt' or 'onnx'")


    def _load_onnx_model(self) -> None:
        """Load ONNX model using onnxruntime."""
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in self.device else ['CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.ort_session.get_outputs()]

    def _preprocess_onnx(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX inference.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image ready for ONNX inference
        """
        # Get input dimensions
        _, _, height, width = self.input_shape

        # Resize image
        img = cv2.resize(image, (width, height))

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Transpose to CHW format
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess_onnx(self, outputs: List[np.ndarray], original_shape: Tuple[int, int]) -> List[Dict]:
        """Postprocess ONNX outputs to detection format.

        Args:
            outputs: Raw ONNX model outputs
            original_shape: Original image shape (height, width)

        Returns:
            List of detections
        """
        # YOLOv8 ONNX输出格式: [batch, 84, 8400]
        # 84 = 4 (bbox) + 80 (classes) 或 4 + 2 (for 2 classes)
        predictions = outputs[0][0]  # Remove batch dimension

        # Get dimensions
        num_classes = predictions.shape[0] - 4
        num_boxes = predictions.shape[1]

        # Extract boxes and scores
        boxes = predictions[:4, :].T  # [8400, 4]
        scores = predictions[4:, :].T  # [8400, num_classes]

        # Apply confidence threshold
        max_scores = np.max(scores, axis=1)
        mask = max_scores > self.confidence

        if not np.any(mask):
            return []

        # Filter by confidence
        boxes = boxes[mask]
        scores = scores[mask]
        max_scores = max_scores[mask]
        class_ids = np.argmax(scores, axis=1)

        # Convert from YOLO format to xyxy
        detections = []
        h_orig, w_orig = original_shape
        _, _, h_model, w_model = self.input_shape

        for i in range(len(boxes)):
            # YOLO format: [cx, cy, w, h]
            cx, cy, w, h = boxes[i]

            # Convert to xyxy
            x1 = (cx - w/2) * w_orig / w_model
            y1 = (cy - h/2) * h_orig / h_model
            x2 = (cx + w/2) * w_orig / w_model
            y2 = (cy + h/2) * h_orig / h_model

            detections.append({
                'x1': max(0, int(x1)),
                'y1': max(0, int(y1)),
                'x2': min(w_orig, int(x2)),
                'y2': min(h_orig, int(y2)),
                'confidence': float(max_scores[i]),
                'class': int(class_ids[i])
            })

        # Apply NMS
        detections = self._nms(detections, iou_threshold=0.45)
        return detections

    def _nms(self, detections: List[Dict], iou_threshold: float = 0.45) -> List[Dict]:
        """Apply Non-Maximum Suppression.

        Args:
            detections: List of detection dictionaries
            iou_threshold: IoU threshold for NMS

        Returns:
            Filtered detections after NMS
        """
        if not detections:
            return []

        # Convert to arrays for easier manipulation
        boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        classes = np.array([d['class'] for d in detections])

        # Apply NMS per class
        keep = []
        for cls in np.unique(classes):
            cls_mask = classes == cls
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # Sort by score
            order = cls_scores.argsort()[::-1]

            keep_cls = []
            while order.size > 0:
                i = order[0]
                keep_cls.append(cls_indices[i])

                # Calculate IoU
                xx1 = np.maximum(cls_boxes[i, 0], cls_boxes[order[1:], 0])
                yy1 = np.maximum(cls_boxes[i, 1], cls_boxes[order[1:], 1])
                xx2 = np.minimum(cls_boxes[i, 2], cls_boxes[order[1:], 2])
                yy2 = np.minimum(cls_boxes[i, 3], cls_boxes[order[1:], 3])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)

                inter = w * h
                area1 = (cls_boxes[i, 2] - cls_boxes[i, 0]) * (cls_boxes[i, 3] - cls_boxes[i, 1])
                area2 = (cls_boxes[order[1:], 2] - cls_boxes[order[1:], 0]) * \
                       (cls_boxes[order[1:], 3] - cls_boxes[order[1:], 1])

                iou = inter / (area1 + area2 - inter)
                order = order[np.where(iou <= iou_threshold)[0] + 1]

            keep.extend(keep_cls)

        return [detections[i] for i in keep]

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
        detections = []
        fall_count = 0
        normal_count = 0
        annotated_image = image.copy()

        if self.model_type == 'onnx':
            # ONNX推理路径
            # Preprocess image
            preprocessed = self._preprocess_onnx(image)

            # Run inference
            outputs = self.ort_session.run(self.output_names, {self.input_name: preprocessed})

            # Postprocess results
            onnx_detections = self._postprocess_onnx(outputs, image.shape[:2])

            # Convert ONNX format to standard format
            for det in onnx_detections:
                # Determine class
                if det['class'] == 0:  # Normal
                    class_name = "Normal"
                    normal_count += 1
                else:  # Fall
                    class_name = "Fall"
                    fall_count += 1

                detections.append({
                    "class": class_name,
                    "confidence": det['confidence'],
                    "bbox": [det['x1'], det['y1'], det['x2'], det['y2']]
                })

        else:
            # 原始PyTorch推理路径
            results = self.model(image,
                                verbose=False,
                                device=self.device,
                                conf=self.confidence)

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
                            normal_count += 1
                        else:  # Fall
                            class_name = "Fall"
                            fall_count += 1

                        # Save detection result
                        detections.append({
                            "class": class_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

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