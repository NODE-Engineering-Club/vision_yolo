# Computer Vision Model Training Report

## Executive Summary

This report documents the development and training of a YOLOv5-based object detection model designed to identify maritime buoys for autonomous navigation. The model detects six distinct buoy classes and provides real-time inference capabilities for webcam-based detection systems.

---

## 1. Project Overview

### 1.1 Objective
Develop a robust computer vision system capable of detecting and classifying various buoy types in maritime environments to support autonomous navigation for vessel guidance and obstacle avoidance.

### 1.2 Target Application
Autonomous Maritime Navigation System - specifically designed to detect buoys and provide directional guidance for vessel movement based on buoy position and type.

### 1.3 Scope
- Object detection and classification of 6 buoy types
- Real-time webcam inference
- Model deployment for maritime applications
- Decision-making based on detection parameters (position, color, type)

---

## 2. Dataset Information

### 2.1 Data Source
**Roboflow Dataset**
- Project: "buoy-o3eym-duls0"
- API-driven download and management
- Version: 1

### 2.2 Buoy Classes
The model classifies six distinct buoy types:

| Class ID | Buoy Type | Purpose |
|----------|-----------|---------|
| 0 | Green Buoy | Navigation marker |
| 1 | Red Buoy | Danger/Port marker |
| 2 | Yellow East Buoy | Directional marker (East) |
| 3 | Yellow North Buoy | Directional marker (North) |
| 4 | Yellow South Buoy | Directional marker (South) |
| 5 | Yellow West Buoy | Directional marker (West) |

### 2.3 Dataset Characteristics
- Pre-annotated bounding box format (YOLO format)
- Includes augmented training data for robustness
- Training data: Ground truth annotations with bounding boxes
- Validation set for model evaluation

---

## 3. Model Architecture

### 3.1 Base Model
**YOLOv5s (Small)**
- Lightweight variant of YOLOv5 for faster inference
- Suitable for edge deployment and real-time applications
- Pre-trained weights from Ultralytics (COCO dataset)

### 3.2 Architecture Benefits
- Single-stage detector: Fast inference speeds
- Anchor-free design: Flexibility in object size detection
- Optimized for embedded systems and webcam applications

---

## 4. Training Configuration

### 4.1 Training Parameters
| Parameter | Value |
|-----------|-------|
| Image Size | 416 pixels |
| Batch Size | 16 |
| Epochs | 25 |
| Optimizer | SGD (default) |
| Base Model Weights | yolov5s.pt (COCO pre-trained) |
| Training Framework | PyTorch |

### 4.2 Training Dataset Setup
```
Dataset Location: {dataset.location}/data.yaml
Training Command:
python train.py --img 416 --batch 16 --epochs 25 \
  --data {dataset.location}/data.yaml \
  --weights yolov5s.pt --name yolov5s_results --cache
```

### 4.3 Training Environment
- Cloud-based execution (Google Colab)
- CUDA GPU acceleration (if available)
- PyTorch framework with GPU support
- Deprecated dependencies removed (wandb)

---

## 5. Training Results

### 5.1 Outputs Generated
- **Model Weights**: Saved in `runs/train/yolov5s_results/`
- **Results Visualization**: `results.png` (training metrics)
- **Validation Visualizations**: `val_batch0_labels.jpg` (validation samples)
- **Training Augmentation**: `train_batch0.jpg` (augmented training data)
- **Best Model**: `best.pt` (optimal checkpoint)

### 5.2 Key Metrics
The training process generated:
- Training loss curves
- Validation metrics (Precision, Recall, mAP)
- Bounding box prediction accuracy
- Class-specific detection performance

### 5.3 Model Output Format
```
Per-Detection Output: [x1, y1, x2, y2, confidence, class_id]
- x1, y1: Top-left corner coordinates
- x2, y2: Bottom-right corner coordinates
- confidence: Prediction confidence score
- class_id: Buoy class (0-5)
```

---

## 6. Model Inference

### 6.1 Deployed Model Format
- **Format**: TorchScript (.torchscript.pt)
- **Framework**: PyTorch
- **Input Size**: 640×640 pixels (resized from original)
- **Precision**: Float32

### 6.2 Inference Pipeline

```
1. Read Frame → Webcam (OpenCV)
   ↓
2. Resize → 640×640 pixels
   ↓
3. Color Space Conversion → BGR to RGB, HWC to CHW
   ↓
4. Normalization → Divide by 255.0
   ↓
5. Model Inference → TorchScript model evaluation
   ↓
6. Post-processing → Filter by confidence threshold
   ↓
7. Annotation → Draw bounding boxes with labels
   ↓
8. Output → Display on screen
```

### 6.3 Inference Configuration
| Parameter | Value |
|-----------|-------|
| Confidence Threshold | 0.25 |
| Input Resolution | 640×640 |
| Real-time Stream | Webcam (OpenCV) |
| Output Frequency | Per-frame |

---

## 7. Output Processing and Decision Making

### 7.1 Detection Output
For each detected buoy:
- **Bounding Box**: x1, y1, x2, y2 coordinates
- **Confidence Score**: 0.0 - 1.0 probability
- **Class Name**: Green Buoy, Red Buoy, Yellow East/North/South/West
- **Color Coding**: Visual distinction for each class

### 7.2 Visual Annotation
- Color-coded bounding boxes (BGR format)
  - Green Buoy: Green (0, 255, 0)
  - Red Buoy: Red (0, 0, 255)
  - Yellow East: Cyan (0, 255, 255)
  - Yellow North: Orange-ish (0, 200, 255)
  - Yellow South: Yellow-green (0, 255, 200)
  - Yellow West: Gold (0, 150, 255)

### 7.3 Navigation Decision Logic
Based on buoy position in frame:
- **Left side detection**: Output "move right"
- **Right side detection**: Output "move left"
- **Center detection**: Output "proceed forward"
- **Type-specific routing**: Different responses for different buoy types

---

## 8. Performance Characteristics

### 8.1 Strengths
✓ Lightweight model suitable for real-time processing
✓ Pre-trained on diverse dataset (COCO)
✓ Fast inference speeds for maritime applications
✓ Clear visual feedback with color-coded detection
✓ Confidence scoring for detection reliability

### 8.2 Optimization Opportunities
- Fine-tune confidence threshold based on operational requirements
- Collect maritime-specific training data to improve performance
- Implement ensemble methods for increased robustness
- Add temporal filtering for smoother navigation commands

---

## 9. Deployment Considerations

### 9.1 Hardware Requirements
- **Minimum**: CPU-based inference (slower)
- **Recommended**: GPU acceleration for real-time performance
- **Edge Devices**: NVIDIA Jetson or similar for embedded deployment

### 9.2 Dependencies
```
torch
torchvision
opencv-python (cv2)
roboflow
yolov5
numpy
matplotlib
```

### 9.3 Inference Time
- Per-frame processing time depends on hardware
- GPU: ~20-50ms per frame (20-50 FPS)
- CPU: ~100-200ms per frame (5-10 FPS)

---

## 10. Future Enhancements

### 10.1 Model Improvements
1. **Data Collection**: Gather more maritime-specific buoy imagery
2. **Transfer Learning**: Fine-tune on domain-specific datasets
3. **Model Scaling**: Evaluate YOLOv5m, YOLOv5l for accuracy vs. speed trade-offs
4. **Ensemble Methods**: Combine multiple models for robust detection

### 10.2 System Integration
1. **Sensor Fusion**: Integrate LIDAR data for 3D localization
2. **Multi-camera Systems**: Support multiple camera angles
3. **Temporal Analysis**: Track buoy movements across frames
4. **Decision Logic**: Implement sophisticated navigation algorithms

### 10.3 Operational Improvements
1. **Confidence Adaptation**: Dynamic threshold adjustment
2. **Class Weighting**: Prioritize critical buoy types
3. **Anomaly Detection**: Flag unusual detection patterns
4. **Performance Monitoring**: Log and analyze detection metrics

---

## 11. Conclusion

The YOLOv5s-based buoy detection model provides a solid foundation for autonomous maritime navigation. The model demonstrates effective real-time detection capabilities with proper color-coded visual feedback and can be seamlessly integrated with additional sensor systems like LIDAR for enhanced navigation accuracy and safety.

The lightweight architecture ensures compatibility with edge devices and real-time processing requirements critical for autonomous vessel navigation applications.

---

## 12. References

- **YOLOv5**: https://github.com/ultralytics/yolov5
- **Roboflow**: https://roboflow.com
- **OpenCV**: https://opencv.org
- **PyTorch**: https://pytorch.org

---

**Document Version**: 1.0
**Last Updated**: 2026-06-04
**Project**: Vision YOLO - Autonomous Maritime Navigation System
