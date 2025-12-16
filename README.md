# DriverAid - Real-time Drowsiness Detection System

## Overview

DriverAid is a real-time drowsiness detection system combining computer vision, deep learning, and expert rules to monitor driver alertness.

## System Architecture

1. **Vision Pipeline** - MediaPipe Face Mesh for face detection, EAR calculation, head pose estimation
2. **CNN Inference** - Lightweight CNN (98.28% accuracy) for eye state prediction
3. **Expert Rules Engine** - Combines signals for drowsiness assessment
4. **Real-time Alerts** - Audio and visual warnings with actionable recommendations

## Quick Start

### Installation

```bash
cd driveraid
pip install -r requirements.txt
```

### Train Model

```bash
cd ml
python train.py
```

### Run Application

```bash
streamlit run app.py
```

## Model Performance

- **Accuracy:** 98.28%
- **Precision:** 98.70% (open), 97.86% (closed)
- **Recall:** 97.86% (open), 98.70% (closed)
- **F1-Score:** 98.28%
- **Parameters:** ~51,000
- **Inference Time:** <50ms average

## Dataset

**MRL Eye Dataset**
- Total: 84,896 images
- Open eyes: 42,952 images
- Closed eyes: 41,944 images
- Split: 70% train, 15% validation, 15% test

## Directory Structure

```
driveraid/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Dependencies
├── services/                   # Core modules
│   ├── vision.py              # Vision pipeline
│   ├── inference.py           # CNN inference
│   └── rules.py               # Expert rules engine
├── models/                     # Trained models
│   ├── cnn_model.keras        # CNN model (98.28% accuracy)
│   └── training_report.json   # Training metrics
├── ml/                         # Training scripts
│   ├── train.py               # Training pipeline
│   ├── dataset/               # MRL Eye Dataset
│   └── DATASET_INSTRUCTIONS.md
└── docs/                       # Documentation
    └── ml_strategy.md         # ML design document
```

## System Requirements

- Python 3.10+
- Webcam
- 4GB RAM minimum
- Windows/Linux/Mac

## Features

- Real-time face detection and tracking
- Eye aspect ratio (EAR) monitoring
- Head pose estimation
- CNN-based eye state classification
- Multi-level alert system (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- Audio alerts with configurable beep patterns
- Temporal smoothing to reduce false positives
- Performance metrics dashboard

## Technical Details

### Vision Pipeline
- MediaPipe Face Mesh (468 landmarks)
- EAR calculation for eye closure detection
- Head pose estimation (pitch, yaw, roll)
- Real-time performance: 10-15 FPS

### Expert Rules
- Eye closure detection (>2s threshold for CRITICAL)
- Head pose monitoring (>35° pitch for >4s triggers HIGH)
- Temporal smoothing with 30-frame history window
- Confidence-based alert decisions

### Alert System
- NONE: Normal operation
- LOW: Early warning signs
- MEDIUM: Moderate drowsiness (slow beeps every 3s)
- HIGH: High drowsiness (fast beeps every 1s)
- CRITICAL: Immediate attention required (loud beeps every 0.5s)

## Performance Optimization

- Frame skipping (CNN inference every 2 frames)
- Lightweight CNN architecture (~51K parameters)
- MediaPipe optimization for CPU execution
- Session state caching in Streamlit
- Efficient numpy operations

## License

Educational project - 2025
