# ML Strategy - Phase 2 Design Document

## Objective
Train a **lightweight, fast** CNN for binary eye state classification (Open vs Closed) optimized for real-time inference on laptops.

---

## Design Principles

### 1. Speed Over Accuracy
**Target:** <50ms inference time per frame on CPU
- **Why:** Real-time detection requires 10-15 FPS
- **Tradeoff:** Accept 90-95% accuracy instead of 99% if it runs 3x faster

### 2. Lightweight Architecture
**Model Size:** <500KB
- **Input:** 32x32 grayscale images (minimal preprocessing)
- **Layers:** 2 Conv blocks + 1 Dense layer
- **Parameters:** ~50,000 (vs typical CNNs with millions)

### 3. Binary Classification Only
**Classes:** 
- 0 = Open Eyes
- 1 = Closed Eyes

**Not detecting:**
- Yawning (handled by MediaPipe + Rules)
- Head pose (handled by MediaPipe)
- Facial features (delegated to MediaPipe Face Mesh)

---

## Architecture Details

```
Input: 32x32x1 (Grayscale)
    ↓
Conv2D(16 filters, 3x3) + ReLU + MaxPool
    ↓
Dropout(0.25)
    ↓
Conv2D(32 filters, 3x3) + ReLU + MaxPool
    ↓
Dropout(0.25)
    ↓
Flatten
    ↓
Dense(64) + ReLU + Dropout(0.5)
    ↓
Dense(1, Sigmoid) → Output [0, 1]
```

**Total Parameters:** ~50,000
**Output:** Probability score (0.0 = Open, 1.0 = Closed)

---

## Dataset: MRL Eye Dataset

**Source:** http://mrl.cs.vsb.cz/eyedataset

**Structure:**
- Multiple subjects with labeled eye images
- Binary labels: Open / Closed
- Typical size: 2000-4000 images total

**Preprocessing:**
1. Resize to 32x32
2. Convert to grayscale
3. Normalize to [0, 1]

**Splits:**
- Train: 70%
- Validation: 15%
- Test: 15%

---

## Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 15 | Sufficient for small dataset |
| Batch Size | 32 | Balance speed/memory |
| Optimizer | Adam | Fast convergence |
| Loss | Binary Crossentropy | Binary classification |
| Early Stopping | Yes (patience=3) | Prevent overfitting |

---

## Integration Strategy

### Phase 3 & 4 Integration Flow:
```
Webcam Frame
    ↓
MediaPipe Face Mesh (Detects face landmarks)
    ↓
Extract Eye Regions (Left & Right)
    ↓
Resize to 32x32, Grayscale
    ↓
CNN Inference (Every 2-3 frames to save CPU)
    ↓
Output: eye_closed_confidence [0.0 - 1.0]
    ↓
Rules Engine: Combine with EAR, Head Pose
    ↓
Alert Decision
```

**Key Optimization:** 
- **Do NOT run CNN on every frame** (too expensive)
- Sample every 2-3 frames
- Use rolling average for predictions

---

## Expected Performance Metrics

### Accuracy Targets (Test Set):
- **Accuracy:** >90%
- **Precision:** >88%
- **Recall:** >90%
- **F1-Score:** >89%

### Inference Speed:
- **Target:** 20-40ms per inference (CPU)
- **Measurement:** Will be tested in Phase 4 integration

---

## Deliverables

1. **Trained Model:** `models/cnn_model.h5`
2. **Training Report:** `models/training_report.json` (includes metrics)
3. **Classification Report:** Precision, Recall, F1 (for Viva presentation)

---

## Limitations & Mitigation

| Limitation | Mitigation |
|------------|------------|
| False positives during blinks | Rule Engine: Ignore closures <0.5s |
| Low light conditions | MediaPipe handles face detection; CNN only sees normalized pixels |
| Glasses/Sunglasses | Dataset limitation; may need augmentation in future |

---

**Status:** Phase 2 - Awaiting dataset preparation
**Next:** Organize MRL dataset → Run `train.py` → Verify model saves
