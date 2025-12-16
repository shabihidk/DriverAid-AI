# DriverAid - Real-time Drowsiness Detection System

## Progress Tracker

- ✅ **Phase 1:** Setup & Skeleton - COMPLETE
- ✅ **Phase 2:** Data & ML Strategy - COMPLETE (98.28% accuracy!)
- 🚧 **Phase 3:** Vision Pipeline - IN PROGRESS
- ⏳ **Phase 4:** Integration & Rules Engine - PENDING
- ⏳ **Phase 5:** Testing & Documentation - PENDING

---

## Phase 1: Setup & Skeleton ✅

### Quick Start

1. **Install Dependencies:**
   ```bash
   cd driveraid
   pip install -r requirements.txt
   ```

2. **Run Webcam Test:**
   ```bash
   streamlit run app.py
   ```

3. **Verify:**
   - Check the "Start Webcam Test" checkbox
   - Confirm you see live video feed
   - Verify frame counter is incrementing

---

## Phase 2: Data & ML Strategy ✅

### CNN Training Complete

**Model Performance:**
- **Accuracy:** 98.28%
- **Precision:** 98.28%
- **Recall:** 98.28%
- **F1-Score:** 98.28%
- **Parameters:** ~51,000 (Lightweight for real-time inference)

**Dataset:** MRL Eye Dataset
- Training: 59,427 images
- Validation: 12,734 images
- Test: 12,735 images

**To retrain the model:**
```bash
cd ml
python train.py
```

**Outputs:**
- `models/cnn_model.keras` - Trained model
- `models/training_report.json` - Metrics for Viva presentation

---

## Directory Structure
```
driveraid/
├── app.py                  # Main Streamlit Entry
├── requirements.txt        # Python dependencies
├── services/               # Core logic modules
│   ├── vision.py          # Phase 3: MediaPipe integration
│   ├── inference.py       # Phase 4: CNN inference wrapper
│   └── rules.py           # Phase 4: Expert system
├── models/                 # Trained CNN models
│   ├── cnn_model.keras    # ✅ Trained model (98.28% acc)
│   └── training_report.json
├── ml/                     # ML training scripts
│   ├── train.py           # ✅ Training pipeline
│   └── dataset/           # MRL Eye Dataset (excluded from git)
├── tests/                  # Unit tests
└── docs/                   # Documentation
    └── ml_strategy.md     # ✅ Phase 2 design document
```

### System Requirements
- Python 3.10+
- Webcam
- 4GB RAM minimum
- Windows/Linux/Mac

---
**Status:** Phase 2 Complete | Starting Phase 3: Vision Pipeline
