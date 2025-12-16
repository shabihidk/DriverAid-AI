# DriverAid - Real-time Drowsiness Detection System

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

### Directory Structure
```
driveraid/
├── app.py                  # Main Streamlit Entry
├── requirements.txt        # Python dependencies
├── services/               # Core logic modules
│   ├── vision.py          # (Coming in Phase 3)
│   ├── inference.py       # (Coming in Phase 2)
│   └── rules.py           # (Coming in Phase 4)
├── models/                 # Trained CNN models
├── tests/                  # Unit tests
├── docs/                   # Documentation
└── ml/                     # ML training scripts
    ├── train.py           # (Coming in Phase 2)
    └── dataset/           # MRL Eye Dataset
```

### System Requirements
- Python 3.10+
- Webcam
- 4GB RAM minimum
- Windows/Linux/Mac

---
**Status:** Phase 1 Complete - Awaiting Verification
