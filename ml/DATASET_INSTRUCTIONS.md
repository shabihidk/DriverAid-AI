# MRL Eye Dataset - Setup Instructions

## Phase 2: Data Preparation

### 📥 Dataset Source
**MRL Eye Dataset** - A public dataset for eye state classification.

**Download Link:** http://mrl.cs.vsb.cz/eyedataset

**Alternative:** Search for "MRL Eye Dataset" or use similar datasets like:
- CEW Dataset
- Closed Eyes In The Wild (CEW)
- Any labeled open/closed eyes dataset

---

### 📁 Required Directory Structure

After downloading, organize images into this structure:

```
ml/dataset/
├── open/           # All OPEN eye images go here
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── closed/         # All CLOSED eye images go here
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

---

### ⚙️ Preparation Steps

#### Option 1: MRL Eye Dataset (Recommended)
1. Download from http://mrl.cs.vsb.cz/eyedataset
2. Extract the archive
3. The dataset typically has folders like:
   - `1/` (Person 1's images)
   - `2/` (Person 2's images)
   - etc.
4. Inside each person folder, there are images labeled with eye state
5. **Organize by eye state:**
   - Move/copy all images with open eyes → `ml/dataset/open/`
   - Move/copy all images with closed eyes → `ml/dataset/closed/`

#### Option 2: Manual Dataset
If you can't download MRL, create a minimal test dataset:
1. Capture ~100 images of your own eyes (open state)
2. Capture ~100 images of your own eyes (closed state)
3. Place them in the respective folders

---

### 📊 Dataset Requirements

**Minimum for Training:**
- At least 200 images per class (open/closed)
- Recommended: 500-1000 images per class

**Image Format:**
- JPG, PNG, or JPEG
- Any resolution (will be auto-resized to 32x32)
- Grayscale or color (will be converted to grayscale)

---

### ✅ Verification

After organizing, run this to verify:

```bash
cd ml
python -c "from train import DrowsinessDataLoader; DrowsinessDataLoader().verify_structure()"
```

Expected output:
```
✅ Dataset verified:
   Open eyes: XXX images
   Closed eyes: XXX images
   Total: XXX images
```

---

### 🚀 Next Steps

Once dataset is organized:
```bash
cd ml
python train.py
```

This will:
1. Load and split the dataset (70% train, 15% val, 15% test)
2. Train a lightweight CNN (~15 epochs)
3. Generate classification report (for Viva)
4. Save model to `models/cnn_model.h5`

**Expected training time:** 2-5 minutes on CPU (depending on dataset size)

---

### ⚠️ Troubleshooting

**Error: "Missing 'open' folder"**
- Ensure folders are named exactly: `open` and `closed` (lowercase)
- Check you're in the correct directory: `ml/dataset/`

**Error: "No images found"**
- Verify image formats: .jpg, .png, .jpeg
- Check file permissions

**Low accuracy (<70%)**
- Dataset may be too small
- Try collecting more diverse images
- Ensure images are properly labeled

---

**Status:** Ready for dataset preparation
**Next:** After dataset is ready, run `python train.py`
