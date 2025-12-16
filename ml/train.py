"""
DriverAid - CNN Training Script
Phase 2: Data & ML Strategy

GOAL: Train a LIGHTWEIGHT, FAST CNN for drowsiness detection.
PRIORITY: Inference speed > Perfect accuracy (Target: <50ms per inference)

Dataset: MRL Eye Dataset
Structure Expected:
    ml/dataset/
    ├── open/      # Open eyes images
    └── closed/    # Closed eyes images
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

# Configuration
CONFIG = {
    'img_height': 32,  # Small size for speed
    'img_width': 32,
    'batch_size': 32,
    'epochs': 15,
    'validation_split': 0.2,
    'test_split': 0.15,
    'random_state': 42
}

class DrowsinessDataLoader:
    """Handles MRL Eye Dataset loading and preprocessing"""
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.open_path = os.path.join(dataset_path, 'open')
        self.closed_path = os.path.join(dataset_path, 'closed')
        
    def verify_structure(self):
        """Verify dataset structure exists"""
        if not os.path.exists(self.open_path):
            raise FileNotFoundError(f"Missing 'open' folder at {self.open_path}")
        if not os.path.exists(self.closed_path):
            raise FileNotFoundError(f"Missing 'closed' folder at {self.closed_path}")
        
        open_count = len([f for f in os.listdir(self.open_path) if f.endswith(('.jpg', '.png'))])
        closed_count = len([f for f in os.listdir(self.closed_path) if f.endswith(('.jpg', '.png'))])
        
        print(f"✅ Dataset verified:")
        print(f"   Open eyes: {open_count} images")
        print(f"   Closed eyes: {closed_count} images")
        print(f"   Total: {open_count + closed_count} images")
        
        return open_count, closed_count
    
    def load_images(self, folder_path, label):
        """Load images from a folder with given label"""
        images = []
        labels = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                try:
                    # Load and resize
                    img = keras.preprocessing.image.load_img(
                        img_path,
                        target_size=(CONFIG['img_height'], CONFIG['img_width']),
                        color_mode='grayscale'  # Grayscale for speed
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"⚠️ Skipping {filename}: {e}")
        
        return np.array(images), np.array(labels)
    
    def prepare_dataset(self):
        """Load and prepare train/val/test splits"""
        print("\n📦 Loading dataset...")
        
        # Load images
        X_open, y_open = self.load_images(self.open_path, label=0)  # 0 = Open
        X_closed, y_closed = self.load_images(self.closed_path, label=1)  # 1 = Closed
        
        # Combine
        X = np.concatenate([X_open, X_closed], axis=0)
        y = np.concatenate([y_open, y_closed], axis=0)
        
        # Normalize to [0, 1]
        X = X.astype('float32') / 255.0
        
        # Shuffle and split: Train (70%) | Val (15%) | Test (15%)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, 
            test_size=CONFIG['test_split'], 
            random_state=CONFIG['random_state'],
            stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=CONFIG['validation_split'],
            random_state=CONFIG['random_state'],
            stratify=y_train_val
        )
        
        print(f"✅ Dataset prepared:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_lightweight_cnn():
    """
    Build LIGHTWEIGHT CNN optimized for speed.
    Architecture: 2 Conv blocks + Dense layer (Minimal parameters)
    """
    model = keras.Sequential([
        # Input: 32x32x1 grayscale
        layers.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 1)),
        
        # Conv Block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Conv Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary: Open (0) vs Closed (1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def train_model():
    """Main training pipeline"""
    print("=" * 60)
    print("🚗 DRIVERAID - CNN TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load Data
    loader = DrowsinessDataLoader()
    loader.verify_structure()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_dataset()
    
    # Step 2: Build Model
    print("\n🏗️ Building lightweight CNN...")
    model = build_lightweight_cnn()
    model.summary()
    
    print(f"\n📊 Total parameters: {model.count_params():,}")
    
    # Step 3: Train
    print("\n🔥 Starting training...")
    
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=[early_stop],
        verbose=1
    )
    
    # Step 4: Evaluate on Test Set
    print("\n📈 Evaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    # Generate predictions for classification report
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    # Classification Report (Required for Viva)
    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT (For Final Report)")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Open Eyes', 'Closed Eyes'],
        digits=4
    ))
    
    print("\n🎯 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Step 5: Save Model
    model_path = '../models/cnn_model.h5'
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Step 6: Save Training Report
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': CONFIG,
        'model_params': int(model.count_params()),
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'loss': float(test_loss)
        },
        'training_epochs': len(history.history['loss']),
        'final_train_acc': float(history.history['accuracy'][-1]),
        'final_val_acc': float(history.history['val_accuracy'][-1])
    }
    
    report_path = '../models/training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"📄 Training report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, history, report


if __name__ == '__main__':
    try:
        model, history, report = train_model()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n📖 INSTRUCTIONS:")
        print("1. Download MRL Eye Dataset")
        print("2. Extract and organize as:")
        print("   ml/dataset/")
        print("   ├── open/      # All open eye images")
        print("   └── closed/    # All closed eye images")
        print("3. Re-run this script")
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        raise
