"""
DriverAid CNN Training Script

Trains a lightweight CNN for drowsiness detection using the MRL Eye Dataset.
Target inference speed: <50ms per inference.

Expected dataset structure:
    ml/dataset/
    ├── open/
    └── closed/
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

CONFIG = {
    'img_height': 32,
    'img_width': 32,
    'batch_size': 32,
    'epochs': 15,
    'validation_split': 0.2,
    'test_split': 0.15,
    'random_state': 42
}

class DrowsinessDataLoader:
    """Handles MRL Eye Dataset loading and preprocessing."""
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.open_path = os.path.join(dataset_path, 'open')
        self.closed_path = os.path.join(dataset_path, 'closed')
        
    def verify_structure(self):
        """Verify dataset structure and return image counts."""
        if not os.path.exists(self.open_path):
            raise FileNotFoundError(f"Missing 'open' folder at {self.open_path}")
        if not os.path.exists(self.closed_path):
            raise FileNotFoundError(f"Missing 'closed' folder at {self.closed_path}")
        
        open_count = len([f for f in os.listdir(self.open_path) if f.endswith(('.jpg', '.png'))])
        closed_count = len([f for f in os.listdir(self.closed_path) if f.endswith(('.jpg', '.png'))])
        
        print(f"Dataset verified:")
        print(f"   Open eyes: {open_count} images")
        print(f"   Closed eyes: {closed_count} images")
        print(f"   Total: {open_count + closed_count} images")
        
        return open_count, closed_count
    
    def load_images(self, folder_path, label):
        """Load and preprocess images from folder."""
        images = []
        labels = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = keras.preprocessing.image.load_img(
                        img_path,
                        target_size=(CONFIG['img_height'], CONFIG['img_width']),
                        color_mode='grayscale'
                    )
                    img_array = keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Warning: Skipping {filename}: {e}")
        
        return np.array(images), np.array(labels)
    
    def prepare_dataset(self):
        """Load and split dataset into train/validation/test sets."""
        print("\nLoading dataset...")
        
        X_open, y_open = self.load_images(self.open_path, label=0)
        X_closed, y_closed = self.load_images(self.closed_path, label=1)
        
        X = np.concatenate([X_open, X_closed], axis=0)
        y = np.concatenate([y_open, y_closed], axis=0)
        
        X = X.astype('float32') / 255.0
        
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
        
        print(f"Dataset prepared:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_lightweight_cnn():
    """Build lightweight CNN with 2 conv blocks optimized for speed."""
    model = keras.Sequential([
        layers.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 1)),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model


def train_model():
    """Execute complete training pipeline."""
    print("=" * 60)
    print("DRIVERAID - CNN TRAINING PIPELINE")
    print("=" * 60)
    
    loader = DrowsinessDataLoader()
    loader.verify_structure()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.prepare_dataset()
    
    print("\nBuilding lightweight CNN...")
    model = build_lightweight_cnn()
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
    
    print("\nStarting training...")
    
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
    
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_test, y_pred,
        target_names=['Open Eyes', 'Closed Eyes'],
        digits=4
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    model_path = '../models/cnn_model.keras'
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
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
    print(f"Training report saved to: {report_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return model, history, report


if __name__ == '__main__':
    try:
        model, history, report = train_model()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nINSTRUCTIONS:")
        print("1. Download MRL Eye Dataset")
        print("2. Extract and organize as:")
        print("   ml/dataset/")
        print("   ├── open/")
        print("   └── closed/")
        print("3. Re-run this script")
    except Exception as e:
        print(f"\nTRAINING FAILED: {e}")
        raise
