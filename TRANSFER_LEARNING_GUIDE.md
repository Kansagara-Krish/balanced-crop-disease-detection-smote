# Transfer Learning Implementation Guide

## Overview

This document explains the transfer learning upgrade to the crop leaf disease classification project. The codebase now supports three pretrained architectures in addition to the existing custom CNN baseline:

1. **MobileNetV2** - Lightweight model, fast inference, good for mobile deployment
2. **EfficientNetB0** - Efficient architecture, balanced accuracy/speed trade-off
3. **ResNet50** - Powerful model, best accuracy, higher computational cost

All models use ImageNet pretrained weights and follow the same training pipeline for consistency.

---

## Architecture Overview

### Model Components

Each transfer learning model follows this architecture:

```
Input (224×224×3)
    ↓
Pretrained Base Model (ImageNet weights, include_top=False)
    ↓ [Base Model Frozen Initially]
GlobalAveragePooling2D()
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Dense(num_classes, activation='softmax') [Output Layer]
```

### Key Features

| Component | Specification |
|-----------|---------------|
| Input Shape | (224, 224, 3) |
| Base Model Weights | ImageNet (pretrained) |
| Base Model Top Layers | Excluded (include_top=False) |
| Pooling Strategy | Global Average Pooling |
| Dense Head | 128 units, ReLU activation |
| Regularization | Dropout(0.3) |
| Output | Softmax (multiclass) |
| Optimizer | Adam(lr=0.0001) |
| Loss | Categorical Crossentropy |
| Metrics | Accuracy |

---

## File Structure

### New Files Created

```
cnn_model/
├── utils/
│   ├── transfer_learning_utils.py       # ✨ NEW: Transfer learning utilities
│   ├── data_utils.py                     # [Unchanged]
│   ├── model_utils.py                    # [Unchanged - Custom CNN still available]
│   └── visualization_utils.py            # [Unchanged]
│
├── models/
│   ├── train_mobilenetv2.py             # ✨ NEW: MobileNetV2 training script
│   ├── train_efficientnetb0.py          # ✨ NEW: EfficientNetB0 training script
│   ├── train_resnet50.py                # ✨ NEW: ResNet50 training script
│   ├── mobilenetv2/                      # Models saved here
│   ├── efficientnetb0/                   # Models saved here
│   ├── resnet50/                         # Models saved here
│   └── cnn/                              # [Existing custom CNN models]
│
├── results/
│   ├── reports/
│   │   ├── mobilenetv2_config.json      # ✨ NEW: Model configuration
│   │   ├── mobilenetv2_metrics.json     # ✨ NEW: Evaluation metrics
│   │   ├── efficientnetb0_config.json   # ✨ NEW
│   │   ├── efficientnetb0_metrics.json  # ✨ NEW
│   │   ├── resnet50_config.json         # ✨ NEW
│   │   └── resnet50_metrics.json        # ✨ NEW
│   └── [Other result files]
│
└── requirements.txt                      # [Updated with optional dependencies]
```

---

## Transfer Learning Utilities (`transfer_learning_utils.py`)

### Model Builders

#### MobileNetV2
```python
from utils.transfer_learning_utils import build_mobilenetv2_model

model = build_mobilenetv2_model(
    num_classes=25,                # Your number of classes
    input_shape=(224, 224, 3),     # Fixed input size
    learning_rate=0.0001            # Initial learning rate
)
```

**Advantages:**
- Very lightweight (~14MB)
- Fast inference, ideal for mobile/edge devices
- Lower memory requirements
- Good performance for resource-constrained environments

---

#### EfficientNetB0
```python
from utils.transfer_learning_utils import build_efficientnetb0_model

model = build_efficientnetb0_model(
    num_classes=25,
    input_shape=(224, 224, 3),
    learning_rate=0.0001
)
```

**Advantages:**
- Efficient scaling (great accuracy/efficiency ratio)
- Moderate model size (~29MB)
- Balanced performance
- Good for production systems

---

#### ResNet50
```python
from utils.transfer_learning_utils import build_resnet50_model

model = build_resnet50_model(
    num_classes=25,
    input_shape=(224, 224, 3),
    learning_rate=0.0001
)
```

**Advantages:**
- Highest accuracy among the three
- Well-established architecture
- Large model (~98MB)
- Best for accuracy-critical applications

---

### Preprocessing Functions

Each model has model-specific preprocessing to match ImageNet normalization:

```python
from utils.transfer_learning_utils import preprocess_images

# Apply model-specific preprocessing
X_train_processed = preprocess_images(X_train, model_type='mobilenetv2')
X_val_processed = preprocess_images(X_val, model_type='mobilenetv2')
X_test_processed = preprocess_images(X_test, model_type='mobilenetv2')
```

**Supported model types:**
- `'mobilenetv2'`
- `'efficientnetb0'`
- `'resnet50'`

---

### Two-Phase Training

All transfer learning models use a two-phase training approach for better results:

#### Phase 1: Train with Frozen Base
```python
from utils.transfer_learning_utils import train_two_phase_transfer_learning

history = train_two_phase_transfer_learning(
    model=model,
    X_train=X_train_processed,
    y_train=y_train,
    X_val=X_val_processed,
    y_val=y_val,
    epochs_phase1=20,              # Phase 1: Frozen base
    epochs_phase2=15,              # Phase 2: Fine-tuning
    batch_size=32,
    num_unfreeze_layers=50,        # Layers to unfreeze in phase 2
    model_name='mobilenetv2',
    output_dir='models/mobilenetv2'
)
```

**Phase 1 (20 epochs, frozen base):**
- Only custom head layers are trained
- Base model weights remain fixed
- Fast convergence, lower memory usage
- Benefits from ImageNet pretraining

**Phase 2 (15 epochs, fine-tuning):**
- Unfreeze last 50 layers of base model
- Lower learning rate (0.0001) to prevent drastic changes
- Fine-tune for task-specific features
- Improved performance on leaf disease classification

---

### Evaluation and Saving

```python
from utils.transfer_learning_utils import (
    evaluate_transfer_learning_model,
    save_transfer_learning_model,
    load_transfer_learning_model
)

# Evaluate on test set
metrics = evaluate_transfer_learning_model(
    model=model,
    X_test=X_test_processed,
    y_test=y_test,
    class_names=['Healthy', 'Disease1', 'Disease2', ...]
)

# Metrics include:
# - accuracy, precision, recall, f1_score
# - confusion_matrix
# - classification_report
# - predictions

# Save trained model
model_path = save_transfer_learning_model(
    model=model,
    model_name='mobilenetv2_model',
    output_dir='models/mobilenetv2'
)

# Load saved model
loaded_model = load_transfer_learning_model(model_path)
```

---

## Training Scripts

### Running Individual Models

Each model has a dedicated training script that handles the complete pipeline:

#### MobileNetV2
```bash
cd cnn_model/models
python train_mobilenetv2.py
```

**Script includes:**
1. Configuration setup
2. Data loading from preprocessed files
3. Model-specific preprocessing
4. Model building
5. Two-phase training
6. Test set evaluation
7. Model and metrics saving
8. Training summary

---

#### EfficientNetB0
```bash
cd cnn_model/models
python train_efficientnetb0.py
```

Same pipeline as MobileNetV2.

---

#### ResNet50
```bash
cd cnn_model/models
python train_resnet50.py
```

Same pipeline as MobileNetV2.

---

### Expected Output

Each training script produces:

```
======================================================================
MobileNetV2 Transfer Learning Training
======================================================================

Configuration:
  Model: MobileNetV2
  Input Shape: (224, 224, 3)
  Learning Rate: 0.0001
  Phase 1 Epochs (frozen): 20
  Phase 2 Epochs (fine-tuning): 15
  ...

Loading preprocessed data...
✓ Data loaded successfully!
  X_train shape: (1234, 224, 224, 3)
  X_val shape: (456, 224, 224, 3)
  X_test shape: (789, 224, 224, 3)
  Number of classes: 25

Applying MobileNetV2-specific preprocessing...
✓ Preprocessing completed!

Building MobileNetV2 model...
✓ Model built successfully!
  Total parameters: 3,532,163
  Trainable parameters: 10,691
  Non-trainable parameters: 3,521,472

PHASE 1: Training with Frozen Base Layers
...
✓ Phase 1 completed!

PHASE 2: Fine-tuning with Unfrozen Layers
...
✓ Phase 2 completed!

Evaluating on test set...
✓ Evaluation completed!
  Accuracy:  0.9234
  Precision: 0.9187
  Recall:    0.9234
  F1-Score:  0.9210

Saving model and metrics...
✓ Model saved to: models/mobilenetv2/mobilenetv2_model.h5
✓ Metrics saved to: results/reports/mobilenetv2_metrics.json
✓ Config saved to: results/reports/mobilenetv2_config.json

======================================================================
Training Summary
======================================================================
Model Name: mobilenetv2
Model Path: models/mobilenetv2/mobilenetv2_model.h5
...
```

---

## Advanced Usage

### Custom Two-Phase Training

You can also use the functions directly in notebooks for more control:

```python
import sys
sys.path.insert(0, '../utils')

from transfer_learning_utils import (
    build_mobilenetv2_model,
    preprocess_images,
    train_two_phase_transfer_learning,
    evaluate_transfer_learning_model,
    save_transfer_learning_model
)
import numpy as np
import json

# Load data
X_train = np.load('../results/preprocessed_data/X_train.npy')
y_train = np.load('../results/preprocessed_data/y_train.npy')
X_val = np.load('../results/preprocessed_data/X_val.npy')
y_val = np.load('../results/preprocessed_data/y_val.npy')
X_test = np.load('../results/preprocessed_data/X_test.npy')
y_test = np.load('../results/preprocessed_data/y_test.npy')

with open('../results/preprocessed_data/label_encoding.json', 'r') as f:
    label_encoding = json.load(f)

num_classes = len(label_encoding)

# Preprocess
X_train_proc = preprocess_images(X_train, 'mobilenetv2')
X_val_proc = preprocess_images(X_val, 'mobilenetv2')
X_test_proc = preprocess_images(X_test, 'mobilenetv2')

# Build and train
model = build_mobilenetv2_model(num_classes=num_classes)

history = train_two_phase_transfer_learning(
    model=model,
    X_train=X_train_proc,
    y_train=y_train,
    X_val=X_val_proc,
    y_val=y_val,
    epochs_phase1=20,
    epochs_phase2=15
)

# Evaluate
class_names = sorted(label_encoding.keys(), key=lambda x: label_encoding[x])
metrics = evaluate_transfer_learning_model(
    model=model,
    X_test=X_test_proc,
    y_test=y_test,
    class_names=class_names
)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

---

### Fine-tuning Different Numbers of Layers

```python
from utils.transfer_learning_utils import unfreeze_base_model

# Freeze everything initially
model = build_mobilenetv2_model(num_classes=25)

# Train phase 1 with frozen base...
# history1 = train_transfer_learning_model(...)

# Phase 2: Unfreeze only last 30 layers (less aggressive)
model = unfreeze_base_model(model, num_layers=30, learning_rate=0.00005)
# history2 = train_transfer_learning_model(...)

# Or unfreeze all layers (most aggressive)
model = unfreeze_base_model(model, num_layers=200, learning_rate=0.00001)
# history3 = train_transfer_learning_model(...)
```

---

### Comparison Script Example

To compare all three models:

```python
import json
from pathlib import Path

results_dir = Path('../results/reports')

models = ['mobilenetv2', 'efficientnetb0', 'resnet50']
metrics_data = {}

for model_name in models:
    with open(results_dir / f'{model_name}_metrics.json', 'r') as f:
        metrics_data[model_name] = json.load(f)

# Print comparison
print("\nModel Comparison on Test Set:")
print("="*60)
print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'Params':<12}")
print("="*60)

for model_name in models:
    metrics = metrics_data[model_name]
    with open(results_dir / f'{model_name}_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f}")

print("="*60)
```

---

## Data Pipeline

### Input Requirements

All transfer learning models expect:

**Input shape:** (224, 224, 3)
**Data type:** uint8 or float32
**Value range:** [0, 255] (uint8) or [0, 1] (float32)

### Preprocessing Pipeline

1. **Load original images** from `../results/preprocessed_data/`
2. **Apply model-specific preprocessing** using `preprocess_images()`
   - MobileNetV2: Normalizes to [-1, 1]
   - EfficientNetB0: Normalizes to [-1, 1]
   - ResNet50: Normalizes to [-1, 1]
3. **Batch training** with specified batch size

### Example Data Shapes

```python
# After loading from preprocessed_data/
X_train.shape  # (num_train_samples, 224, 224, 3)
y_train.shape  # (num_train_samples, num_classes)  [One-hot encoded]

# After preprocessing
X_train_proc.shape  # (num_train_samples, 224, 224, 3)
X_train_proc.dtype  # float32
X_train_proc.min()  # Approximately -1.0
X_train_proc.max()  # Approximately 1.0
```

---

## Backward Compatibility

### Existing Functionality Preserved

✓ **Custom CNN Model** - Still available in `model_utils.py`
✓ **Data Loading** - Same `data_utils.py` functions
✓ **Visualization** - Same `visualization_utils.py` functions
✓ **Dataset Structure** - No changes to folder organization
✓ **Preprocessing Pipeline** - Existing `03_data_preprocessing.ipynb` unchanged
✓ **Evaluation Logic** - Compatible with existing notebooks

### Using Custom CNN

The original custom CNN model is still available:

```python
from utils.model_utils import build_cnn_model

model = build_cnn_model(num_classes=25, input_shape=(224, 224, 3))
# Training proceeds as before
```

---

## Performance Comparison

Expected performance on leaf disease classification (approximate):

| Model | Model Size | Inference Time | Accuracy | Training Time |
|-------|-----------|-----------------|----------|---------------|
| MobileNetV2 | ~14 MB | Fast | ~90-92% | ~15 min |
| EfficientNetB0 | ~29 MB | Medium | ~92-94% | ~20 min |
| ResNet50 | ~98 MB | Medium | ~93-95% | ~25 min |
| Custom CNN | ~50 MB | Slow | ~85-88% | ~30 min |

*Times and accuracies are approximate and depend on dataset size and hardware.*

---

## Troubleshooting

### Issue: "Error: Preprocessed data not found!"

**Solution:** Run `03_data_preprocessing.ipynb` first to generate:
- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`
- `label_encoding.json`

### Issue: Out of memory during training

**Solution:**
1. Reduce batch size: `BATCH_SIZE = 16` (default is 32)
2. Reduce epochs: `EPOCHS_PHASE1 = 10`
3. Use smaller model: Switch to MobileNetV2

### Issue: Poor accuracy

**Solutions:**
1. Increase Phase 2 epochs
2. Reduce learning rate further (0.00005)
3. Unfreeze more layers (100 instead of 50)
4. Check data preprocessing is correct

### Issue: Training takes too long

**Solution:**
1. Increase batch size: `BATCH_SIZE = 64`
2. Reduce epochs
3. Use MobileNetV2 (fastest)
4. Enable GPU acceleration

---

## GPU Support

All models are GPU-compatible. To verify GPU is being used:

```python
import tensorflow as tf

# Check available GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")
for gpu in gpus:
    print(f"  {gpu}")

# Enable memory growth (prevents OOM)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

## Model Export for Deployment

### Save as TensorFlow SavedModel

```python
model.save('models/mobilenetv2/saved_model')
```

### Convert to TensorFlow Lite (Mobile)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Convert to ONNX (Interoperability)

```bash
pip install onnx tf2onnx

python -m tf2onnx.convert --saved-model models/mobilenetv2/saved_model \
    --output model.onnx --target onnxruntime_12.1
```

---

## References

- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [ImageNet Preprocessing](https://github.com/keras-team/keras-applications)

---

## Support and Updates

For issues or improvements, refer to:
- Model documentation in each function's docstring
- Training script output and error messages
- Jupyter notebooks for visualization and debugging

---

**Last Updated:** May 2026
**Version:** 2.0 (Transfer Learning Upgrade)
