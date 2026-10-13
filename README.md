# Crop Leaf Disease Classification using TensorFlow/Keras

A professional deep learning project for multi-crop disease classification using Convolutional Neural Networks (CNNs) and transfer learning with multiple pretrained architectures.

## 📋 Project Overview

This project implements an end-to-end deep learning pipeline for classifying crop leaf diseases across multiple crop types:
- **Crops**: Papaya, Guava, Brinjal, Castor, Cumin
- **Classification**: Healthy vs Unhealthy (specific diseases)
- **Approaches**: 
  - Custom CNN baseline
  - Transfer Learning Models: MobileNetV2, EfficientNetB0, ResNet50
- **Output**: Trained models, TensorFlow Lite export, Grad-CAM visualizations

## ✨ Version 2.0 Update - Transfer Learning Suite

**NEW:** Support for three modern transfer learning architectures:

| Model | Size | Speed | Test Accuracy | F1-Score | Use Case |
|-------|------|-------|---------------|----------|----------|
| **Custom CNN** | checkpoint-dependent | Moderate | 96.40% | 96.46% | Baseline reference |
| **MobileNetV2** | 14 MB | ⚡ Fast | 90.12% | 89.91% | Mobile/Edge devices |
| **EfficientNetB0** | 35.70 MB | 🚀 Medium | 96.98%* | Not reported | Production systems |
| **ResNet50** | checkpoint-dependent | 🔥 Balanced | 96.74%* | Not reported | High accuracy required |

`*` Accuracy is available from a checkpoint metadata or prediction report, but the matching precision, recall, F1-score, and complete per-class report are not currently archived for that exact artifact. Do not treat these values as a complete paper comparison until the same evaluator has been rerun on every final checkpoint.

### Paper-ready metric sources

| Model | Accuracy | Precision | Recall | F1-score | Confusion matrix / detailed evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Custom CNN | 96.40% | 96.67% | 96.40% | 96.46% | `results/plots/cnn_confusion_matrix.png`, `cnn_confusion_matrix_normalized.png` |
| MobileNetV2 | 90.12% | 90.72% | 90.12% | 89.91% | `results/plots/mobilenet_v2_confusion_matrix_full.png`; 40-class report in `results/reports/comprehensive_evaluation_report.json` |
| EfficientNetB0 | 96.98% | Pending | Pending | Pending | Phase-1 matrix only; regenerate phase-2 matrix and classification report |
| ResNet50 | 96.74% | Pending | Pending | Pending | `results/confusion_matrix/resnet50_phase1_cm.npy`, `resnet50_phase2_cm.npy`; prediction and Grad-CAM reports available |

The repository also contains conflicting final-artifact values in `results/reports/model_comparison.csv` (MobileNetV2 80.12%, EfficientNetB0 81.28%). These likely represent different checkpoints or preprocessing conditions. Use one fixed test split, preprocessing pipeline, label order, and evaluator for the definitive publication table. See the detailed evidence and checklist in `notebooks_only_repo/README.md`.

For detailed transfer learning documentation, see [TRANSFER_LEARNING_GUIDE.md](TRANSFER_LEARNING_GUIDE.md)

## 📁 Project Structure

```
cnn_model/
│
├── notebooks/                          # Jupyter notebooks (ordered pipeline)
│   ├── 01_data_check.ipynb            # Dataset validation
│   ├── 02_dataset_visualization.ipynb  # EDA & visualization
│   ├── 03_targeted_augmentation.ipynb  # Target-class data augmentation
│   ├── 04_data_preprocessing.ipynb     # Preprocessing & memory-efficient dataset preparation
│   ├── 05_cnn_baseline.ipynb           # Custom CNN training
│   ├── 06_mobilenet_training.ipynb     # Transfer learning (MobileNetV2)
│   ├── 07_offline_augmentation.ipynb   # Offline dataset augmentations
│   ├── 08_model_evaluation.ipynb       # Model comparison, metrics & confusion matrices
│   ├── 09_resnet50_transfer_learning.ipynb # Transfer learning (ResNet50)
│   ├── 10_efficientnetb0_training.ipynb  # Transfer learning (EfficientNetB0)
│   ├── 11_prediction_testing.ipynb     # Inference & error analysis on test set
│   ├── 12_tflite_conversion.ipynb      # Convert models to TensorFlow Lite format
│   └── 13_gradcam_explainability.ipynb  # Interpretability & explainable AI (Grad-CAM)
│
├── models/                             # Trained models & training scripts
│   ├── train_mobilenetv2.py            # MobileNetV2 training script
│   ├── train_efficientnetb0.py         # EfficientNetB0 training script
│   ├── train_resnet50.py               # ResNet50 training script
│   ├── cnn/                            # Custom CNN models (empty)
│   ├── mobilenet/                      # MobileNetV2 model weights
│   │   ├── mobilenet_v2_final.h5       # Final saved MobileNetV2 model
│   │   └── best_model.h5               # Best saved MobileNetV2 model
│   ├── efficientnetb0/                 # EfficientNetB0 model weights & checkpoints
│   │   ├── efficientnetb0_final.h5     # Final saved EfficientNetB0 model
│   │   └── best_efficientnetb0_phase*_stage*.h5 # Checkpoints
│   ├── resnet50/                       # ResNet50 model weights & checkpoints
│   │   ├── resnet50_final.h5           # Final saved ResNet50 model
│   │   └── resnet50_phase*_best.h5     # Checkpoints
│   └── tflite/                         # Mobile-ready models
│       ├── model_float.tflite          # Floating point TFLite model
│       ├── model_quantized.tflite      # Quantized TFLite model
│       └── metadata.json               # Model metadata
│
├── utils/                              # Utility modules
│   ├── transfer_learning_utils.py      # Transfer learning functions
│   ├── efficient_data_loader.py        # Memory-efficient data loaders (batch loading on-demand)
│   ├── npz_consolidator.py            # NPZ array loading consolidator
│   ├── data_utils.py                   # Data loading & validation
│   ├── model_utils.py                  # Custom CNN & model utilities
│   └── visualization_utils.py          # Plotting & visualization
│
├── results/                            # Results & outputs
│   ├── confusion_matrix/               # Confusion matrix arrays & plots
│   ├── logs/                           # TensorBoard and training logs
│   ├── plots/                          # Generated visualizations and Grad-CAM outputs
│   ├── reports/                        # Evaluation reports, metrics and guides
│   │   ├── mobilenet_history.json      # MobileNetV2 training history
│   │   ├── mobilenet_metrics.json      # MobileNetV2 performance metrics
│   │   ├── model_comparison.csv        # Summary model metrics
│   │   ├── prediction_report.json      # Test predictions summary
│   │   └── explainability_report.json  # Grad-CAM validation report
│   └── preprocessed_data_efficient/    # Processed datasets (memory-efficient NPZ files & split metadata)
│
├── Final_images_phd/                   # Dataset (symlink or actual data)
│   ├── Brinjal/
│   ├── Castor/
│   ├── Cumin/
│   ├── Guava/
│   └── Papaya/
│
├── TRANSFER_LEARNING_GUIDE.md          # ✨ Comprehensive transfer learning guide
├── README.md                           # This file
└── requirements.txt                    # Python dependencies
```

## 🎯 Notebook Execution Order

### Phase 1: Dataset, Visualization, and Preprocessing ⚙️
```
01_data_check.ipynb              # Verify dataset structure and integrity
    ↓
02_dataset_visualization.ipynb    # EDA & visualize sample images
    ↓
03_targeted_augmentation.ipynb   # Target-class augmentations
    ↓
04_data_preprocessing.ipynb      # Create train/val/test splits & save NPZ files
```

**Tasks:**
- Verify dataset structure, sizes, and integrity.
- Perform exploratory data analysis and visualize class distributions.
- Implement target-class augmentations to balance underrepresented categories.
- Resize images to 224×224, normalize, and divide into train/val/test splits (70/15/15) stored memory-efficiently as individual `.npz` files.

---

### Phase 2: Model Training 🤖
```
05_cnn_baseline.ipynb            # Custom CNN baseline training
    ↓
06_mobilenet_training.ipynb      # MobileNetV2 transfer learning
    ↓
07_offline_augmentation.ipynb    # Offline dataset augmentations
    ↓
09_resnet50_transfer_learning.ipynb # ResNet50 transfer learning
    ↓
10_efficientnetb0_training.ipynb  # EfficientNetB0 transfer learning
```

**Tasks:**
- Train custom CNN baseline model.
- Train transfer learning models (MobileNetV2, ResNet50, EfficientNetB0) using a 2-phase approach (frozen base + fine-tuning).
- Use callback routines for early stopping, learning rate schedules, and logging.
- Train with memory-efficient generator-based datasets.

---

### Phase 3: Evaluation, Prediction, and Deployment 📊
```
08_model_evaluation.ipynb        # Compare models and plot metrics
    ↓
11_prediction_testing.ipynb      # Detailed test set predictions & confidence
    ↓
12_tflite_conversion.ipynb       # Convert models to TensorFlow Lite formats
    ↓
13_gradcam_explainability.ipynb  # Interpretability & explainable AI (Grad-CAM)
```

**Tasks:**
- Evaluate test set performance and compile detailed comparison metrics.
- Plot confusion matrices, training histories, and precision/recall stats.
- Run inference tests to evaluate confidence scores and analyze misclassifications.
- Export trained models to floating point and integer-quantized TFLite formats for mobile deployment.
- Generate explainability maps (Grad-CAM) to visualize focus regions.

---

## 🚀 Quick Start - Transfer Learning

### Train Individual Models

After completing Phase 1 (preprocessing), run individual transfer learning models:

**MobileNetV2** (Lightweight, fastest)
```bash
cd models
python train_mobilenetv2.py
```

**EfficientNetB0** (Balanced performance)
```bash
cd models
python train_efficientnetb0.py
```

**ResNet50** (High accuracy)
```bash
cd models
python train_resnet50.py
```

Each script:
1. Loads preprocessed data from `results/preprocessed_data/`
2. Applies model-specific preprocessing
3. Builds and trains the model (2-phase approach)
4. Evaluates on test set
5. Saves model to `models/<model_name>/`
6. Saves metrics to `results/reports/`

### Expected Output

```
======================================================================
MobileNetV2 Transfer Learning Training
======================================================================

Configuration:
  Model: MobileNetV2
  ...

PHASE 1: Training with Frozen Base Layers
✓ Phase 1 completed!

PHASE 2: Fine-tuning with Unfrozen Layers
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

======================================================================
```

---

### Using Transfer Learning in Jupyter Notebooks

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

# Load data
X_train = np.load('../results/preprocessed_data/X_train.npy')
# ... load other data ...

# Preprocess for MobileNetV2
X_train_proc = preprocess_images(X_train, model_type='mobilenetv2')

# Build model
model = build_mobilenetv2_model(num_classes=num_classes)

# Train with two-phase approach
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
metrics = evaluate_transfer_learning_model(
    model=model,
    X_test=X_test_proc,
    y_test=y_test,
    class_names=class_names
)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

For advanced usage and detailed documentation, see [TRANSFER_LEARNING_GUIDE.md](TRANSFER_LEARNING_GUIDE.md)
    ↓
08_tflite_conversion.ipynb
    ↓
09_gradcam_explainability.ipynb
```

**Tasks:**
- Compare CNN vs MobileNet performance
- Evaluate on test set with detailed metrics
- Generate confusion matrices and classification reports
- Test predictions with confidence analysis
- Convert to TensorFlow Lite for mobile
- Generate Grad-CAM visualizations for interpretability

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow tensorflow-lite
pip install numpy pandas scikit-learn matplotlib seaborn
pip install opencv-python pillow
pip install jupyter ipython
```

### Running the Project

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Execute notebooks in order:**
   - Start with `01_data_check.ipynb` to verify dataset
   - Progress through phases as outlined above

3. **Key Outputs:**
   - Trained models in `models/` directory
   - Visualizations in `results/plots/`
   - Metrics & reports in `results/reports/`

---

## 📊 Model Architectures

### CNN Baseline
```
Input (224, 224, 3)
    ↓
Conv2D (32) → BatchNorm → Conv2D (32) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (64) → BatchNorm → Conv2D (64) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (128) → BatchNorm → Conv2D (128) → BatchNorm → MaxPool → Dropout
    ↓
Conv2D (256) → BatchNorm → Conv2D (256) → BatchNorm → MaxPool → Dropout
    ↓
Flatten → Dense(512) → BatchNorm → Dropout(0.5)
    ↓
Dense(256) → BatchNorm → Dropout(0.5)
    ↓
Output (num_classes, softmax)
```

### Transfer Learning Models (MobileNetV2, EfficientNetB0, ResNet50) ✨

All transfer learning models use the same architecture pattern:

```
Input (224, 224, 3)
    ↓
Pretrained Base Model (ImageNet weights)
    ├─ Phase 1: Base FROZEN
    └─ Phase 2: Last 50 layers UNFROZEN for fine-tuning
    ↓
GlobalAveragePooling2D()
    ↓
Dense(128, activation='relu')
    ↓
Dropout(0.3)
    ↓
Output (num_classes, softmax)

Training Strategy:
├─ Phase 1: Train custom head only (20 epochs)
│  └─ Base model frozen, learns task-specific features from pretrained weights
│
└─ Phase 2: Fine-tune last 50 layers (15 epochs)
   └─ Reduce learning rate to 0.0001 to preserve learned features
```

**Key Hyperparameters:**
- Input Shape: 224×224×3
- Optimizer: Adam (LR=0.0001 in Phase 2)
- Loss: Categorical Crossentropy
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## 🎓 Key Features

### Data Preprocessing
- ✅ Image resizing to 224×224
- ✅ Normalization to [0, 1]
- ✅ Data augmentation (rotation, shifts, zoom, flips)
- ✅ Stratified train/val/test splits
- ✅ Corruption detection & handling

### Model Training
- ✅ Custom CNN with batch normalization
- ✅ MobileNetV2 transfer learning
- ✅ Early stopping & learning rate reduction
- ✅ Model checkpointing (save best models)
- ✅ Training history tracking

### Evaluation
- ✅ Confusion matrices
- ✅ Classification reports (precision, recall, F1)
- ✅ Per-class performance analysis
- ✅ ROC curves (per-class)
- ✅ Misclassification analysis

### Deployment
- ✅ TensorFlow Lite export (float & quantized)
- ✅ Model quantization for mobile efficiency
- ✅ Size reduction analysis
- ✅ Accuracy preservation validation

### Explainability
- ✅ Grad-CAM visualizations
- ✅ Attention map generation
- ✅ Per-class heatmaps
- ✅ Misclassification interpretation

---

## 📈 Expected Performance

### Model Comparison

| Model | Model Size | Inference Time | Test Accuracy | Training Time | GPU Memory |
|-------|-----------|-----------------|-------|-------|---------|
| Custom CNN | ~150 MB | Slow | 75-85% | 40-50 min | High |
| MobileNetV2 | 14 MB | ⚡ Fast | 90-92% | 15-20 min | Low |
| **EfficientNetB0** | **29 MB** | **Medium** | **92-94%** | **20-25 min** | **Medium** |
| ResNet50 | 98 MB | Medium | 93-95% | 25-30 min | High |

*Times and accuracies are approximate and depend on dataset size, hardware, and hyperparameters.*

### Performance Benefits

✅ **Transfer Learning Advantages:**
- Faster convergence (15-25 min vs 40-50 min)
- Better accuracy (90%+ vs 75-85%)
- Smaller model size (14-98 MB vs 150 MB)
- Better generalization
- Lower computational requirements

### TensorFlow Lite Conversion

- **Float Model:** ~14-98 MB (99%+ accuracy preserved)
- **Quantized Model:** ~3-25 MB (95%+ accuracy preserved)
- **Inference Time:** 50-150 ms per image (mobile device)

---

## 📝 Generated Reports

### Data Validation
- `data_check_report.json` - Dataset integrity check
- `label_encoding.json` - Class label mapping

### Model Performance
- `model_comparison.csv` - CNN vs MobileNet metrics
- `comprehensive_evaluation_report.json` - Detailed metrics
- `best_model.json` - Best model information

### Predictions
- `prediction_report.json` - Test set prediction statistics
- `per_class_prediction_stats.csv` - Per-class accuracy

### Deployment
- `tflite_conversion_report.json` - TFLite conversion details
- `explainability_report.json` - Grad-CAM analysis

---

## 🎯 Use Cases

### 1. Research & Development
- Establish baseline models for crop disease classification
- Compare CNN vs transfer learning approaches
- Analyze model attention patterns

### 2. Mobile Deployment
- Deploy TFLite quantized model to Android/iOS
- Real-time crop health monitoring in fields
- Lightweight inference on edge devices

### 3. Explainability & Trust
- Generate Grad-CAM heatmaps for interpretability
- Validate model decisions for agricultural experts
- Identify if model focuses on disease symptoms

### 4. Dataset Creation
- Use trained models as weak labels for new data
- Identify difficult examples for annotation
- Bootstrap larger labeled datasets

---

## 🔧 Customization

### Modify Parameters
Edit the configuration sections in each notebook:

```python
# Image preprocessing
IMAGE_SIZE = (224, 224)  # Change input size
BATCH_SIZE = 32          # Adjust batch size

# Training
EPOCHS = 50              # Number of epochs
LEARNING_RATE = 0.001    # Learning rate
TRAINABLE_LAYERS = 50    # Layers to fine-tune

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
```

### Add New Crops
1. Add crop folder to `Final_images_phd/`
2. Organize as: `Crop_name/Healthy_crop_name/` and `Crop_name/Unhealthy_crop_name/Disease_1/`, etc.
3. Re-run notebooks from `01_data_check.ipynb`

### Use Different Models
Replace in `model_utils.py`:
```python
# Alternative: ResNet50, EfficientNet, etc.
base_model = keras.applications.ResNet50(weights='imagenet')
# or
base_model = keras.applications.EfficientNetB0(weights='imagenet')
```

---

## 📚 Dependencies

```
tensorflow >= 2.10
numpy >= 1.21
pandas >= 1.3
scikit-learn >= 1.0
opencv-python >= 4.5
matplotlib >= 3.5
seaborn >= 0.12
pillow >= 8.3
jupyter >= 1.0
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce image size (e.g., 224 → 160)
- Use only phase 1 training (without fine-tuning)

### Slow Training
- Reduce number of `EPOCHS`
- Use early stopping with lower `patience`
- Enable GPU: `tf.config.list_physical_devices('GPU')`

### Poor Model Performance
- Increase augmentation intensity
- Use class weights if dataset is imbalanced
- Collect more training data
- Increase model capacity (more layers/filters)

### TFLite Quantization Loss
- Use float model instead of quantized
- Reduce quantization intensity
- Fine-tune with quantization-aware training

---

## 📖 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02055)
- [Keras API Guide](https://keras.io/)

---

## 👥 Contributing

Contributions are welcome! Please:
1. Create a feature branch
2. Add tests for new functionality
3. Submit a pull request with description

---

## 📄 License

This project is provided for educational and research purposes.

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review notebook comments and docstrings
3. Consult generated reports in `results/reports/`

---

## ✅ Project Status

✅ **Complete**: All 9 notebooks implemented
✅ **Tested**: Utility functions validated
✅ **Documented**: Comprehensive docstrings
✅ **Production-Ready**: TFLite models exported

**Last Updated:** May 9, 2026
**Version:** 1.0

---

## 🎓 Learning Outcomes

After completing this project, you will understand:
- ✅ Complete ML pipeline from data to deployment
- ✅ Custom CNN architecture design
- ✅ Transfer learning with MobileNetV2
- ✅ Data preprocessing and augmentation
- ✅ Model evaluation and comparison
- ✅ TensorFlow Lite conversion for mobile
- ✅ Model interpretability with Grad-CAM
- ✅ Professional code organization

---

**Happy Learning! 🚀**
#   c n n _ m o d e l _ l e a f _ c l a s s i f i c a t i o n - 
 
 