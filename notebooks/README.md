# Crop Leaf Disease Classification using TensorFlow / Keras

A professional deep learning project for multi-crop leaf disease classification using CNNs and transfer learning.

---

## 📌 Project Overview

This repository implements a complete workflow for crop leaf disease detection across multiple crop types.

- **Crops:** Papaya, Guava, Brinjal, Castor, Cumin
- **Task:** Healthy vs Unhealthy leaf disease classification
- **Approaches:** Custom CNN baseline, MobileNetV2, EfficientNetB0, ResNet50
- **Deliverables:** Trained models, evaluation reports, TFLite exports, Grad-CAM visualizations

---

## 🌟 Highlights

- Modern transfer learning with MobileNetV2, EfficientNetB0, and ResNet50
- Full notebook pipeline for dataset validation, model training, evaluation, and deployment
- Mobile-ready model export using TensorFlow Lite
- Interpretability via Grad-CAM explainability
- Professional documentation for stakeholder review

---

## 📈 Model Summary

| Model | Approx. Size | Speed | Test Accuracy | Use Case |
| --- | --- | --- | --- | --- |
| Custom CNN | Baseline | Moderate | ~12.3% | Baseline reference |
| MobileNetV2 | 14 MB | Fast | 90.12% | Mobile / edge deployment |
| EfficientNetB0 | 29 MB | Balanced | 96.98% | Production-ready |
| ResNet50 | 98 MB | High accuracy | 96.74% | Accuracy-critical applications |

For more details, see [TRANSFER_LEARNING_GUIDE.md](TRANSFER_LEARNING_GUIDE.md).

---

## 📁 Repository Structure

```text
cnn_model/
├── notebooks/                        # Ordered Jupyter notebooks
│   ├── 01_data_check.ipynb
│   ├── 02_dataset_visualization.ipynb
│   ├── 03_targeted_augmentation.ipynb
│   ├── 04_data_preprocessing.ipynb
│   ├── 05_cnn_baseline.ipynb
│   ├── 06_mobilenet_training.ipynb
│   ├── 07_offline_augmentation.ipynb
│   ├── 08_model_evaluation.ipynb
│   ├── 09_resnet50_transfer_learning.ipynb
│   ├── 10_efficientnetb0_training.ipynb
│   ├── 11_prediction_testing.ipynb
│   ├── 12_tflite_conversion.ipynb
│   └── 13_gradcam_explainability.ipynb
├── models/                           # Training scripts and saved model weights
├── utils/                            # Reusable utilities
├── results/                          # Outputs, plots, and reports
├── TRANSFER_LEARNING_GUIDE.md        # Transfer learning documentation
├── README.md                         # Project overview and instructions
└── requirements.txt                  # Python dependencies
```

---

## 🧠 Workflow Overview

### Phase 1: Data Preparation

1. `01_data_check.ipynb` — Dataset validation and integrity checks
2. `02_dataset_visualization.ipynb` — Exploratory data analysis and visualization
3. `03_targeted_augmentation.ipynb` — Targeted augmentation for class balance
4. `04_data_preprocessing.ipynb` — Preprocessing, resizing, normalization, and split creation

### Phase 2: Model Training

1. `05_cnn_baseline.ipynb` — Custom CNN baseline training
2. `06_mobilenet_training.ipynb` — MobileNetV2 transfer learning
3. `07_offline_augmentation.ipynb` — Offline augmentation experiments
4. `09_resnet50_transfer_learning.ipynb` — ResNet50 transfer learning
5. `10_efficientnetb0_training.ipynb` — EfficientNetB0 transfer learning

### Phase 3: Evaluation and Deployment

1. `08_model_evaluation.ipynb` — Model comparison and metrics
2. `11_prediction_testing.ipynb` — Inference and confidence analysis
3. `12_tflite_conversion.ipynb` — TensorFlow Lite export
4. `13_gradcam_explainability.ipynb` — Grad-CAM interpretability

---

## 🚀 Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook pipeline

```bash
jupyter notebook
```

Execute notebooks in the order listed above.

### Train models directly

```bash
cd models
python train_mobilenetv2.py
python train_efficientnetb0.py
python train_resnet50.py
```

---

## 🧪 Expected Outputs

- Trained model artifacts in `models/`
- Evaluation metrics in `results/reports/`
- Visualization plots in `results/plots/`
- TFLite models in `models/tflite/`
- Grad-CAM explainability outputs in notebooks and reports

---

## ✅ Why this project is professional

- Structured notebook workflow for reproducible research
- Clear reporting and evaluation metrics
- Transfer learning with production-ready architectures
- Mobile deployment via TensorFlow Lite
- Explainable AI using Grad-CAM

---

## 🔍 Files to review first

- `README.md` — Project summary and usage
- `TRANSFER_LEARNING_GUIDE.md` — Detailed architecture and training guidance
- `notebooks/` — Full development and experimentation pipeline
- `models/` — Saved model weights and training scripts
- `results/` — Analysis outputs and reports

---

## 📌 Notes

This repository is designed for stakeholder review, academic presentation, and deployment-ready demonstration for multi-crop leaf disease classification.
