# Crop Leaf Disease Classification using TensorFlow / Keras

A reproducible research project for 40-class crop-leaf disease classification using a custom CNN and ImageNet transfer-learning models. This README records the currently available evidence for each model so that results can be reused in a PhD paper without mixing metrics from different runs.

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

## 📈 Verified Results and Evidence

The common held-out test set contains **860 images and 40 classes**. Aggregate metrics below are percentages. The metric source is shown explicitly because the repository contains several checkpoints and historical experiments.

| Model / artifact | Accuracy | Precision | Recall | F1-score | Evidence status |
| --- | ---: | ---: | ---: | ---: | --- |
| Custom CNN baseline report | 96.40% | 96.67% | 96.40% | 96.46% | Complete aggregate report |
| MobileNetV2 detailed evaluation | 90.12% | 90.72% | 90.12% | 89.91% | Complete aggregate and per-class report |
| EfficientNetB0 metadata checkpoint | 96.98% | Not reported | Not reported | Not reported | Accuracy only; regenerate full evaluation before publication |
| ResNet50 prediction evaluation | 96.74% | Not reported | Not reported | Not reported | Accuracy and prediction analysis; regenerate full classification report before publication |

The older `results/reports/model_comparison.csv` contains different final-artifact values (MobileNetV2 80.12% and EfficientNetB0 81.28%). These must not be merged with the detailed evaluations: they appear to come from different checkpoints or preprocessing conditions. Resolve this discrepancy by rerunning the same evaluator on each final checkpoint before making a definitive paper table.

### Model-by-model evidence

#### Custom CNN baseline

- **Purpose:** non-pretrained baseline for measuring the benefit of transfer learning.
- **Reported test metrics:** accuracy 0.963953, macro precision 0.966697, macro recall 0.963953, macro F1 0.964581.
- **Training evidence:** `results/reports/cnn_baseline_history.json`, `custom_cnn_part1_history.json`, and `custom_cnn_part2_history.json`.
- **Confusion matrices:** `results/plots/cnn_confusion_matrix.png` and `cnn_confusion_matrix_normalized.png`.
- **Metric source:** `results/reports/cnn_baseline_metrics.json`.
- **Available checkpoints:** `models/cnn/cnn_baseline.h5`, `best_model.h5`, `best_model.keras`, and `custom_cnn_final_valacc_0.9744.h5`.

#### MobileNetV2

- **Purpose:** lightweight model for mobile and edge inference.
- **Reported test metrics:** accuracy 0.901163, macro precision 0.907166, macro recall 0.901163, macro F1 0.899090.
- **Per-class evidence:** the detailed classification report contains precision, recall, F1-score, and support for all 40 classes.
- **Confusion matrices:** `results/plots/mobilenet_confusion_matrix.png`, `mobilenet_confusion_matrix_normalized.png`, and `mobilenet_v2_confusion_matrix_full.png`.
- **Metric source:** `results/reports/mobilenet_metrics.json` and the MobileNet section of `comprehensive_evaluation_report.json`.
- **Training evidence:** `results/reports/mobilenet_history.json`.
- **Caution:** `models/mobilenetv2/` is empty; the available MobileNetV2 files are under `models/mobilenet/`. Preserve the exact checkpoint path in any paper or release package.

#### EfficientNetB0

- **Purpose:** accuracy-efficiency trade-off for production deployment.
- **Available result:** metadata reports final accuracy 0.969767 (96.98%), with 4,420,683 total parameters and input size 256x256x3.
- **Training configuration in metadata:** 25 frozen-base epochs, 20 fine-tuning epochs, batch size 8.
- **Evidence:** `models/efficientnetb0/efficientnetb0_metadata.json`, phase history plots, and `results/confusion_matrix/efficientnetb0_phase1_cm.npy`.
- **Missing for a complete paper table:** precision, recall, F1-score, a phase-2 confusion matrix, and a per-class classification report. Do not infer these from accuracy.
- **Deployment evidence:** `results/reports/tflite_conversion_report.json` reports a 35.70 MB original model, 16.71 MB float TFLite model, and 4.71 MB quantized TFLite model. The reported TFLite accuracy values are a separate conversion evaluation and must be labelled as such.

#### ResNet50

- **Purpose:** high-capacity accuracy reference and explainability model.
- **Reported prediction evaluation:** accuracy 0.967442, 832 correct predictions, 28 incorrect predictions, mean confidence 0.982082, and 22 predictions below 0.70 confidence.
- **Confusion matrices:** `results/confusion_matrix/resnet50_phase1_cm.npy`, `resnet50_phase1_cm.png`, and `resnet50_phase2_cm.npy`.
- **Explainability evidence:** Grad-CAM uses `conv5_block3_3_conv`; see `results/reports/explainability_report.json` and `results/plots/gradcam_*.png`.
- **Metric source:** `results/reports/prediction_report.json` and `explainability_report.json`.
- **Conflict requiring rerun:** `models/resnet50/resnet50_metadata.json` records a separate 2.44% evaluation. Because this contradicts the 96.74% prediction report, the checkpoint, preprocessing, label order, and split must be verified before publication.

### Required paper reporting checklist

For every final model, rerun one identical evaluator and archive: accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1/support, the raw 40x40 confusion matrix, normalized confusion matrix, test loss, parameter count, model size, inference time, random seed, preprocessing, checkpoint hash, and the exact test split. Report confidence intervals or repeated-seed variance where possible. The existing notebooks generate much of this material, especially `notebooks/08_model_evaluation.ipynb`, `11_prediction_testing.ipynb`, and `13_gradcam_explainability.ipynb`.

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
