# Crop Leaf Disease Classification using TensorFlow / Keras

A reproducible research project for 40-class crop-leaf disease classification using a custom CNN and ImageNet transfer-learning models. This README records the currently available evidence for each model so that results can be reused in a PhD paper without mixing metrics from different runs.

---

## 📌 Project Overview

This repository implements a complete workflow for crop leaf disease detection across multiple crop types.

- **Crops:** Papaya, Guava, Brinjal, Castor, Cumin
- **Task:** Healthy vs Unhealthy leaf disease classification
- **Approaches:** Custom CNN baseline, MobileNetV2, EfficientNetB0, ResNet50
- **Deliverables:** Trained models, evaluation reports, TFLite exports, Grad-CAM visualizations

### Soil XGBoost Branch

The soil-health classification branch is maintained in [`notebooks_only_repo/notebooks/XGBOOST_model_soil.ipynb`](notebooks_only_repo/notebooks/XGBOOST_model_soil.ipynb). The notebook retains the original training and tuning cells for reproducibility, but its final cell evaluates the existing saved model without retraining. The saved artifact is [`models/XG_boost_soil_data/soil_xgboost_final.joblib`](models/XG_boost_soil_data/soil_xgboost_final.joblib).

The final evaluation cell:

- loads the existing model when the in-memory model is unavailable;
- uses the saved 63-feature schema and encoders;
- evaluates the group-aware test split;
- displays accuracy, ROC-AUC, macro and weighted F1, the classification report, confusion matrix, ROC curve, and top feature importances;
- writes `final_test_metrics.csv`, `soil_predictions_for_fusion.csv`, `04_confusion_matrix.png`, `05_roc_curve.png`, and `03_xgboost_feature_importance.png` under `results/soil_xgboost/paper_figures/`.

No model fitting or hyperparameter search is performed by the final evaluation cell.

#### Verified Soil Model Results

Metrics below were independently recomputed from [`soil_predictions_for_fusion.csv`](soil_predictions_for_fusion.csv), which contains the saved model's 368 prediction records.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Test samples | 368 | Unique prediction records |
| Accuracy | **96.47%** | 355 correct predictions |
| ROC-AUC | **98.97%** | Excellent probability ranking |
| Precision, macro | **90.20%** | Average precision across both classes |
| Recall, macro | **86.76%** | Average recall across both classes |
| F1-score, macro | **88.38%** | Balanced class-level F1 |
| F1-score, weighted | **96.39%** | Support-weighted overall F1 |

#### Evaluation Integrity

| Check | Result |
| --- | --- |
| Confusion matrix | `[[24, 8], [5, 331]]` |
| Actual class distribution | 32 class-0, 336 class-1 samples |
| Duplicate sample IDs | **0** |
| Probability range | 0.0260 to 0.9990 |
| Label/probability consistency | All predictions agree with a 0.5 threshold |
| Crop coverage | Corn 96, Soybean 105, Tomato 102, Wheat 65 |

The test export is internally consistent and suitable for reporting these metrics. However, the CSV alone cannot prove that the samples were excluded from training; that requires the original dataset and split metadata.

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
| MobileNetV2 detailed evaluation | 90.12% | 90.72% | 90.12% | 89.91% | Complete weighted aggregate and per-class report |
| EfficientNetB0 final checkpoint | 96.86% | 97.10% | 96.86% | 96.86% | Complete weighted aggregate and per-class report |
| ResNet50 phase-2 training evaluation | 96.74% | 96.62% | 96.65% | 96.49% | Complete macro aggregate; weighted F1 96.78% |

The older `results/reports/model_comparison.csv` contains different final-artifact values (MobileNetV2 80.12% and EfficientNetB0 81.28%). These must not be merged with the detailed evaluations: they appear to come from different checkpoints or preprocessing conditions. Resolve this discrepancy by rerunning the same evaluator on each final checkpoint before making a definitive paper table.

### Model-by-model evidence

#### Custom CNN baseline

- **Purpose:** non-pretrained baseline for measuring the benefit of transfer learning.
- **Reported test metrics:** accuracy 0.963953, weighted precision 0.966697, weighted recall 0.963953, weighted F1 0.964581.
- **Training evidence:** `results/reports/cnn_baseline_history.json`, `custom_cnn_part1_history.json`, and `custom_cnn_part2_history.json`.
- **Confusion matrices:** `results/plots/cnn_confusion_matrix.png` and `cnn_confusion_matrix_normalized.png`.
- **Metric source:** `results/reports/cnn_baseline_metrics.json`.
- **Available checkpoints:** `models/cnn/cnn_baseline.h5`, `best_model.h5`, `best_model.keras`, and `custom_cnn_final_valacc_0.9744.h5`.

#### MobileNetV2

- **Purpose:** lightweight model for mobile and edge inference.
- **Reported test metrics:** accuracy 0.901163, weighted precision 0.907166, weighted recall 0.901163, weighted F1 0.899090.
- **Per-class evidence:** the detailed classification report contains precision, recall, F1-score, and support for all 40 classes.
- **Confusion matrices:** `results/plots/mobilenet_confusion_matrix.png`, `mobilenet_confusion_matrix_normalized.png`, and `mobilenet_v2_confusion_matrix_full.png`.
- **Metric source:** `results/reports/mobilenet_metrics.json` and the MobileNet section of `comprehensive_evaluation_report.json`.
- **Training evidence:** `results/reports/mobilenet_history.json`.
- **Aggregation method:** the notebook calculates all four aggregate metrics with `average='weighted'`; these are not macro averages.
- **Stale-artifact warning:** the current `results/reports/mobilenet_metrics.json` contains a separate failed run at approximately 2.33% accuracy. Use the 90.12% result from the MobileNet evaluation notebook and `comprehensive_evaluation_report.json` only after confirming the checkpoint and test split.
- **Caution:** `models/mobilenetv2/` is empty; the available MobileNetV2 files are under `models/mobilenet/`. Preserve the exact checkpoint path in any paper or release package.

#### EfficientNetB0

- **Purpose:** accuracy-efficiency trade-off for production deployment.
- **Verified final-checkpoint metrics:** accuracy 0.968605 (96.86%), weighted precision 0.970987 (97.10%), weighted recall 0.968605 (96.86%), and weighted F1 0.968578 (96.86%). Macro precision is 0.969413 (96.94%), macro recall is 0.968915 (96.89%), and macro F1 is 0.967927 (96.79%).
- **Model details:** 4,420,683 total parameters, input size 256x256x3, and 860 test samples.
- **Training configuration in metadata:** 25 frozen-base epochs, 20 fine-tuning epochs, batch size 8.
- **Evidence:** `models/efficientnetb0/efficientnetb0_metadata.json`, `results/reports/efficientnetb0_metrics_complete.json`, phase history plots, and `results/confusion_matrix/efficientnetb0_phase2_cm.npy`.
- **Complete evaluation output:** `results/reports/efficientnetb0_metrics_complete.json` records accuracy, weighted and macro precision/recall/F1, the per-class classification report, and the raw confusion matrix for `models/efficientnetb0/efficientnetb0_final.h5`.
- **Deployment evidence:** `results/reports/tflite_conversion_report.json` reports a 35.70 MB original model, 16.71 MB float TFLite model, and 4.71 MB quantized TFLite model. The reported TFLite accuracy values are a separate conversion evaluation and must be labelled as such.

### Mobile and Edge Deployment Models

The project includes TensorFlow Lite exports for running inference on Android, embedded Linux, Raspberry Pi, and other resource-constrained devices. These are deployment conversions, not new independently trained architectures.

| Mobile artifact | Format | Size | Reported accuracy | Size reduction | Recommended use |
| --- | --- | ---: | ---: | ---: | --- |
| `models/tflite/model_float.tflite` | Float32 TFLite | 16.71 MB | 88% | 53.20% vs original | Devices where accuracy is prioritized and storage is available |
| `models/tflite/model_quantized.tflite` | Quantized TFLite | 4.71 MB | 90% | 86.81% vs original | Default mobile and edge deployment candidate |

- **Source model:** EfficientNetB0 at `models/efficientnetb0/efficientnetb0_final.h5`.
- **Conversion report:** `results/reports/tflite_conversion_report.json`.
- **Metadata:** `models/tflite/metadata.json`, when included with the exported model package.
- **Deployment recommendation:** use `model_quantized.tflite` for the smallest footprint; validate latency and accuracy on the target device before release.
- **Important evaluation note:** the 88% and 90% values come from the TFLite conversion evaluation. They must not be compared directly with the 96.98% EfficientNetB0 training/checkpoint metadata value unless the same test images, preprocessing, labels, and evaluation code are used.
- **Runtime requirement:** the mobile application must use the same input resolution, channel order, scaling, class-index mapping, and output label list used during training. A wrong preprocessing range can make a strong model appear close to random performance.

Example Android/Python deployment flow:

1. Load the `.tflite` file with TensorFlow Lite or the platform TFLite runtime.
2. Resize each image to the model input shape and apply the recorded preprocessing.
3. Run inference and map the output index to the saved 40-class label list.
4. Display the predicted class and confidence, while applying a confidence threshold for an `uncertain` result.
5. Measure cold-start time, average inference latency, peak memory, battery impact, and accuracy on the target hardware.

#### ResNet50

- **Purpose:** high-capacity accuracy reference and explainability model.
- **Verified phase-2 training metrics:** accuracy 0.9674 (96.74%), macro precision 0.9662 (96.62%), macro recall 0.9665 (96.65%), macro F1 0.9649 (96.49%), and weighted F1 0.9678 (96.78%).
- **Phase-2 comparison:** phase 1 accuracy was 0.8605, while phase 2 improved by 0.1070 (+12.4%). Macro precision improved from 0.8782 to 0.9662, macro recall from 0.8676 to 0.9665, macro F1 from 0.8649 to 0.9649, and weighted F1 from 0.8598 to 0.9678.
- **Primary metric source:** the phase-2 output in `09_resnet50_transfer_learning.ipynb`, generated by `evaluate_and_report_metrics` on the phase-2 model and the 860-image test split.
- **Phase-2 evidence:** `results/logs/resnet50_phase2_class_metrics.json` and `results/confusion_matrix/resnet50_phase2_cm.npy`.
- **Saved-checkpoint evidence:** `results/reports/resnet50_metrics_complete.json` contains a separate evaluation of the current `resnet50_final.h5` artifact.
- **Historical artifacts:** `results/confusion_matrix/resnet50_phase1_cm.npy`, `resnet50_phase1_cm.png`, and `resnet50_phase2_cm.npy` belong to earlier runs.
- **Explainability evidence:** Grad-CAM uses `conv5_block3_3_conv`; see `results/reports/explainability_report.json` and `results/plots/gradcam_*.png`.
- **Artifact warning:** the current saved `resnet50_final.h5` evaluation reports 2.44%, while the phase-2 in-memory training evaluation reports 96.74%. Use the phase-2 values for the training-results table, but regenerate and verify the saved checkpoint before claiming deployment or release reproducibility.

### Required paper reporting checklist

For every final model, rerun one identical evaluator and archive: accuracy, macro and weighted precision/recall/F1, per-class precision/recall/F1/support, the raw 40x40 confusion matrix, normalized confusion matrix, test loss, parameter count, model size, inference time, random seed, preprocessing, checkpoint hash, and the exact test split. Always label whether reported aggregate values are macro or weighted averages. Report confidence intervals or repeated-seed variance where possible. The existing notebooks generate much of this material, especially `notebooks/08_model_evaluation.ipynb`, `11_prediction_testing.ipynb`, and `13_gradcam_explainability.ipynb`.

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

Mobile deployment artifacts include the Float32 and quantized TensorFlow Lite models, their conversion report, and model metadata. Keep these files together when packaging the mobile application.

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
