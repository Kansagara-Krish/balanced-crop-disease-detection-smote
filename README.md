# Crop Leaf Disease Classification using TensorFlow / Keras

A reproducible research project for 40-class crop-leaf disease classification using a custom CNN and ImageNet transfer-learning models. This README records the currently available evidence for each model so that results can be reused in a PhD paper without mixing metrics from different runs.

---

## 📌 Project Overview

This repository implements a complete workflow for crop leaf disease detection across multiple crop types.

- **Crops:** Papaya, Guava, Brinjal, Castor, Cumin
- **Task:** Healthy vs Unhealthy leaf disease classification
- **Approaches:** Custom CNN baseline, MobileNetV2, EfficientNetB0, ResNet50
- **Deliverables:** Trained models, evaluation reports, TFLite exports, Grad-CAM visualizations

### Soil Health XGBoost & SHAP Explainability Model

The primary soil-health classification and interpretability pipeline is implemented in [`notebooks/14_soil_xgboost_shap_analysis.ipynb`](notebooks/14_soil_xgboost_shap_analysis.ipynb).

- **Dataset:** 2,073 soil samples across 5 major crops (Castor, Cumin, Guava, Papaya, Brinjal) from [`Final_soil_new_clean_data_updated_modified (1).csv`](Final_soil_new_clean_data_updated_modified%20(1).csv).
- **Target:** `Soil_Status_Binary` (Unhealthy = 0: 1,137 samples; Healthy = 1: 936 samples).
- **Feature Engineering (23 Clean Predictors):**
  - **19 Physicochemical & Spectral Features:** `pH`, `Conductivity`, `Nitrogen`, `Phosphorus`, `Potassium`, `Copper`, `Ferrous`, `Manganese`, `Sulphur`, `Zinc`, `LAI`, `TII`, `ARI`, `RBI`, `CRI`, `FAI`, `MRI`, `HCI`, `C/N_Ratio`.
  - **4 One-Hot Crop Indicators:** `Crop_Name_Castor`, `Crop_Name_Cumin`, `Crop_Name_Guava`, `Crop_Name_Papaya`.
  - **Dropped Non-Predictive / Harmful Features:** Arbitrary tracking IDs (`Sample_Name`, `Farm_Name`), sparse location metadata (`City_Name` with 27 redundant dummies), invariant zero-variance columns (`Boron`, `Organic_Carbon`), and target leakage (`Soil_Health_Score` which has a direct rule relationship with the label).
- **Model Architecture & Serialization:** Optimized `XGBClassifier` serialized to:
  - [`models/xgb_soil_health_best.pkl`](models/xgb_soil_health_best.pkl)
  - [`models/xgb_soil_health_best.json`](models/xgb_soil_health_best.json)
  - [`models/xgb_soil_health_best.joblib`](models/xgb_soil_health_best.joblib)
  - Aligned feature schema: [`models/soil_model_features.json`](models/soil_model_features.json)
- **Tuned Hyperparameters (Stratified 5-Fold Grid Search):**
  - `n_estimators`: 150
  - `max_depth`: 4
  - `learning_rate`: 0.03
  - `min_child_weight`: 1
  - `subsample`: 1.0
  - `colsample_bytree`: 1.0
  - `eval_metric`: `'logloss'`

#### 📊 Model Performance & Accuracy Verification

The model was rigorously validated across two independent 80/20 train/test partitions and full 5-fold cross-validation across all 2,073 samples:

| Evaluation Split | Accuracy | F1-Score (Healthy) | F1-Score (Unhealthy) | Macro F1 | Weighted F1 | Precision (Healthy) | Recall (Healthy) | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Test Cycle 1 (Holdout 20%, Seed 42)** | **90.60%** | **0.8843** | **0.9209** | **0.9026** | **0.9044** | **99.33%** | 79.68% | **0.9321** |
| **Test Cycle 2 (Independent 20%, Seed 101)** | **91.81%** | **0.9012** | **0.9300** | **0.9156** | **0.9170** | **98.73%** | 82.89% | **0.9525** |
| **5-Fold Stratified CV (All 2,073 samples)** | **91.12% ± 1.04%** | — | — | — | **89.19% ± 1.34%** | — | — | **94.27% ± 0.67%** |

##### Confusion Matrix Breakdown (Test Cycle 1 — 415 Samples)
- **True Unhealthy (0):** 227 / 228 correctly identified (**99.56% Recall**, only 1 false positive)
- **True Healthy (1):** 149 / 187 correctly identified (**79.68% Recall**, **99.33% Precision**)
- `Confusion Matrix:` `[[227, 1], [38, 149]]`

---

#### 🔬 Explainable AI (SHAP) Analysis & Agronomic Insights

Using `shap.TreeExplainer` on the 23-feature space in [`notebooks/14_soil_xgboost_shap_analysis.ipynb`](notebooks/14_soil_xgboost_shap_analysis.ipynb), we extracted global feature importance, local sample waterfall attributions, and non-linear agronomic thresholds:

1. **Top Influential Features (Mean Absolute SHAP Value):**
   - **`Conductivity` (3.007):** Overwhelmingly dominant physical predictor of soil health.
   - **`ARI` (Anthocyanin Reflectance Index) (0.149):** Primary spectral vegetation/canopy stress indicator.
   - **`C/N_Ratio` (0.074):** Balance of carbon-to-nitrogen organic matter equilibrium.
   - **`pH` (0.072):** Soil acidity/alkalinity bioavailability controller.
   - **`Ferrous (Fe)` (0.066):** Essential micronutrient availability.
   - **`Nitrogen (N)` (0.058):** Primary macronutrient driver.
   - **`MRI` (0.052), `Phosphorus (P)` (0.046), `LAI` (0.031), `Sulphur (S)` (0.024)**.

2. **Agronomic Non-Linear Thresholds (from Multi-Feature SHAP Dependence Grid):**
   - **Conductivity (Salinity Stress):** Sharp inflection cliff at **~2,200 µS/cm**. Values below 2,000 µS/cm contribute strong positive SHAP (+4.5 log-odds) towards Healthy, whereas values above 2,500 µS/cm receive heavy salinity penalties (-2.0 to -4.0 log-odds).
   - **Potassium (K):** Positive contribution observed when Potassium exceeds **230 kg/ha**, peaking near 245 kg/ha.
   - **Soil pH:** Optimal neutral-to-slightly alkaline range (**7.0 – 7.6**) imparts maximal Healthy classification probability.
   - **Nitrogen & Phosphorus:** Balanced N:P stoichiometry prevents soil exhaustion and stabilizes high-confidence healthy predictions.
   - **Publication Figure:** Multi-feature 8-panel SHAP dependence visualization saved at 300 DPI to `results/fig_shap_dependence_multi_panel.png`.

---

#### Historical Soil Branch & Diagnostic Proxy Notes

The historical exploratory branch is archived in [`notebooks_only_repo/notebooks/XGBOOST_model_soil.ipynb`](notebooks_only_repo/notebooks/XGBOOST_model_soil.ipynb) and [`models/XG_boost_soil_data/soil_xgboost_final.joblib`](models/XG_boost_soil_data/soil_xgboost_final.joblib):

| Track | Features | Accuracy | Balanced accuracy | ROC-AUC | Macro F1 | Research interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Diagnostic Proxy (with raw `Conductivity` rule) | 54 | **99.83%** | **99.84%** | **1.0000** | **99.82%** | Reconstructs the internal dataset labelling rule |
| Independent Research Model (Excl. proxy artifacts) | 53 | 54.09% | 52.34% | 0.5241 | 51.32% | Non-conductive baseline without spectral index feature engineering |
| **New Optimized Clean Model ([Notebook 14](notebooks/14_soil_xgboost_shap_analysis.ipynb))** | **23** | **90.60% – 91.81%** | **89.62%** | **0.9321 – 0.9525** | **90.26% – 91.56%** | **Production-grade generalizable model with SHAP interpretability** |

---

### Precision Fertilizer Recommendation & Crop Recovery Engine (Notebook 15)

The autonomous decision support system for translating diagnostic findings (from Leaf Disease CNNs and Soil Chemical sensors) into actionable chemical prescriptions, soil amendments, dosages, and recovery advisories is implemented in [`notebooks/15_fertilizer_recommendation_xgboost_shap.ipynb`](notebooks/15_fertilizer_recommendation_xgboost_shap.ipynb).

- **Knowledge Base Dataset:** 122 expert agricultural diagnostic rules covering 5 cash crops (**Guava, Papaya, Brinjal, Cumin, Castor**) spanning Biotic and Abiotic conditions from [`Final Fertilizer recovery sheet.xlsx`](Final%20Fertilizer%20recovery%20sheet.xlsx).
- **Target Standardization (9 Mutually Exclusive Intervention Classes):**
  1. `Nitrogenous_Fertilizer (Urea)`
  2. `Phosphatic_Fertilizer (SSP/DAP)`
  3. `Potassic_Fertilizer (MOP/SOP)`
  4. `Micronutrient_Supplement (Zn/Fe/B/S/Mg)`
  5. `Soil_Amendment (Gypsum/Lime)`
  6. `Chemical_Fungicide (Mancozeb/Copper)`
  7. `Targeted_Insecticide (Pest/Vector)`
  8. `Bio_Control_Organic_Care (Trichoderma/FYM)`
  9. `No_Chemical_Fertilizer (Toxicity/Flushing)` — Essential agronomic intervention to cease fertilization upon detecting toxic excess.
- **Smart ML Engineering & Feature Necessity Audit:**
  - **Discarded / Non-Predictive:** `CNN_Label_ID` (arbitrary 0–121 row index memorized by trees).
  - **Discarded from Input (Target Leakage):** `Dosage (gm or ml per liter / plant)` and `Farmer Advisory / Recovery Suggestion` (downstream prescription outputs, not input sensor features).
  - **Retained & Harmonized Predictors:** `Crop`, `Type` (Biotic vs Abiotic), `Param_Clean` (15 standardized physiological domains), `Parameter Level (Low/High)`, and composite stress terms (`Param_x_Level`).
- **Trained Model Artifacts:**
  - [`models/xgb_fertilizer_recommendation_best.pkl`](models/xgb_fertilizer_recommendation_best.pkl)
  - [`models/xgb_fertilizer_recommendation_best.json`](models/xgb_fertilizer_recommendation_best.json)
  - [`models/xgb_fertilizer_recommendation_best.joblib`](models/xgb_fertilizer_recommendation_best.joblib)
  - Feature schema: [`models/fertilizer_model_features.json`](models/fertilizer_model_features.json)
  - Master prescription & advisory database: [`models/fertilizer_prescriptions_lookup.json`](models/fertilizer_prescriptions_lookup.json)

#### 📊 Model Performance & Verification Metrics

| Metric | Result | Interpretation |
| :--- | :---: | :--- |
| **Stratified 5-Fold CV Accuracy** | **96.01%** | Consistent generalization across cross-validation folds |
| **Test Set Accuracy (Holdout 20%)** | **92.62%** | High multiclass classification accuracy (226 / 244 test samples correct) |
| **Weighted F1-Score** | **93.28%** | Support-weighted harmonic mean across all 9 intervention classes |
| **Macro F1-Score** | **91.26%** | Balanced performance across both frequent and rare intervention types |
| **Multi-Class ROC-AUC (OvR)** | **0.9986** | Near-perfect probability calibration and ranking |

#### 🔬 Explainable AI (SHAP) & Publication-Ready Figures (300 DPI)
- **Global Feature Weights:** SHAP TreeExplainer reveals that interaction between parameter domain and deficiency/excess direction (`Param_x_Level`) dominates tree splits, followed by etiology (`Type_Biotic` vs `Type_Abiotic`) and crop species.
- **Publication Figures generated for PhD Research:**
  1. `results/fig_fertilizer_target_distribution.png` — Distribution of target intervention classes and crop stress profiles.
  2. `results/fig_fertilizer_confusion_matrix.png` — Normalized multiclass confusion matrix and One-vs-Rest ROC curves.
  3. `results/fig_fertilizer_shap_global_summary.png` — Top 15 influential agronomic features from global SHAP attribution.
  4. `results/fig_fertilizer_shap_dependence.png` — 6-panel Multi-Class SHAP attribution grid showing physiological stress-to-chemical mappings.
  5. `results/fig_fertilizer_shap_waterfall.png` — Local waterfall plots demonstrating exact decision logic for Nitrogen deficiency (Urea), Salinity (Gypsum), and Toxicity (Ceasing fertilizer).

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
│   ├── 13_gradcam_explainability.ipynb
│   ├── 14_soil_xgboost_shap_analysis.ipynb
│   └── 15_fertilizer_recommendation_xgboost_shap.ipynb
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

### Phase 2: Leaf Disease Model Training

1. `05_cnn_baseline.ipynb` — Custom CNN baseline training
2. `06_mobilenet_training.ipynb` — MobileNetV2 transfer learning
3. `07_offline_augmentation.ipynb` — Offline augmentation experiments
4. `09_resnet50_transfer_learning.ipynb` — ResNet50 transfer learning
5. `10_efficientnetb0_training.ipynb` — EfficientNetB0 transfer learning

### Phase 3: Evaluation and Edge Deployment

1. `08_model_evaluation.ipynb` — Model comparison and metrics
2. `11_prediction_testing.ipynb` — Inference and confidence analysis
3. `12_tflite_conversion.ipynb` — TensorFlow Lite export
4. `13_gradcam_explainability.ipynb` — Grad-CAM interpretability

### Phase 4: Soil Health Modeling & Explainable AI (SHAP)

1. `14_soil_xgboost_shap_analysis.ipynb` — Leakage-free feature curation, Stratified 5-Fold XGBoost hyperparameter tuning, multi-cycle validation, and global & local SHAP interpretability analysis.

### Phase 5: Precision Fertilizer Recommendation & Crop Recovery

1. `15_fertilizer_recommendation_xgboost_shap.ipynb` — Diagnostic etiology categorization, multiclass fertilizer/recovery recommendation XGBoost model, publication-quality 300 DPI figures, multi-class SHAP attribution, and production inference lookup pipeline.

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
