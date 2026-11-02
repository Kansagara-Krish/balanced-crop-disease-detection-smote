# Decision-Level Integration Methodology for Explainable Multimodal Agricultural Decision-Support System

**PhD Research Documentation**  
**Author:** Agricultural AI Research Initiative  
**Crops Supported:** Cumin, Castor, Papaya, Guava, and Brinjal  
**Repository Integration:** `d:\cnn_model`

---

## 📌 1. Executive Summary & Research Objective

This research develops an **explainable, multimodal agricultural decision-support system (MADSS)** for five cash and staple crops: **Cumin, Castor, Papaya, Guava, and Brinjal**. 

The system bridges computer vision (leaf pathology), tabular predictive modeling (soil physicochemical fertility and salinity stress), and agronomic knowledge representation into a unified, transparent decision-support system.

Crucially, rather than retraining existing models or forcibly concatenating disparate image-soil pairs into fragile early-fusion networks, this research implements a rigorous **Decision-Level Integration Architecture**. Each specialized modality retains its independently validated weights, evaluation evidence, and explainable AI (XAI) attributions, while a transparent, rule-based decision engine enforces validated agronomic science and safety guardrails.

---

## 🔬 2. Existing Trained Models Audit & Specification

The repository contains three primary machine learning models, each independently developed, cross-validated, and serialized:

```
                            ┌──────────────────────────────────────────────┐
                            │               Multimodal Inputs              │
                            │  (Leaf Image + Crop ID + Soil Sensor Values) │
                            └──────────────────────┬───────────────────────┘
                                                   │
                ┌──────────────────────────────────┼──────────────────────────────────┐
                │                                  │                                  │
                ▼                                  ▼                                  ▼
    ┌───────────────────────┐          ┌───────────────────────┐          ┌───────────────────────┐
    │     Model 1: CNN      │          │  Model 2: Soil Health │          │   Model 3: Fertilizer │
    │   Leaf Disease Vision │          │    XGBoost Classifier │          │    XGBoost Classifier │
    ├───────────────────────┤          ├───────────────────────┤          ├───────────────────────┤
    │ • 40 Disease Classes  │          │ • Target: Soil Health │          │ • Target: 9 Standard  │
    │ • 256x256 / 224x224   │          │   Binary (0 / 1)      │          │   Interventions       │
    │ • Normalized [0, 1]   │          │ • 23 Clean Predictors │          │ • 55 One-Hot Features │
    │ • Grad-CAM XAI Map    │          │ • SHAP TreeExplainer  │          │ • Multi-Class SHAP    │
    └───────────┬───────────┘          └───────────┬───────────┘          └───────────┬───────────┘
                │                                  │                                  │
                │ Leaf Class + Conf                │ Soil Status + SHAP               │ Intervention Class
                └──────────────────┐               │               ┌──────────────────┘
                                   │               │               │
                                   ▼               ▼               ▼
                        ┌─────────────────────────────────────────────────────┐
                        │          Decision-Level Integration Engine          │
                        │             (recommendation_engine.py)              │
                        ├─────────────────────────────────────────────────────┤
                        │ • Sample Association & Pairing Guardrails           │
                        │ • Crop Mismatch & Low-Confidence Checks             │
                        │ • Biotic vs Abiotic Separation                      │
                        │ • Soil Chemical Benchmark Evaluation                │
                        │ • Conflict Resolution (Lockout / Toxicity / Flush)  │
                        │ • Validated Agronomic Prescriptions (KB CSV)        │
                        └──────────────────────────┬──────────────────────────┘
                                                   │
                                                   ▼
                        ┌─────────────────────────────────────────────────────┐
                        │      Actionable Prescriptions & XAI Advisories      │
                        │  (Foliar Protection + Fertilizer + Soil Amendments) │
                        └─────────────────────────────────────────────────────┘
```

### 2.1 Model 1: CNN Leaf Disease Classification
- **Target:** 40 multi-crop classes (10 Brinjal, 7 Castor, 5 Cumin, 9 Guava, 9 Papaya) distinguishing healthy foliage, biotic pathogens (fungal, bacterial, viral, insects), and visual foliar expressions of nutrient deficiency (e.g., `Brinjal_Nitrogen`, `Guava_phosphorus`, `Papaya_magnesiuam`).
- **Architectures & Checkpoints:**
  - `models/efficientnetb0/efficientnetb0_final_savedmodel` (Accuracy: **96.86%**, Input: 256×256×3).
  - `models/resnet50/resnet50_final_savedmodel` (Phase-2 Accuracy: **96.74%**, Input: 224×224×3).
  - `models/cnn/cnn_baseline.h5` (Custom Deep ResCNN Baseline, Accuracy: **96.40%**, Input: 224×224×3).
  - `models/tflite/model_quantized.tflite` (Edge deployment candidate, 4.71 MB, Accuracy: **90%**).
- **Preprocessing:** Resize input image to target resolution, normalize float32 pixels to `[0.0, 1.0]`.
- **Explainability:** Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual heatmaps highlighting exact foliar lesion structures and necrotic margins.

### 2.2 Model 2: XGBoost Soil Health Classifier (Notebook 14)
- **Target:** `Soil_Status_Binary` (0 = Unhealthy, 1137 samples; 1 = Healthy, 936 samples).
- **Dataset:** 2,073 soil samples from `Final_soil_new_clean_data_updated_modified (1).csv`.
- **Feature Space (23 Clean Predictors):**
  - *19 Physicochemical & Spectral Features:* `pH`, `Conductivity`, `Nitrogen`, `Phosphorus`, `Potassium`, `Copper`, `Ferrous`, `Manganese`, `Sulphur`, `Zinc`, `LAI`, `TII`, `ARI`, `RBI`, `CRI`, `FAI`, `MRI`, `HCI`, `C/N_Ratio`.
  - *4 One-Hot Crop Indicators:* `Crop_Name_Castor`, `Crop_Name_Cumin`, `Crop_Name_Guava`, `Crop_Name_Papaya` (all zeros encodes baseline reference `Brinjal`).
  - *Dropped Non-Predictive & Harmful Features:* `Sample_Name`, `Farm_Name` (arbitrary tracking IDs that cause overfitting), `City_Name` (27 sparse redundant columns), `Boron`, `Organic_Carbon` (zero variance), and `Soil_Health_Score` (target leakage).
- **Serialized Artifacts:** `models/xgb_soil_health_best.joblib`, `models/xgb_soil_health_best.json`, schema in `models/soil_model_features.json`.
- **Performance:** Test Accuracy **90.60% – 91.81%**, Weighted F1 **0.9044 – 0.9170**, Stratified 5-Fold CV **91.12% ± 1.04%**, ROC-AUC **0.9321 – 0.9525**.
- **Explainability:** SHAP `TreeExplainer` isolates `Conductivity` as the dominant predictor (mean |SHAP| = 3.01) with a sharp non-linear salinity cliff at **~2,200 µS/cm**.

### 2.3 Model 3: Precision Fertilizer Recommendation Engine (Notebook 15)
- **Target:** 9 standardized agronomic intervention categories:
  1. `Nitrogenous_Fertilizer (Urea)`
  2. `Phosphatic_Fertilizer (SSP/DAP)`
  3. `Potassic_Fertilizer (MOP/SOP)`
  4. `Micronutrient_Supplement (Zn/Fe/B/S/Mg)`
  5. `Soil_Amendment (Gypsum/Lime)`
  6. `Chemical_Fungicide (Mancozeb/Copper)`
  7. `Targeted_Insecticide (Pest/Vector)`
  8. `Bio_Control_Organic_Care (Trichoderma/FYM)`
  9. `No_Chemical_Fertilizer (Toxicity/Flushing)` — Enforces the mandatory cessation of fertilizer when toxic excess or salinity lockout is detected.
- **Dataset:** 122 expert agricultural diagnostic rules from `Final Fertilizer recovery sheet.xlsx`.
- **Feature Space (55 Aligned One-Hot Features):** Encodes `Crop`, `Type` (Biotic vs Abiotic), `Param_Clean` (17 standardized physiological domains), `Parameter Level (Low/High)`, and interaction terms `Param_x_Level`.
- **Serialized Artifacts:** `models/xgb_fertilizer_recommendation_best.joblib`, `models/xgb_fertilizer_recommendation_best.json`, schema in `models/fertilizer_model_features.json`, labels in `models/fertilizer_label_mapping.json`, lookup DB in `models/fertilizer_prescriptions_lookup.json`.
- **Performance:** Stratified 5-Fold CV Accuracy **96.01%**, Holdout Test Accuracy **92.62%**, Multi-Class ROC-AUC (OvR) **0.9986**.

---

## 📐 3. Decision-Level Integration Methodology

### 3.1 Why Decision-Level Integration?
Early fusion (concatenating raw pixels with soil tabular numbers) suffers from severe flaws in agricultural research:
1. **Spatiotemporal Decoupling:** Soil chemistry changes slowly over months across an orchard or field, whereas foliar pathogen lesions emerge dynamically within days.
2. **Data Hallucination Risk:** Randomly pairing leaf photographs with soil samples from different plants or farms produces spurious correlations that collapse in field validation.
3. **Loss of Explainability:** End-to-end black boxes cannot provide transparent, auditable justification required by agronomists and farmers.

Decision-level integration allows each specialized AI model to operate at its highest accuracy within its domain, after which their outputs are synthesized through transparent agronomic rules.

### 3.2 The Validated Agronomic Knowledge Base (`fertilizer_knowledge_base.csv`)
Exported directly from validated research trials, `fertilizer_knowledge_base.csv` links:
- `Crop`: Guava, Papaya, Brinjal, Cumin, Castor.
- `Leaf Disease / Condition`: Target pathological or nutritional symptom.
- `Type`: Biotic (infections) vs Abiotic (nutritional/environmental).
- `Affected Parameter`: Physiological parameter or causal agent.
- `Parameter Level`: 'Low', 'High', 'Mod'.
- `Value`: Validated scientific thresholds (e.g., Soil N `< 250 mg/kg`, Salinity `EC > 2000 µS/cm`, pH `< 5.5`, etc.).
- `Recommended Fertilizer / Pesticide`: Specific formulation (e.g., Urea, Single Super Phosphate, MOP, Mancozeb, Copper Oxychloride, Agricultural Gypsum).
- `Dosage (gm or ml per liter / plant)`: Agronomically validated rates.
- `Farmer Advisory / Recovery Suggestion`: Practical field execution steps.

---

## 🛡️ 4. Research Constraints & Conflict Resolution Rules

### Rule 1: Sample Pairing & Association Guardrail
- **Constraint:** Do not assume leaf images and soil records belong to the same plant without verified identifiers.
- **Implementation:** The system requires an explicit `is_paired_sample` flag. If unchecked or unverified, the engine issues an `UNVERIFIED_SAMPLE_PAIRING` safety advisory, evaluating foliar health and soil fertility independently to prevent invalid causal coupling.

### Rule 2: Soil-Health != Nutrient-Deficiency
- **Constraint:** Do not treat soil-health classification as nutrient-deficiency classification.
- **Implementation:** Model 2 classifies overall physical/fertility status (primarily governed by salinity and soil balance). Quantitative soil readings (N, P, K, Zn, Fe, pH, EC) are evaluated separately against crop-specific agronomic benchmarks to identify exact element-level deficiencies or toxicities.

### Rule 3: Biotic Pathogen Segregation
- **Constraint:** Do not generate fertilizer recommendations solely from disease predictions.
- **Implementation:** If the leaf CNN detects a biotic pathogen (fungus, bacteria, virus, pest), the decision engine prescribes plant protection measures (fungicide, insecticide, or bio-control). Chemical fertilizers are **strictly suppressed** unless independently justified by a verified soil deficiency.

### Rule 4: Cross-Modal Conflict Resolution (Nutrient Lockout vs Toxic Excess)
- **Constraint:** Handle conflicting model outputs safely.
- **Implementation:** If foliar imagery exhibits visual deficiency symptoms (e.g. leaf chlorosis from Nitrogen shortage), but the soil test reports **high/toxic nitrogen** or **high electrical conductivity (EC > 2200 µS/cm)**:
  1. The engine triggers a `NUTRIENT_LOCKOUT_CONFLICT` or `SALINITY_INDUCED_LOCKOUT` critical alert.
  2. The root cause is identified: root osmotic stress or chemical salt burn is blocking nutrient absorption.
  3. The engine **forbids** adding fertilizer and prescribes: `Cease Fertilizer & Flush Soil` with agricultural gypsum and leaching irrigation.

### Rule 5: Crop Mismatch & Low-Confidence Filtering
- **Implementation:** If the user specifies crop *Guava* but the CNN strongly predicts *Brinjal_Leaf_curl*, a `CROP_MISMATCH` alert is raised. If CNN confidence falls below 60%, a `LOW_CONFIDENCE_CNN` advisory prompts the farmer to cross-check with Grad-CAM heatmaps or field scouting.

---

## 📦 5. Deliverables Catalog

All deliverables have been implemented in `d:\cnn_model`:

| Deliverable | Purpose | Path |
| :--- | :--- | :--- |
| `integration.py` | Complete Multimodal Decision-Support Integration Controller | [`integration.py`](file:///d:/cnn_model/integration.py) |
| `recommendation_engine.py` | Rule-based Agronomic Decision Engine & XGBoost Model 2 Wrapper | [`recommendation_engine.py`](file:///d:/cnn_model/recommendation_engine.py) |
| `fertilizer_knowledge_base.csv` | 122 validated crop-specific expert agronomic rules | [`fertilizer_knowledge_base.csv`](file:///d:/cnn_model/fertilizer_knowledge_base.csv) |
| `app.py` | Interactive Streamlit Web Application Dashboard | [`app.py`](file:///d:/cnn_model/app.py) |
| `cli.py` | Automated Command-Line Interface | [`cli.py`](file:///d:/cnn_model/cli.py) |
| `test_integration.py` | 10-point Automated Integration & Safety Test Suite | [`test_integration.py`](file:///d:/cnn_model/test_integration.py) |
| `DECISION_INTEGRATION_GUIDE.md` | Full PhD Research Methodology Documentation | [`DECISION_INTEGRATION_GUIDE.md`](file:///d:/cnn_model/DECISION_INTEGRATION_GUIDE.md) |

---

## 🧪 6. Verification and Integration Test Results

The automated integration test suite (`test_integration.py`) verifies all 10 core system requirements:

```text
======================================================================
STARTING DECISION-LEVEL INTEGRATION TEST SUITE
======================================================================
Initializing Multimodal Agricultural Decision-Support System...
[OK] Loaded CNN SavedModel from: models/efficientnetb0/efficientnetb0_final_savedmodel (Target resolution: (256, 256))
[OK] All system modules loaded and verified successfully.

test_01_knowledge_base_integrity: 122 records across 5 crops ............................ [PASS]
test_02_xgboost_models_schemas: Model 1 (23 feats) & Model 2 (55 feats, 9 classes) ...... [PASS]
test_03_healthy_leaf_and_healthy_soil: Low Risk, standard cultural practices ............ [PASS]
test_04_biotic_disease_strictly_no_fertilizer: Plant protection only, NO fertilizer ...... [PASS]
test_05_nutritional_foliar_deficiency_convergence: Prescribes Urea on verified deficit .. [PASS]
test_06_conflict_resolution_nutrient_lockout: Ceases fertilizer upon soil excess/salinity  [PASS]
test_07_crop_mismatch_detection: Flags discrepancy between declared and predicted crop .. [PASS]
test_08_low_confidence_detection: Issues warning when confidence < 60% ................... [PASS]
test_09_missing_inputs_modes: Leaf-only and Soil-only modes execute gracefully .......... [PASS]
test_10_unverified_sample_pairing_guardrail: Flags unlinked records safely .............. [PASS]

----------------------------------------------------------------------
Ran 10 tests in 7.361s

OK
```

---

## 🚀 7. Application Execution Guide

### 7.1 Running the Interactive Streamlit Web Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
**Features:**
- Real-time foliar image upload or gallery browsing from `Final_images_phd`.
- Interactive soil parameter tuning with built-in scenario presets.
- Side-by-side original image and Grad-CAM saliency heatmaps.
- Soil health gauges and SHAP local feature attribution horizontal bar charts.
- Synthesized agronomic prescriptions with one-click JSON report download.

### 7.2 Running the Command-Line Interface (CLI)
For batch testing and terminal evaluation:
```bash
# Full Multimodal Diagnosis
python cli.py --crop Guava --image Final_images_phd/Guava/Healthy_Guava/Guava_Healthy_001.jpg --soil-json "{\"pH\": 7.2, \"Conductivity\": 1150.0, \"Nitrogen\": 320.0}"

# Leaf-Only Diagnosis
python cli.py --crop Cumin --image Final_images_phd/Cumin/Unhealthy_Cumin/Powdery_Mildew/aug_rot_001.jpg

# Save Report to JSON
python cli.py --crop Brinjal --soil-json "{\"pH\": 8.6, \"Conductivity\": 2400.0, \"Nitrogen\": 620.0}" --out report.json
```

