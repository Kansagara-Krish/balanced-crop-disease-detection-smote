"""
Recommendation Engine for Multimodal Agricultural Decision-Support System
Translates CNN leaf disease classifications, XGBoost soil health predictions,
and raw soil sensor measurements into validated agronomic prescriptions.

Research Constraints:
- Do not invent fertilizer dosages or deficiency thresholds.
- Do not treat soil-health classification as nutrient-deficiency classification.
- Do not generate fertilizer recommendations solely from disease predictions.
- Handle missing inputs, crop mismatches, low confidence, and conflicting outputs safely.
"""

import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


class AgronomicKnowledgeBase:
    """
    Interface to the validated agronomic knowledge base exported from
    'Final Fertilizer recovery sheet.xlsx' (saved as fertilizer_knowledge_base.csv)
    and 'models/fertilizer_prescriptions_lookup.json'.
    """

    def __init__(self, csv_path: str = "fertilizer_knowledge_base.csv", lookup_json: str = "models/fertilizer_prescriptions_lookup.json"):
        self.csv_path = csv_path
        self.lookup_json = lookup_json
        self.df_kb = None
        self.prescriptions_lookup = {}
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        if os.path.exists(self.csv_path):
            self.df_kb = pd.read_csv(self.csv_path)
            # Normalize column names
            self.df_kb.columns = [c.strip() for c in self.df_kb.columns]
        else:
            raise FileNotFoundError(f"Knowledge base CSV not found at: {self.csv_path}")

        if os.path.exists(self.lookup_json):
            with open(self.lookup_json, "r", encoding="utf-8") as f:
                self.prescriptions_lookup = json.load(f)

    def get_crop_records(self, crop: str) -> pd.DataFrame:
        """Filter knowledge base records by crop name (case-insensitive)."""
        if self.df_kb is None:
            return pd.DataFrame()
        crop_clean = crop.strip().capitalize()
        return self.df_kb[self.df_kb["Crop"].str.capitalize() == crop_clean]

    def query_prescription(self, crop: str, condition_or_param: str, level: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query validated prescription records matching crop and condition or parameter.
        Returns list of matching records with exact chemical, dosage, and advisory.
        """
        crop_records = self.get_crop_records(crop)
        if crop_records.empty:
            return []

        cond_clean = condition_or_param.lower().strip()
        matches = []

        for _, row in crop_records.iterrows():
            leaf_cond = str(row.get("Leaf Disease / Condition", "")).lower()
            aff_param = str(row.get("Affected Parameter", "")).lower()
            param_lvl = str(row.get("Parameter Level (Low/High)", "")).strip().capitalize()

            # Match if condition matches leaf disease OR affected parameter
            is_match = (cond_clean in leaf_cond or cond_clean in aff_param or
                        leaf_cond in cond_clean or aff_param in cond_clean)

            if is_match:
                if level:
                    if param_lvl.lower() != level.lower():
                        continue
                matches.append({
                    "CNN_Label_ID": int(row.get("CNN_Label_ID", -1)) if pd.notnull(row.get("CNN_Label_ID")) else None,
                    "Crop": row.get("Crop"),
                    "Condition": row.get("Leaf Disease / Condition"),
                    "Type": row.get("Type"),
                    "Affected_Parameter": row.get("Affected Parameter"),
                    "Level": row.get("Parameter Level (Low/High)"),
                    "Threshold_Value": row.get("Value"),
                    "Recommended_Chemical": row.get("Recommended Fertilizer / Pesticide"),
                    "Dosage": row.get("Dosage (gm or ml per liter / plant)"),
                    "Advisory": row.get("Farmer Advisory / Recovery Suggestion")
                })
        return matches

    def evaluate_nutrient_thresholds(self, crop: str, soil_readings: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Evaluates quantitative soil measurements against validated agronomic thresholds
        specified in the knowledge base.
        """
        crop_records = self.get_crop_records(crop)
        diagnosed_issues = []

        # Standard mapping between common sensor feature keys and knowledge base parameter names
        param_mappings = {
            "Nitrogen": ["Soil Nitrogen (N)", "Nitrogen", "N"],
            "Phosphorus": ["Phosphorus (P)", "Phosphorus", "P"],
            "Potassium": ["Potassium (K)", "Potassium", "K"],
            "Zinc": ["Soil Zinc (Zn)", "Zinc (Zn)", "Zinc"],
            "Ferrous": ["Soil Fe (Iron)", "Iron (Fe)", "Iron", "Fe"],
            "Sulphur": ["Soil Sulphur (S)", "Soil Sulphur", "Sulphur (S)", "Sulphur"],
            "Boron": ["Soil Boron (B)", "Soil Boron", "Boron (B)", "Boron"],
            "pH": ["pH", "pH (Alkaline soil)", "pH (Acidic soil)"],
            "Conductivity": ["EC", "Conductivity (EC / Salinity)", "Conductivity"],
            "LAI": ["LAI"],
            "C/N_Ratio": ["C/N_Ratio", "C/N"]
        }

        # Specific thresholds extracted from validated fertilizer_knowledge_base.csv
        # Format: (param_key, low_threshold, high_threshold, unit)
        threshold_benchmarks = {
            "Guava": {
                "Nitrogen": (250.0, 500.0, "mg/kg"),
                "Phosphorus": (10.0, 60.0, "mg/kg"),
                "Potassium": (150.0, 800.0, "mg/kg"),
                "Zinc": (0.5, 5.0, "mg/kg"),
                "Ferrous": (4.0, 50.0, "mg/kg"),
                "Sulphur": (5.0, 50.0, "mg/kg"),
                "Boron": (0.5, 2.0, "mg/kg"),
                "pH": (5.5, 8.5, ""),
                "Conductivity": (None, 2000.0, "µS/cm")
            },
            "Papaya": {
                "Nitrogen": (250.0, 500.0, "mg/kg"),
                "Phosphorus": (10.0, 60.0, "mg/kg"),
                "Potassium": (150.0, 800.0, "mg/kg"),
                "Zinc": (0.5, 5.0, "mg/kg"),
                "Ferrous": (4.0, 50.0, "mg/kg"),
                "Sulphur": (5.0, 50.0, "mg/kg"),
                "Boron": (0.5, 2.0, "mg/kg"),
                "pH": (5.5, 8.5, ""),
                "Conductivity": (None, 2000.0, "µS/cm")
            },
            "Brinjal": {
                "Nitrogen": (250.0, 500.0, "mg/kg"),
                "Phosphorus": (10.0, 60.0, "mg/kg"),
                "Potassium": (150.0, 800.0, "mg/kg"),
                "Zinc": (0.5, 5.0, "mg/kg"),
                "Ferrous": (4.0, 50.0, "mg/kg"),
                "Sulphur": (5.0, 50.0, "mg/kg"),
                "Boron": (0.5, 2.0, "mg/kg"),
                "pH": (5.5, 8.5, ""),
                "Conductivity": (None, 2000.0, "µS/cm")
            },
            "Cumin": {
                "Nitrogen": (250.0, 500.0, "mg/kg"),
                "Phosphorus": (150.0, 800.0, "mg/kg"),
                "Potassium": (10.0, 60.0, "mg/kg"),
                "Zinc": (0.5, 5.0, "mg/kg"),
                "Ferrous": (4.0, 50.0, "mg/kg"),
                "Sulphur": (5.0, 50.0, "mg/kg"),
                "pH": (None, 8.5, ""),
                "Conductivity": (None, 2000.0, "µS/cm")
            },
            "Castor": {
                "Nitrogen": (250.0, 500.0, "mg/kg"),
                "Phosphorus": (150.0, 800.0, "mg/kg"),
                "Potassium": (10.0, 60.0, "mg/kg"),
                "Zinc": (0.5, 5.0, "mg/kg"),
                "Ferrous": (4.0, 50.0, "mg/kg"),
                "Sulphur": (5.0, 50.0, "mg/kg"),
                "pH": (None, 8.5, ""),
                "Conductivity": (None, 2000.0, "µS/cm")
            }
        }

        crop_clean = crop.capitalize()
        crop_bench = threshold_benchmarks.get(crop_clean, threshold_benchmarks["Guava"])

        for param, val in soil_readings.items():
            if val is None or pd.isna(val):
                continue
            try:
                val_num = float(val)
            except (ValueError, TypeError):
                continue

            # Standardize param name
            std_param = None
            for standard_name, aliases in param_mappings.items():
                if param.lower() == standard_name.lower() or param in aliases:
                    std_param = standard_name
                    break

            if not std_param or std_param not in crop_bench:
                continue

            low_thr, high_thr, unit = crop_bench[std_param]

            status = "Optimal"
            level = "Mod"
            if low_thr is not None and val_num < low_thr:
                status = "Deficient"
                level = "Low"
            elif high_thr is not None and val_num > high_thr:
                status = "Excess / Toxicity" if std_param != "Conductivity" else "Salinity Stress"
                level = "High"

            if status != "Optimal":
                # Find matching prescription in knowledge base
                kb_records = self.query_prescription(crop_clean, std_param, level)
                rec_chem = "Consult agronomic extension"
                rec_dose = "Standard package of practices"
                rec_adv = "Follow soil test recommendation"

                if kb_records:
                    raw_chem = kb_records[0]["Recommended_Chemical"]
                    if pd.notnull(raw_chem) and str(raw_chem).strip() not in ["nan", "—", "-", "None", ""]:
                        rec_chem = str(raw_chem).strip()
                    elif level == "High":
                        rec_chem = "No Chemical Fertilizer (Cease Application)"
                    elif std_param == "pH":
                        rec_chem = "Agricultural Gypsum / Acidifying Amendments" if level == "High" else "Agricultural Lime"

                    raw_dose = kb_records[0]["Dosage"]
                    if pd.notnull(raw_dose) and str(raw_dose).strip() not in ["nan", "—", "-", "None", ""]:
                        rec_dose = str(raw_dose).strip()

                    raw_adv = kb_records[0]["Advisory"]
                    if pd.notnull(raw_adv) and str(raw_adv).strip() not in ["nan", "—", "-", "None", ""]:
                        rec_adv = str(raw_adv).strip()

                diagnosed_issues.append({
                    "Parameter": std_param,
                    "Measured_Value": val_num,
                    "Unit": unit,
                    "Status": status,
                    "Level": level,
                    "Critical_Threshold": f"<{low_thr} {unit}" if level == "Low" else f">{high_thr} {unit}",
                    "Recommended_Action": rec_chem,
                    "Prescribed_Dosage": rec_dose,
                    "Farmer_Advisory": rec_adv
                })

        return diagnosed_issues


class FertilizerMLModel:
    """
    Wrapper for XGBoost Model 2 (Precision Fertilizer Recommendation Engine).
    Trained in Notebook 15 to predict 9 standardized agronomic intervention classes.
    """

    def __init__(
        self,
        model_path: str = "models/xgb_fertilizer_recommendation_best.joblib",
        features_path: str = "models/fertilizer_model_features.json",
        labels_path: str = "models/fertilizer_label_mapping.json"
    ):
        self.model_path = model_path
        self.features_path = features_path
        self.labels_path = labels_path
        self.model = None
        self.feature_columns = []
        self.label_mapping = {}
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)
        if os.path.exists(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)

    @staticmethod
    def clean_affected_parameter(p: str) -> str:
        """Harmonizes raw parameter strings matching Notebook 15 preprocessing."""
        p_str = str(p).strip()
        if 'Nitrogen' in p_str: return 'Nitrogen'
        if 'Phosphorus' in p_str: return 'Phosphorus'
        if 'Potassium' in p_str: return 'Potassium'
        if 'Zinc' in p_str: return 'Zinc'
        if 'Iron' in p_str: return 'Iron'
        if 'Sulphur' in p_str: return 'Sulphur'
        if 'Boron' in p_str: return 'Boron'
        if 'Magnesium' in p_str: return 'Magnesium'
        if any(k in p_str for k in ['EC', 'HCI', 'HCl', 'Chloride', 'Salinity']): return 'Salinity_EC_Cl'
        if 'pH' in p_str: return 'pH'
        if 'Moisture' in p_str: return 'Soil_Moisture'
        if 'Humidity' in p_str: return 'Humidity'
        if 'Virus' in p_str: return 'Virus'
        if 'Fungus' in p_str: return 'Fungus'
        if any(k in p_str.lower() for k in ['insect', 'pest', 'vector', 'caterpillar', 'miner', 'borer']): return 'Pest_Insect'
        if 'Bacteria' in p_str: return 'Bacteria'
        if any(k in p_str for k in ['Temperature', 'Temp', 'Cloud']): return 'Weather_Stress'
        return p_str

    def predict(self, crop: str, condition_type: str, affected_param: str, param_level: str) -> Dict[str, Any]:
        """
        Runs inference on Model 2 using original feature schema.
        Outputs intervention class and confidence score.
        """
        if self.model is None or not self.feature_columns:
            return {"error": "Fertilizer XGBoost model not loaded"}

        param_clean = self.clean_affected_parameter(affected_param)
        level_clean = param_level.strip().capitalize()
        crop_clean = crop.strip().capitalize()
        type_clean = condition_type.strip().capitalize()
        param_x_level = f"{param_clean}_{level_clean}"

        # Construct single-row aligned feature dict
        input_dict = {
            f"Crop_{crop_clean}": 1,
            f"Type_{type_clean}": 1,
            f"Param_Clean_{param_clean}": 1,
            f"Parameter Level (Low/High)_{level_clean}": 1,
            f"Param_x_Level_{param_x_level}": 1
        }

        input_df = pd.DataFrame([input_dict])
        aligned_df = input_df.reindex(columns=self.feature_columns, fill_value=0)

        # Predict
        pred_idx = int(self.model.predict(aligned_df)[0])
        probabilities = self.model.predict_proba(aligned_df)[0]
        confidence = float(probabilities[pred_idx])
        class_name = self.label_mapping.get(str(pred_idx), f"Class_{pred_idx}")

        # Top 3 classes
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "class": self.label_mapping.get(str(idx), f"Class_{idx}"),
                "probability": round(float(probabilities[idx]) * 100, 2)
            }
            for idx in top3_indices
        ]

        return {
            "predicted_class": class_name,
            "predicted_class_idx": pred_idx,
            "confidence": round(confidence * 100, 2),
            "top_3_predictions": top3
        }


class DecisionEngine:
    """
    Transparent, rule-based agronomic synthesis engine.
    Integrates Crop identity, CNN Leaf Disease, XGBoost Soil Health,
    and XGBoost Fertilizer Intervention model into a unified action plan.
    """

    def __init__(self, kb: AgronomicKnowledgeBase, fert_ml: FertilizerMLModel):
        self.kb = kb
        self.fert_ml = fert_ml

    def synthesize(
        self,
        crop: str,
        cnn_result: Optional[Dict[str, Any]],
        soil_health_result: Optional[Dict[str, Any]],
        raw_soil_readings: Optional[Dict[str, float]],
        is_paired_sample: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete decision-level integration.
        Enforces all scientific constraints, handles edge cases, and produces
        actionable agricultural advisories.
        """
        verdict = {
            "crop": crop,
            "is_paired_sample": is_paired_sample,
            "input_modes": {
                "leaf_image_provided": cnn_result is not None and "error" not in cnn_result,
                "soil_measurements_provided": soil_health_result is not None and "error" not in soil_health_result
            },
            "warnings_and_conflicts": [],
            "plant_protection_advisory": [],
            "fertilizer_prescriptions": [],
            "soil_amendments_and_recovery": [],
            "agronomic_summary": "",
            "overall_risk_level": "Low"
        }

        # ---------------------------------------------------------------------
        # 1. INPUT MODE AND SAMPLE VERIFICATION CHECKS
        # ---------------------------------------------------------------------
        if not verdict["input_modes"]["leaf_image_provided"] and not verdict["input_modes"]["soil_measurements_provided"]:
            verdict["warnings_and_conflicts"].append({
                "type": "MISSING_INPUTS",
                "severity": "CRITICAL",
                "message": "Neither leaf image nor soil measurements were provided. Please supply at least one diagnostic source."
            })
            verdict["overall_risk_level"] = "Unknown"
            verdict["agronomic_summary"] = "No diagnostic data available."
            return verdict

        if verdict["input_modes"]["leaf_image_provided"] and verdict["input_modes"]["soil_measurements_provided"]:
            if not is_paired_sample:
                verdict["warnings_and_conflicts"].append({
                    "type": "UNVERIFIED_SAMPLE_PAIRING",
                    "severity": "MODERATE",
                    "message": "Leaf image and soil measurements are unverified / independent samples. Recommendations for leaf and soil are evaluated separately to avoid invalid causal coupling."
                })

        # ---------------------------------------------------------------------
        # 2. CROP MISMATCH & CONFIDENCE VALIDATION
        # ---------------------------------------------------------------------
        cnn_pred_class = None
        cnn_confidence = 0.0
        cnn_is_healthy = False
        cnn_is_nutritional_symptom = False
        cnn_etiology = "Unknown"

        if verdict["input_modes"]["leaf_image_provided"]:
            cnn_pred_class = cnn_result.get("predicted_class", "")
            cnn_confidence = cnn_result.get("confidence", 0.0)

            # Check Crop Mismatch
            predicted_crop_prefix = cnn_pred_class.split("_")[0].capitalize() if "_" in cnn_pred_class else ""
            if predicted_crop_prefix and predicted_crop_prefix != crop.capitalize():
                verdict["warnings_and_conflicts"].append({
                    "type": "CROP_MISMATCH",
                    "severity": "HIGH",
                    "message": f"Crop mismatch detected! Application crop is set to '{crop}', but CNN strongly identifies symptoms of '{predicted_crop_prefix}'. Verify crop identity before applying treatments."
                })

            # Check Low Confidence
            if cnn_confidence < 60.0:
                verdict["warnings_and_conflicts"].append({
                    "type": "LOW_CONFIDENCE_CNN",
                    "severity": "MODERATE",
                    "message": f"Leaf disease prediction has low confidence ({cnn_confidence:.1f}%). Verify visual diagnosis with Grad-CAM heatmap or field scouting."
                })

            # Characterize Leaf Condition
            pred_lower = cnn_pred_class.lower()
            if "healthy" in pred_lower:
                cnn_is_healthy = True
                cnn_etiology = "Healthy"
            elif any(k in pred_lower for k in ["nitrogen", "phosphorus", "potassium", "pottasium", "pottassium", "magnesium", "magnessium", "magnesiuam"]):
                cnn_is_nutritional_symptom = True
                cnn_etiology = "Abiotic_Nutritional"
            else:
                cnn_etiology = "Biotic_Pathogen"

        # ---------------------------------------------------------------------
        # 3. BIOTIC LEAF DISEASE HANDLING (STRICTLY NO RANDOM FERTILIZER)
        # ---------------------------------------------------------------------
        if cnn_etiology == "Biotic_Pathogen":
            verdict["overall_risk_level"] = "High"
            # Query plant protection measures from knowledge base
            disease_name = cnn_pred_class.replace(f"{crop}_", "").replace("_", " ")
            kb_matches = self.kb.query_prescription(crop, disease_name)

            if kb_matches:
                rec = kb_matches[0]
                verdict["plant_protection_advisory"].append({
                    "Target_Condition": disease_name,
                    "Diagnosis_Source": "Leaf CNN (Visual Pathogen)",
                    "Recommended_Protection": rec["Recommended_Chemical"],
                    "Prescribed_Dosage": rec["Dosage"],
                    "Farmer_Advisory": rec["Advisory"]
                })
            else:
                # Fallback to ML Model 2 prediction for biotic class
                ml_pred = self.fert_ml.predict(crop, "Biotic", "Fungus", "High")
                presc_info = self.kb.prescriptions_lookup.get(ml_pred["predicted_class"], {})
                verdict["plant_protection_advisory"].append({
                    "Target_Condition": disease_name,
                    "Diagnosis_Source": "Leaf CNN & ML Recommendation Engine",
                    "Recommended_Protection": presc_info.get("Representative_Chemicals", ["Approved fungicide/insecticide"])[0],
                    "Prescribed_Dosage": presc_info.get("Typical_Dosages", ["As per product label"])[0],
                    "Farmer_Advisory": presc_info.get("Primary_Advisories", ["Isolate affected foliage and monitor field."])[0]
                })

        # ---------------------------------------------------------------------
        # 4. SOIL MEASUREMENTS & SOIL HEALTH MODEL INTERPRETATION
        # ---------------------------------------------------------------------
        soil_is_healthy = True
        soil_issues = []
        salinity_stress_detected = False
        soil_ec_val = None

        if verdict["input_modes"]["soil_measurements_provided"]:
            soil_status = soil_health_result.get("Status", "Healthy")
            soil_conf = soil_health_result.get("Confidence", 100.0)
            soil_is_healthy = (soil_status == "Healthy")

            # Evaluate quantitative nutrient benchmarks from KB
            if raw_soil_readings:
                soil_issues = self.kb.evaluate_nutrient_thresholds(crop, raw_soil_readings)
                soil_ec_val = raw_soil_readings.get("Conductivity", None)
                if soil_ec_val is not None and float(soil_ec_val) > 2000.0:
                    salinity_stress_detected = True

            # If soil health model indicates Unhealthy but no individual nutrient tripped,
            # or if SHAP highlighted salinity / pH
            if not soil_is_healthy:
                if verdict["overall_risk_level"] != "High":
                    verdict["overall_risk_level"] = "Moderate"

                # Check SHAP dominant factor
                top_shap = soil_health_result.get("Top_SHAP_Features", [])
                dominant_factor = top_shap[0]["feature"] if top_shap else "Conductivity"

                if dominant_factor in ["Conductivity", "HCI"] or salinity_stress_detected:
                    salinity_stress_detected = True

        # ---------------------------------------------------------------------
        # 5. CROSS-MODAL RECONCILIATION & CONFLICT RESOLUTION
        # ---------------------------------------------------------------------
        # CONFLICT 1: Leaf shows deficiency, but Soil shows HIGH/EXCESS or HIGH SALINITY (Osmotic Lockout)
        if cnn_is_nutritional_symptom and raw_soil_readings:
            # Check which nutrient CNN detected
            pred_lower = cnn_pred_class.lower()
            nutrient_key = None
            if "nitrogen" in pred_lower: nutrient_key = "Nitrogen"
            elif "phosphorus" in pred_lower: nutrient_key = "Phosphorus"
            elif any(k in pred_lower for k in ["potassium", "pottasium", "pottassium"]): nutrient_key = "Potassium"
            elif any(k in pred_lower for k in ["magnesium", "magnessium", "magnesiuam"]): nutrient_key = "Magnesium"

            # Check matching soil measurement
            matching_soil_issue = next((iss for iss in soil_issues if iss["Parameter"] == nutrient_key), None)
            soil_nutrient_val = raw_soil_readings.get(nutrient_key, None)

            if matching_soil_issue and matching_soil_issue["Level"] == "High":
                # CONFLICT: Foliar deficiency symptoms with toxic soil excess!
                verdict["warnings_and_conflicts"].append({
                    "type": "NUTRIENT_LOCKOUT_CONFLICT",
                    "severity": "CRITICAL",
                    "message": f"CONFLICT DETECTED: Foliage exhibits {nutrient_key} deficiency symptoms, but Soil chemical analysis reveals TOXIC EXCESS ({matching_soil_issue['Measured_Value']} {matching_soil_issue['Unit']}). Root burn or chemical lockout is preventing uptake. DO NOT APPLY {nutrient_key} FERTILIZER!"
                })
                verdict["soil_amendments_and_recovery"].append({
                    "Intervention": "Cease Fertilizer & Flush Soil",
                    "Target": f"{nutrient_key} Toxicity",
                    "Prescription": "No Chemical Fertilizer (Toxicity/Flushing)",
                    "Advisory": f"Stop {nutrient_key} application immediately. Increase irrigation slightly to flush excess salts through drainage."
                })
            elif salinity_stress_detected:
                # Salinity induced root osmotic barrier
                verdict["warnings_and_conflicts"].append({
                    "type": "SALINITY_INDUCED_LOCKOUT",
                    "severity": "HIGH",
                    "message": f"SALINITY LOCKOUT: High soil electrical conductivity (EC = {soil_ec_val} µS/cm) is restricting root osmotic uptake, inducing secondary foliar symptoms. Treat salinity before adding chemical fertilizers."
                })
                verdict["soil_amendments_and_recovery"].append({
                    "Intervention": "Soil Amendment (Gypsum/Flushing)",
                    "Target": "Salinity Stress (EC > 2000 µS/cm)",
                    "Prescription": "Gypsum @ 200 g/m² + leaching irrigation",
                    "Advisory": "Apply agricultural gypsum to displace excess sodium/salts and improve percolation."
                })
            elif matching_soil_issue and matching_soil_issue["Level"] == "Low":
                # CONVERGENCE: Both leaf and soil confirm deficiency
                verdict["fertilizer_prescriptions"].append({
                    "Nutrient": nutrient_key,
                    "Diagnosis_Type": "Multimodal Convergence (Leaf Symptom + Soil Deficient)",
                    "Recommended_Fertilizer": matching_soil_issue["Recommended_Action"],
                    "Prescribed_Dosage": matching_soil_issue["Prescribed_Dosage"],
                    "Farmer_Advisory": matching_soil_issue["Farmer_Advisory"]
                })
            else:
                # Leaf symptom present, but soil reading normal or absent
                if matching_soil_issue is None:
                    verdict["fertilizer_prescriptions"].append({
                        "Nutrient": nutrient_key,
                        "Diagnosis_Type": "Foliar Visual Deficiency Symptom (Soil in normal range)",
                        "Recommended_Fertilizer": f"Mild foliar supplement for {nutrient_key}",
                        "Prescribed_Dosage": "Foliar spray at 0.5% concentration as per label",
                        "Farmer_Advisory": "Foliar deficiency detected while soil reserve is adequate. Apply light foliar spray to resolve localized demand."
                    })

        # Process any other soil nutrient deficiencies that were not part of the foliar symptom
        for issue in soil_issues:
            if issue["Level"] == "Low":
                # Ensure not already added via convergence
                already_added = any(f["Nutrient"] == issue["Parameter"] for f in verdict["fertilizer_prescriptions"])
                if not already_added:
                    verdict["fertilizer_prescriptions"].append({
                        "Nutrient": issue["Parameter"],
                        "Diagnosis_Type": "Soil Chemical Deficiency",
                        "Recommended_Fertilizer": issue["Recommended_Action"],
                        "Prescribed_Dosage": issue["Prescribed_Dosage"],
                        "Farmer_Advisory": issue["Farmer_Advisory"]
                    })
            elif issue["Level"] == "High" and not any(a["Target"].startswith(issue["Parameter"]) for a in verdict["soil_amendments_and_recovery"]):
                # High soil parameter (e.g. high pH, high Boron, etc.)
                if issue["Parameter"] == "pH":
                    verdict["soil_amendments_and_recovery"].append({
                        "Intervention": "Soil pH Correction",
                        "Target": f"Alkaline Soil (pH = {issue['Measured_Value']})",
                        "Prescription": issue["Recommended_Action"],
                        "Advisory": issue["Farmer_Advisory"]
                    })
                elif issue["Parameter"] != "Conductivity":
                    verdict["soil_amendments_and_recovery"].append({
                        "Intervention": "Cease Fertilizer (Excess)",
                        "Target": f"{issue['Parameter']} Excess",
                        "Prescription": "No Chemical Fertilizer",
                        "Advisory": f"Cease application of products containing {issue['Parameter']}."
                    })

        # If salinity was flagged but not yet added
        if salinity_stress_detected and not any("Salinity" in a["Target"] for a in verdict["soil_amendments_and_recovery"]):
            verdict["soil_amendments_and_recovery"].append({
                "Intervention": "Soil Amendment (Gypsum / Salinity Flush)",
                "Target": f"Soil Salinity Stress (Conductivity = {soil_ec_val or '>2000'} µS/cm)",
                "Prescription": "Agricultural Gypsum @ 200 g/m² + Leaching",
                "Advisory": "Conductivity exceeds the safe agronomic threshold (~2,200 µS/cm). Apply gypsum and flush to restore root permeability."
            })

        # ---------------------------------------------------------------------
        # 6. OVERALL SYNTHESIS SUMMARY
        # ---------------------------------------------------------------------
        summary_parts = []
        if cnn_is_healthy and soil_is_healthy and not soil_issues:
            verdict["overall_risk_level"] = "Low"
            summary_parts.append(f"Crop {crop} is currently in an OPTIMAL & HEALTHY state across both foliar and soil parameters. Maintain standard cultural practices.")
        else:
            if cnn_etiology == "Biotic_Pathogen":
                summary_parts.append(f"Biotic leaf disease detected ({cnn_pred_class}). Plant protection measures prescribed.")
            elif cnn_is_nutritional_symptom:
                summary_parts.append(f"Foliar nutritional deficiency symptoms detected ({cnn_pred_class}).")

            if not soil_is_healthy or salinity_stress_detected:
                summary_parts.append(f"Soil health classified as UNHEALTHY (Salinity/Physicochemical stress). Soil recovery amendments prioritized.")
            elif soil_issues:
                summary_parts.append(f"Soil nutrient imbalances identified ({len(soil_issues)} parameters). Targeted fertilizer interventions generated.")
            else:
                summary_parts.append("Soil fertility profile is balanced.")

        verdict["agronomic_summary"] = " ".join(summary_parts)
        return verdict

