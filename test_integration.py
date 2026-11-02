"""
Integration Testing Suite for Multimodal Agricultural Decision-Support System (MADSS)
Verifies all 10 core integration requirements, safety constraints, and conflict resolutions.

Run via: python test_integration.py
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from recommendation_engine import AgronomicKnowledgeBase, FertilizerMLModel, DecisionEngine
from integration import MultimodalIntegrationSystem, CNNLeafDiseaseModel, SoilHealthModel


class TestDecisionLevelIntegration(unittest.TestCase):
    """Rigorous test cases verifying model loading, inference, and decision rules."""

    @classmethod
    def setUpClass(cls):
        print("\n======================================================================")
        print("STARTING DECISION-LEVEL INTEGRATION TEST SUITE")
        print("======================================================================")
        cls.system = MultimodalIntegrationSystem()
        cls.kb = cls.system.kb
        cls.fert_ml = cls.system.fert_ml
        cls.engine = cls.system.decision_engine

    def test_01_knowledge_base_integrity(self):
        """Test 1: Verify knowledge base loading and record counts."""
        self.assertIsNotNone(self.kb.df_kb, "Knowledge base CSV dataframe is None")
        self.assertEqual(len(self.kb.df_kb), 122, "Expected 122 validated agronomic records")
        crops = set(self.kb.df_kb['Crop'].str.capitalize().unique())
        expected_crops = {"Guava", "Papaya", "Brinjal", "Cumin", "Castor"}
        self.assertTrue(expected_crops.issubset(crops), f"Missing crops in KB: {expected_crops - crops}")
        print("[PASS] Test 1: Agronomic knowledge base verified (122 records across 5 crops).")

    def test_02_xgboost_models_schemas(self):
        """Test 2: Verify both XGBoost models load with exact expected schemas."""
        # Model 1: Soil Health (23 features)
        self.assertEqual(len(self.system.soil_model.feature_columns), 23)
        self.assertEqual(self.system.soil_model.model.n_features_in_, 23)
        # Model 2: Fertilizer Intervention (55 features, 9 classes)
        self.assertEqual(len(self.fert_ml.feature_columns), 55)
        self.assertEqual(self.fert_ml.model.n_features_in_, 55)
        self.assertEqual(len(self.fert_ml.label_mapping), 9)
        print("[PASS] Test 2: XGBoost Model 1 (23 features) and Model 2 (55 features, 9 classes) verified.")

    def test_03_healthy_leaf_and_healthy_soil(self):
        """Test 3: Full Multimodal Alignment - Healthy Leaf + Healthy Soil."""
        soil_readings = {
            'pH': 7.2,
            'Conductivity': 1200.0,
            'Nitrogen': 330.0,
            'Phosphorus': 30.0,
            'Potassium': 280.0,
            'Zinc': 2.0,
            'Ferrous': 15.0
        }
        res = self.engine.synthesize(
            crop="Guava",
            cnn_result={"predicted_class": "Guava_Healthy", "confidence": 98.2},
            soil_health_result={"Status": "Healthy", "Confidence": 95.0},
            raw_soil_readings=soil_readings,
            is_paired_sample=True
        )
        self.assertEqual(res["overall_risk_level"], "Low")
        self.assertEqual(len(res["fertilizer_prescriptions"]), 0, "No fertilizer needed for healthy crop & soil")
        self.assertEqual(len(res["plant_protection_advisory"]), 0, "No plant protection needed for healthy leaf")
        self.assertIn("OPTIMAL & HEALTHY", res["agronomic_summary"])
        print("[PASS] Test 3: Healthy leaf + Healthy soil produces Low Risk and standard maintenance.")

    def test_04_biotic_disease_strictly_no_fertilizer(self):
        """
        Test 4: Biotic Leaf Disease (e.g. Anthracnose) + Healthy Soil.
        Constraint: Never generate fertilizer recommendations solely from disease predictions.
        """
        res = self.engine.synthesize(
            crop="Guava",
            cnn_result={"predicted_class": "Guava_anthrecnose", "confidence": 92.5},
            soil_health_result={"Status": "Healthy", "Confidence": 91.0},
            raw_soil_readings={'pH': 7.1, 'Conductivity': 1100.0, 'Nitrogen': 320.0},
            is_paired_sample=True
        )
        self.assertEqual(res["overall_risk_level"], "High")
        self.assertGreater(len(res["plant_protection_advisory"]), 0, "Plant protection must be prescribed")
        self.assertEqual(len(res["fertilizer_prescriptions"]), 0, "Strictly NO fertilizer for pure biotic disease!")
        prot = res["plant_protection_advisory"][0]
        self.assertIn("Copper", prot["Recommended_Protection"] + prot["Farmer_Advisory"])
        print("[PASS] Test 4: Biotic infection generates plant protection without improper fertilizer prescriptions.")

    def test_05_nutritional_foliar_deficiency_convergence(self):
        """Test 5: Multimodal Deficiency Convergence (Foliar visual symptom + verified soil deficiency)."""
        soil_readings = {
            'Nitrogen': 180.0,  # Below Guava critical threshold < 250 mg/kg
            'Phosphorus': 30.0,
            'Potassium': 280.0,
            'pH': 7.0,
            'Conductivity': 1200.0
        }
        res = self.engine.synthesize(
            crop="Guava",
            cnn_result={"predicted_class": "Guava_Nitrogen", "confidence": 89.0},
            soil_health_result={"Status": "Healthy", "Confidence": 75.0},
            raw_soil_readings=soil_readings,
            is_paired_sample=True
        )
        self.assertGreater(len(res["fertilizer_prescriptions"]), 0)
        fert = res["fertilizer_prescriptions"][0]
        self.assertEqual(fert["Nutrient"], "Nitrogen")
        self.assertIn("Multimodal Convergence", fert["Diagnosis_Type"])
        self.assertIn("Urea", fert["Recommended_Fertilizer"])
        print("[PASS] Test 5: Multimodal convergence correctly prescribes Urea for verified Nitrogen deficiency.")

    def test_06_conflict_resolution_nutrient_lockout(self):
        """
        Test 6: Conflict Resolution - Foliar Deficiency Symptom vs Toxic Soil Excess.
        Safety: Disallow adding fertilizer; command cessation & soil flushing.
        """
        soil_readings = {
            'Nitrogen': 650.0,  # Toxic excess > 500 mg/kg
            'Conductivity': 2300.0,  # Salinity stress > 2000 µS/cm
            'pH': 8.6
        }
        res = self.engine.synthesize(
            crop="Brinjal",
            cnn_result={"predicted_class": "Brinjal_Nitrogen", "confidence": 91.0},
            soil_health_result={"Status": "Unhealthy", "Confidence": 93.0},
            raw_soil_readings=soil_readings,
            is_paired_sample=True
        )
        # Must detect conflict
        conflict_types = [w["type"] for w in res["warnings_and_conflicts"]]
        self.assertIn("NUTRIENT_LOCKOUT_CONFLICT", conflict_types)
        # Must NOT prescribe Nitrogen fertilizer
        self.assertFalse(any(f["Nutrient"] == "Nitrogen" for f in res["fertilizer_prescriptions"]))
        # Must prescribe soil flush / cessation
        amendment_targets = [a["Target"] for a in res["soil_amendments_and_recovery"]]
        self.assertTrue(any("Nitrogen Toxicity" in t or "Salinity" in t for t in amendment_targets))
        print("[PASS] Test 6: Nutrient lockout conflict resolved safely (Fertilizer stopped, flushing prescribed).")

    def test_07_crop_mismatch_detection(self):
        """Test 7: Crop Mismatch Guardrail (User sets Guava, CNN identifies Brinjal)."""
        res = self.engine.synthesize(
            crop="Guava",
            cnn_result={"predicted_class": "Brinjal_Leaf_curl", "confidence": 95.0},
            soil_health_result={"Status": "Healthy", "Confidence": 85.0},
            raw_soil_readings={'pH': 7.0, 'Conductivity': 1100.0},
            is_paired_sample=True
        )
        warning_types = [w["type"] for w in res["warnings_and_conflicts"]]
        self.assertIn("CROP_MISMATCH", warning_types)
        print("[PASS] Test 7: Crop mismatch detected and flagged with High severity warning.")

    def test_08_low_confidence_detection(self):
        """Test 8: Low Confidence Handling (CNN confidence < 60%)."""
        res = self.engine.synthesize(
            crop="Papaya",
            cnn_result={"predicted_class": "Papaya_Healthy", "confidence": 48.5},
            soil_health_result={"Status": "Healthy", "Confidence": 90.0},
            raw_soil_readings={'pH': 7.0, 'Conductivity': 1100.0},
            is_paired_sample=True
        )
        warning_types = [w["type"] for w in res["warnings_and_conflicts"]]
        self.assertIn("LOW_CONFIDENCE_CNN", warning_types)
        print("[PASS] Test 8: Low confidence prediction triggers verification alert.")

    def test_09_missing_inputs_modes(self):
        """Test 9: Graceful handling of Leaf-only and Soil-only operating modes."""
        # 9a: Leaf-only mode
        res_leaf_only = self.engine.synthesize(
            crop="Cumin",
            cnn_result={"predicted_class": "Cumin_Powdery_Mildew", "confidence": 94.0},
            soil_health_result=None,
            raw_soil_readings=None
        )
        self.assertTrue(res_leaf_only["input_modes"]["leaf_image_provided"])
        self.assertFalse(res_leaf_only["input_modes"]["soil_measurements_provided"])
        self.assertGreater(len(res_leaf_only["plant_protection_advisory"]), 0)

        # 9b: Soil-only mode
        res_soil_only = self.engine.synthesize(
            crop="Castor",
            cnn_result=None,
            soil_health_result={"Status": "Unhealthy", "Confidence": 88.0},
            raw_soil_readings={'Phosphorus': 80.0, 'Conductivity': 2400.0}
        )
        self.assertFalse(res_soil_only["input_modes"]["leaf_image_provided"])
        self.assertTrue(res_soil_only["input_modes"]["soil_measurements_provided"])
        self.assertGreater(len(res_soil_only["soil_amendments_and_recovery"]), 0)
        print("[PASS] Test 9: Independent Leaf-only and Soil-only operational modes execute gracefully.")

    def test_10_unverified_sample_pairing_guardrail(self):
        """Test 10: Unverified sample pairing safety warning."""
        res = self.engine.synthesize(
            crop="Guava",
            cnn_result={"predicted_class": "Guava_Healthy", "confidence": 95.0},
            soil_health_result={"Status": "Healthy", "Confidence": 90.0},
            raw_soil_readings={'pH': 7.0, 'Conductivity': 1100.0},
            is_paired_sample=False  # Unlinked samples
        )
        warning_types = [w["type"] for w in res["warnings_and_conflicts"]]
        self.assertIn("UNVERIFIED_SAMPLE_PAIRING", warning_types)
        print("[PASS] Test 10: Unverified sample pairing flagged to prevent invalid causal assumption.")


if __name__ == "__main__":
    unittest.main(verbosity=2)

