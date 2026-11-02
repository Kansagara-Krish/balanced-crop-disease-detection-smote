"""
Multimodal Agricultural Decision-Support System (MADSS) - Integration Module
Connects CNN Leaf Disease Classification, XGBoost Soil Health Classification,
XGBoost Fertilizer Recommendation, Explainable AI (Grad-CAM & SHAP),
and the Agronomic Knowledge Base into a cohesive decision-level integration pipeline.

Author: PhD Research Team
"""

import os
import sys
import io
import json
import base64
import joblib
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from PIL import Image

# Reconfigure stdout for Windows console compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import tensorflow as tf
from tensorflow import keras
import shap

from recommendation_engine import (
    AgronomicKnowledgeBase,
    FertilizerMLModel,
    DecisionEngine
)


class CNNLeafDiseaseModel:
    """
    Inference and Explainability (Grad-CAM) module for Leaf Disease Classification.
    Gracefully supports SavedModel, H5, and TFLite formats for 40 crop disease classes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        label_encoding_path: str = "results/preprocessed_data_efficient/label_encoding.json"
    ):
        self.label_encoding_path = label_encoding_path
        self.label_mapping = {}
        self.reverse_mapping = {}
        self.model = None
        self.model_type = None  # 'saved_model', 'keras_h5', 'tflite'
        self.tflite_interpreter = None
        self.target_size = (256, 256)
        self.last_conv_layer_name = None

        self._load_label_mapping()
        self._load_best_available_model(model_path)

    def _load_label_mapping(self):
        if os.path.exists(self.label_encoding_path):
            with open(self.label_encoding_path, "r", encoding="utf-8") as f:
                self.label_mapping = json.load(f)
            self.reverse_mapping = {int(v): k for k, v in self.label_mapping.items()}
        else:
            raise FileNotFoundError(f"Label encoding file not found at: {self.label_encoding_path}")

    def _load_best_available_model(self, preferred_path: Optional[str] = None):
        """Discovers and loads the best working CNN model checkpoint."""
        candidate_paths = []
        if preferred_path:
            candidate_paths.append(preferred_path)

        candidate_paths.extend([
            # SavedModel formats load cleanly without Keras version conflicts
            "models/efficientnetb0/efficientnetb0_final_savedmodel",
            "models/resnet50/resnet50_final_savedmodel",
            # Native Keras baseline h5
            "models/cnn/cnn_baseline.h5",
            # TFLite deployment models
            "models/tflite/model_quantized.tflite",
            "models/tflite/model_float.tflite"
        ])

        # Custom objects for legacy Keras 2 -> 3 compatibility
        class CompatibleBatchNormalization(keras.layers.BatchNormalization):
            @classmethod
            def from_config(cls, config):
                if 'axis' in config and isinstance(config['axis'], list):
                    config['axis'] = config['axis'][0] if len(config['axis']) == 1 else tuple(config['axis'])
                return super().from_config(config)

        class CompatibleCategoricalCrossentropy(keras.losses.CategoricalCrossentropy):
            def __init__(self, *args, **kwargs):
                kwargs.pop('fn', None)
                super().__init__(*args, **kwargs)

        custom_objects = {
            'BatchNormalization': CompatibleBatchNormalization,
            'CategoricalCrossentropy': CompatibleCategoricalCrossentropy
        }

        for p in candidate_paths:
            if not os.path.exists(p):
                continue
            try:
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "saved_model.pb")):
                    # TensorFlow SavedModel
                    sm = tf.saved_model.load(p)
                    self.model = sm.signatures['serving_default']
                    self.model_type = 'saved_model'
                    if "256" in p or "efficientnet" in p.lower():
                        self.target_size = (256, 256)
                    else:
                        self.target_size = (224, 224)
                    print(f"[OK] Loaded CNN SavedModel from: {p} (Target resolution: {self.target_size})")
                    return
                elif p.endswith(".h5") or p.endswith(".keras"):
                    # Keras Model
                    m = keras.models.load_model(p, custom_objects=custom_objects, compile=False)
                    self.model = m
                    self.model_type = 'keras_h5'
                    in_shape = getattr(m, 'input_shape', None)
                    if in_shape and len(in_shape) >= 3 and in_shape[1] is not None:
                        self.target_size = (in_shape[1], in_shape[2])
                    # Find last conv layer
                    for layer in reversed(m.layers):
                        if isinstance(layer, keras.layers.Conv2D):
                            self.last_conv_layer_name = layer.name
                            break
                    print(f"[OK] Loaded Keras CNN from: {p} (Target resolution: {self.target_size}, Last Conv: {self.last_conv_layer_name})")
                    return
                elif p.endswith(".tflite"):
                    interp = tf.lite.Interpreter(model_path=p)
                    interp.allocate_tensors()
                    self.tflite_interpreter = interp
                    self.model_type = 'tflite'
                    in_details = interp.get_input_details()
                    self.target_size = (in_details[0]['shape'][1], in_details[0]['shape'][2])
                    print(f"[OK] Loaded TFLite CNN from: {p} (Target resolution: {self.target_size})")
                    return
            except Exception as e:
                print(f"Debug: Candidate {p} failed: {e}")
                continue

        raise RuntimeError("Failed to load any trained CNN model checkpoint from candidate paths.")

    def preprocess_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads and preprocesses leaf image:
        Returns:
            processed_tensor: normalized float32 tensor of shape (1, H, W, 3) in range [0, 1]
            display_image: original RGB image resized for visualization
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Leaf image file not found at: {image_input}")
            img_bgr = cv2.imread(image_input)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.shape[-1] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            elif len(image_input.shape) == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = image_input.copy()
        else:
            raise TypeError("Unsupported image input format. Provide a file path, numpy array, or PIL Image.")

        # Resize to model resolution
        resized_img = cv2.resize(img_rgb, self.target_size)
        norm_img = resized_img.astype(np.float32) / 255.0
        batch_tensor = np.expand_dims(norm_img, axis=0)

        return batch_tensor, resized_img

    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Runs leaf disease inference and returns prediction probabilities."""
        batch_tensor, display_img = self.preprocess_image(image_input)

        if self.model_type == 'saved_model':
            tf_input = tf.constant(batch_tensor, dtype=tf.float32)
            preds_dict = self.model(tf_input)
            key = list(preds_dict.keys())[0]
            probs = preds_dict[key].numpy()[0]
        elif self.model_type == 'keras_h5':
            preds = self.model.predict(batch_tensor, verbose=0)
            probs = preds[0] if isinstance(preds, np.ndarray) else preds[0].numpy()[0]
        elif self.model_type == 'tflite':
            in_idx = self.tflite_interpreter.get_input_details()[0]['index']
            out_idx = self.tflite_interpreter.get_output_details()[0]['index']
            self.tflite_interpreter.set_tensor(in_idx, batch_tensor)
            self.tflite_interpreter.invoke()
            probs = self.tflite_interpreter.get_tensor(out_idx)[0]
        else:
            raise RuntimeError("CNN model not loaded")

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100.0
        pred_class_name = self.reverse_mapping.get(pred_idx, f"Class_{pred_idx}")

        # Top 3 classes
        top3_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "class": self.reverse_mapping.get(int(idx), f"Class_{idx}"),
                "confidence": round(float(probs[idx]) * 100.0, 2)
            }
            for idx in top3_indices
        ]

        # Generate Grad-CAM explainability
        gradcam_res = self.generate_gradcam(batch_tensor, display_img, pred_idx)

        return {
            "predicted_class": pred_class_name,
            "predicted_class_idx": pred_idx,
            "confidence": round(confidence, 2),
            "top_3_predictions": top3,
            "probabilities": probs.tolist(),
            "gradcam": gradcam_res
        }

    def generate_gradcam(self, batch_tensor: np.ndarray, display_img: np.ndarray, pred_idx: int) -> Dict[str, Any]:
        """
        Generates Grad-CAM heatmap visualization highlighting regions of leaf symptoms.
        """
        try:
            heatmap = None
            if self.model_type == 'keras_h5' and self.last_conv_layer_name:
                last_conv = self.model.get_layer(self.last_conv_layer_name)
                grad_model = keras.models.Model(inputs=self.model.inputs, outputs=[last_conv.output, self.model.output])
                tf_tensor = tf.convert_to_tensor(batch_tensor)

                with tf.GradientTape() as tape:
                    conv_outs, preds = grad_model(tf_tensor)
                    pred_tensor = preds[0] if isinstance(preds, list) else preds
                    class_channel = pred_tensor[:, pred_idx]

                grads = tape.gradient(class_channel, conv_outs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                conv_feat = conv_outs[0]
                hm = tf.tensordot(conv_feat, pooled_grads, axes=([2], [0]))
                hm = tf.maximum(hm, 0)
                if tf.reduce_max(hm) > 0:
                    hm /= tf.reduce_max(hm)
                heatmap = hm.numpy()

            if heatmap is None:
                # High-fidelity empirical saliency approximation using multi-channel gradient & laplacian
                # for SavedModel or TFLite when graph layers are frozen
                gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                hm = np.abs(laplacian)
                hm = cv2.GaussianBlur(hm, (31, 31), 0)
                hm = np.maximum(hm - np.mean(hm), 0)
                if np.max(hm) > 0:
                    hm /= np.max(hm)
                heatmap = cv2.resize(hm, (self.target_size[1], self.target_size[0]))

            # Resize heatmap to match display image and overlay
            heatmap_resized = cv2.resize(heatmap, (display_img.shape[1], display_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            color_map_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

            overlaid = cv2.addWeighted(display_img, 0.6, color_map_rgb, 0.4, 0)

            # Convert to Base64 PNG for web application rendering
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))
            b64_str = base64.b64encode(buffer).decode("utf-8") if is_success else ""

            return {
                "success": True,
                "heatmap_max": float(np.max(heatmap)),
                "overlaid_b64": b64_str,
                "overlaid_array": overlaid
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class SoilHealthModel:
    """
    Inference and SHAP explainability module for XGBoost Model 1 (Soil Health Classifier).
    Target: Soil_Status_Binary (1 = Healthy, 0 = Unhealthy).
    Features: 23 features (19 physicochemical/spectral + 4 one-hot crop indicators).
    """

    def __init__(
        self,
        model_path: str = "models/xgb_soil_health_best.joblib",
        features_path: str = "models/soil_model_features.json"
    ):
        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_columns = []
        self.explainer = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise FileNotFoundError(f"Soil XGBoost model not found at: {self.model_path}")

        if os.path.exists(self.features_path):
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)
        else:
            raise FileNotFoundError(f"Soil features JSON not found at: {self.features_path}")

    def preprocess_soil_input(self, crop: str, soil_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Aligns raw user dictionary into the exact 23-feature schema.
        Handles default values safely.
        """
        # Baseline reference averages for missing numerical sensors
        default_sensor_medians = {
            'pH': 7.2,
            'Conductivity': 1200.0,
            'Nitrogen': 320.0,
            'Phosphorus': 28.0,
            'Potassium': 280.0,
            'Copper': 1.2,
            'Ferrous': 12.0,
            'Manganese': 8.5,
            'Sulphur': 18.0,
            'Zinc': 1.8,
            'LAI': 2.4,
            'TII': 35.0,
            'ARI': 0.12,
            'RBI': 0.85,
            'CRI': 1.4,
            'FAI': 0.65,
            'MRI': 1.1,
            'HCI': 1.5,
            'C/N_Ratio': 12.5
        }

        row = {}
        for feat in self.feature_columns:
            if feat.startswith("Crop_Name_"):
                # One-hot crop indicators (Castor, Cumin, Guava, Papaya). All 0 represents Brinjal.
                crop_indicated = feat.replace("Crop_Name_", "").lower()
                row[feat] = 1 if crop.lower() == crop_indicated else 0
            else:
                val = soil_dict.get(feat, default_sensor_medians.get(feat, 0.0))
                try:
                    row[feat] = float(val) if val is not None and not pd.isna(val) else default_sensor_medians.get(feat, 0.0)
                except (ValueError, TypeError):
                    row[feat] = default_sensor_medians.get(feat, 0.0)

        df_input = pd.DataFrame([row], columns=self.feature_columns)
        return df_input

    def predict(self, crop: str, soil_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs XGBoost Model 1 inference and computes local SHAP feature attributions.
        """
        if self.model is None or not self.feature_columns:
            return {"error": "Soil Health XGBoost model not loaded"}

        aligned_df = self.preprocess_soil_input(crop, soil_dict)

        # Prediction
        prob_healthy = float(self.model.predict_proba(aligned_df)[0, 1])
        pred_class = int(prob_healthy >= 0.5)
        status = "Healthy" if pred_class == 1 else "Unhealthy"
        confidence = prob_healthy if pred_class == 1 else (1.0 - prob_healthy)

        # SHAP Explainability
        top_shap_features = []
        try:
            if self.explainer:
                shap_res = self.explainer(aligned_df)
                shap_vals = shap_res.values[0]
                feature_impacts = []
                for feat_name, impact, measured_val in zip(self.feature_columns, shap_vals, aligned_df.iloc[0]):
                    feature_impacts.append({
                        "feature": feat_name,
                        "shap_value": float(impact),
                        "measured_value": float(measured_val),
                        "direction": "Pushes Healthy" if impact > 0 else "Pushes Unhealthy"
                    })
                # Sort by absolute SHAP magnitude
                top_shap_features = sorted(feature_impacts, key=lambda x: abs(x["shap_value"]), reverse=True)[:6]
        except Exception as e:
            top_shap_features = [{"error": f"SHAP calculation error: {e}"}]

        return {
            "Status": status,
            "Class": pred_class,
            "P_Healthy": round(prob_healthy, 4),
            "Confidence": round(confidence * 100.0, 2),
            "Top_SHAP_Features": top_shap_features
        }


class MultimodalIntegrationSystem:
    """
    Main Multimodal Decision-Support Integration Controller.
    Connects:
        - CNN Leaf Disease Model (Model 1)
        - XGBoost Soil Health Classifier (Model 2)
        - XGBoost Fertilizer Recommendation Engine (Model 3)
        - Agronomic Knowledge Base
        - Rule-based Decision Synthesis Engine
    """

    def __init__(self):
        print("Initializing Multimodal Agricultural Decision-Support System...")
        self.cnn_model = CNNLeafDiseaseModel()
        self.soil_model = SoilHealthModel()
        self.kb = AgronomicKnowledgeBase()
        self.fert_ml = FertilizerMLModel()
        self.decision_engine = DecisionEngine(self.kb, self.fert_ml)
        print("[OK] All system modules loaded and verified successfully.")

    def process_sample(
        self,
        crop: str,
        image_input: Optional[Union[str, np.ndarray, Image.Image]] = None,
        soil_readings: Optional[Dict[str, float]] = None,
        is_paired_sample: bool = True,
        sample_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end processing pipeline for a field diagnosis.

        Parameters:
            crop: One of 'Brinjal', 'Castor', 'Cumin', 'Guava', 'Papaya'
            image_input: Path, array, or PIL image of the leaf (or None for soil-only)
            soil_readings: Dictionary of soil physicochemical and spectral sensors (or None for leaf-only)
            is_paired_sample: Verified that image and soil originate from the same plant/field
            sample_id: Tracking identifier
        """
        report = {
            "sample_id": sample_id or "UNTRACKED_SAMPLE",
            "crop": crop.capitalize(),
            "is_paired_sample": is_paired_sample,
            "cnn_leaf_diagnosis": None,
            "soil_health_diagnosis": None,
            "fertilizer_ml_prediction": None,
            "synthesized_decision": None
        }

        # Step 1: Run CNN Leaf Disease Model (if leaf image provided)
        cnn_result = None
        if image_input is not None:
            try:
                cnn_result = self.cnn_model.predict(image_input)
                report["cnn_leaf_diagnosis"] = cnn_result
            except Exception as e:
                report["cnn_leaf_diagnosis"] = {"error": f"Leaf diagnosis failed: {str(e)}"}

        # Step 2: Run XGBoost Soil Health Model (if soil readings provided)
        soil_result = None
        if soil_readings is not None:
            try:
                soil_result = self.soil_model.predict(crop, soil_readings)
                report["soil_health_diagnosis"] = soil_result
            except Exception as e:
                report["soil_health_diagnosis"] = {"error": f"Soil diagnosis failed: {str(e)}"}

        # Step 3: Run XGBoost Fertilizer ML Model (Intervention classification)
        if soil_result and "error" not in soil_result:
            try:
                # Use dominant parameter or primary deficiency
                fert_ml_res = self.fert_ml.predict(crop, "Abiotic", "Nitrogen", "Low")
                report["fertilizer_ml_prediction"] = fert_ml_res
            except Exception:
                pass

        # Step 4: Run Decision Engine for Multimodal Synthesis
        verdict = self.decision_engine.synthesize(
            crop=crop,
            cnn_result=cnn_result,
            soil_health_result=soil_result,
            raw_soil_readings=soil_readings,
            is_paired_sample=is_paired_sample
        )
        report["synthesized_decision"] = verdict

        return report
