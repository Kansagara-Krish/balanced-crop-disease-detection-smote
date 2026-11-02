"""
Multimodal Agricultural Decision-Support System (MADSS)
Interactive Streamlit Application for 5 Crops: Guava, Papaya, Brinjal, Cumin, Castor.

Integrates:
1. CNN Leaf Disease Classification + Grad-CAM Explainability
2. XGBoost Soil Health Classification + SHAP Local Feature Attribution
3. XGBoost Precision Fertilizer Recommendation Engine
4. Validated Agronomic Knowledge Base & Cross-Modal Decision Engine

Run via:
    streamlit run app.py
"""

import os
import sys
import json
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from integration import MultimodalIntegrationSystem


# Page Configuration
st.set_page_config(
    page_title="Multimodal Agricultural Decision-Support System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b5e20;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4caf50;
        margin-bottom: 15px;
    }
    .alert-card-critical {
        background-color: #ffebee;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #d32f2f;
        margin-bottom: 15px;
    }
    .alert-card-warning {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #ffb300;
        margin-bottom: 15px;
    }
    .action-header {
        font-weight: 600;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Cache the multimodal integration system instance."""
    return MultimodalIntegrationSystem()


system = load_system()

# Header
st.markdown('<div class="main-title">🌿 Multimodal Agricultural Decision-Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable Decision-Level Integration of Leaf Vision, Soil Health & Agronomic Prescriptions for PhD Research</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("📋 Diagnostic Session Setup")

crops = ["Guava", "Papaya", "Brinjal", "Cumin", "Castor"]
selected_crop = st.sidebar.selectbox("Select Target Crop Species:", crops, index=0)

diagnostic_mode = st.sidebar.radio(
    "Operating Diagnostic Mode:",
    ["Full Multimodal (Leaf + Soil)", "Leaf Vision Only", "Soil Chemical Test Only"]
)

is_paired = st.sidebar.checkbox(
    "Verified Paired Sample",
    value=True,
    help="Check this if the leaf photograph and the soil test sample originate from the same verified field/plant."
)

sample_id_input = st.sidebar.text_input("Field Plot / Sample ID:", value="PLOT_A_101")

st.sidebar.markdown("---")
st.sidebar.info("""
**Agronomic Research Guardrails:**
- Soil health classification is **not** treated as single-nutrient deficiency.
- Chemical fertilizers are **never** prescribed solely on disease detection.
- Cross-modal conflicts (e.g. salinity lockout) trigger safety alerts.
""")

# Sample Soil Presets for Quick Testing
sample_soil_presets = {
    "Balanced Healthy Soil": {
        'pH': 7.2, 'Conductivity': 1150.0, 'Nitrogen': 330.0, 'Phosphorus': 35.0,
        'Potassium': 280.0, 'Zinc': 2.2, 'Ferrous': 15.0, 'Sulphur': 22.0, 'Boron': 1.1
    },
    "Salinity Stress (High EC > 2200)": {
        'pH': 8.4, 'Conductivity': 2600.0, 'Nitrogen': 310.0, 'Phosphorus': 25.0,
        'Potassium': 210.0, 'Zinc': 1.5, 'Ferrous': 9.0, 'Sulphur': 16.0, 'Boron': 1.2
    },
    "Nitrogen Deficient Soil (< 250 mg/kg)": {
        'pH': 7.0, 'Conductivity': 1200.0, 'Nitrogen': 175.0, 'Phosphorus': 32.0,
        'Potassium': 270.0, 'Zinc': 1.9, 'Ferrous': 14.0, 'Sulphur': 18.0, 'Boron': 1.0
    },
    "Nitrogen Toxic Excess (> 500 mg/kg)": {
        'pH': 7.3, 'Conductivity': 1850.0, 'Nitrogen': 620.0, 'Phosphorus': 45.0,
        'Potassium': 310.0, 'Zinc': 2.0, 'Ferrous': 16.0, 'Sulphur': 25.0, 'Boron': 1.1
    }
}

# Main Interface Layout
col_input1, col_input2 = st.columns([1, 1])

uploaded_image = None
raw_soil_inputs = None

# Left Column: Leaf Input
with col_input1:
    st.subheader("📷 1. Foliar Image Input")
    if diagnostic_mode in ["Full Multimodal (Leaf + Soil)", "Leaf Vision Only"]:
        img_source = st.radio("Leaf Image Source:", ["Sample Gallery", "Upload File"], horizontal=True)

        if img_source == "Sample Gallery":
            # Search available test images for selected crop
            crop_img_dir = os.path.join("Final_images_phd", selected_crop)
            gallery_options = {}
            if os.path.exists(crop_img_dir):
                for root, dirs, files in os.walk(crop_img_dir):
                    for f in files:
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                            rel_name = os.path.relpath(os.path.join(root, f), crop_img_dir)
                            gallery_options[rel_name] = os.path.join(root, f)
                            if len(gallery_options) >= 8:
                                break
                    if len(gallery_options) >= 8:
                        break

            if gallery_options:
                selected_sample_rel = st.selectbox("Choose Sample Image from Dataset:", list(gallery_options.keys()))
                sample_img_path = gallery_options[selected_sample_rel]
                uploaded_image = Image.open(sample_img_path)
                st.image(uploaded_image, caption=f"Selected: {selected_sample_rel}", use_container_width=True)
            else:
                st.warning(f"No sample images found for {selected_crop}.")
        else:
            file_upload = st.file_uploader("Upload Leaf Photograph (JPG/PNG):", type=["jpg", "jpeg", "png"])
            if file_upload:
                uploaded_image = Image.open(file_upload)
                st.image(uploaded_image, caption="Uploaded Leaf Photograph", use_container_width=True)
    else:
        st.info("Leaf vision input disabled in Soil Chemical Test Only mode.")

# Right Column: Soil Input
with col_input2:
    st.subheader("🧪 2. Soil Test Sensor Readings")
    if diagnostic_mode in ["Full Multimodal (Leaf + Soil)", "Soil Chemical Test Only"]:
        preset_choice = st.selectbox("Load Test Scenario Preset:", list(sample_soil_presets.keys()))
        preset_vals = sample_soil_presets[preset_choice]

        with st.expander("Adjust Quantitative Soil Sensor Parameters", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                val_ph = st.number_input("Soil pH (acidity/alkalinity):", value=float(preset_vals.get('pH', 7.2)), step=0.1)
                val_ec = st.number_input("Electrical Conductivity EC (µS/cm):", value=float(preset_vals.get('Conductivity', 1200.0)), step=50.0)
                val_n = st.number_input("Available Nitrogen N (mg/kg):", value=float(preset_vals.get('Nitrogen', 320.0)), step=10.0)
                val_p = st.number_input("Available Phosphorus P (mg/kg):", value=float(preset_vals.get('Phosphorus', 30.0)), step=5.0)
            with c2:
                val_k = st.number_input("Available Potassium K (mg/kg):", value=float(preset_vals.get('Potassium', 280.0)), step=10.0)
                val_zn = st.number_input("Zinc Zn (mg/kg):", value=float(preset_vals.get('Zinc', 2.0)), step=0.1)
                val_fe = st.number_input("Ferrous / Iron Fe (mg/kg):", value=float(preset_vals.get('Ferrous', 14.0)), step=1.0)
                val_s = st.number_input("Sulphur S (mg/kg):", value=float(preset_vals.get('Sulphur', 20.0)), step=1.0)

        raw_soil_inputs = {
            'pH': val_ph,
            'Conductivity': val_ec,
            'Nitrogen': val_n,
            'Phosphorus': val_p,
            'Potassium': val_k,
            'Zinc': val_zn,
            'Ferrous': val_fe,
            'Sulphur': val_s,
            'Boron': preset_vals.get('Boron', 1.1)
        }
    else:
        st.info("Soil chemical measurements disabled in Leaf Vision Only mode.")

st.markdown("---")

# Execution Button
if st.button("🚀 Run Multimodal Decision Integration", type="primary", use_container_width=True):
    with st.spinner("Executing Multimodal CNN + XGBoost 1 + XGBoost 2 Decision Pipeline..."):
        img_arg = uploaded_image if diagnostic_mode != "Soil Chemical Test Only" else None
        soil_arg = raw_soil_inputs if diagnostic_mode != "Leaf Vision Only" else None

        result = system.process_sample(
            crop=selected_crop,
            image_input=img_arg,
            soil_readings=soil_arg,
            is_paired_sample=is_paired,
            sample_id=sample_id_input
        )

    st.success("✓ Multimodal Decision Synthesis Completed!")

    # Tabs for Diagnostic Components
    tab_summary, tab_vision, tab_soil, tab_actions = st.tabs([
        "📊 Integrated Decision Verdict",
        "👁️ Leaf Vision & Grad-CAM XAI",
        "🌱 Soil Health & SHAP Attributions",
        "💊 Actionable Prescriptions & Advisories"
    ])

    verdict = result["synthesized_decision"]

    # TAB 1: SUMMARY & VERDICT
    with tab_summary:
        st.subheader("🏁 Integrated Agronomic Diagnostic Report")

        # Risk Banner
        risk = verdict["overall_risk_level"]
        risk_color = "#4caf50" if risk == "Low" else ("#ff9800" if risk == "Moderate" else "#f44336")
        st.markdown(f"""
        <div style="background-color: {risk_color}22; border-left: 6px solid {risk_color}; padding: 12px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="margin: 0; color: {risk_color};">Overall Agronomic Risk Level: {risk.upper()}</h4>
            <p style="margin: 5px 0 0 0; font-size: 1.05rem;">{verdict['agronomic_summary']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Warnings and Conflicts
        if verdict["warnings_and_conflicts"]:
            st.markdown("### ⚠️ Active Safety Alerts & Conflict Resolutions")
            for w in verdict["warnings_and_conflicts"]:
                card_class = "alert-card-critical" if w["severity"] in ["HIGH", "CRITICAL"] else "alert-card-warning"
                st.markdown(f"""
                <div class="{card_class}">
                    <strong>[{w['severity']}] {w['type']}:</strong><br>
                    {w['message']}
                </div>
                """, unsafe_allow_html=True)

        # Overview Metrics Grid
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            leaf_pred = result["cnn_leaf_diagnosis"]["predicted_class"] if result["cnn_leaf_diagnosis"] else "N/A"
            m1.metric("Leaf Diagnosis", leaf_pred)
        with m2:
            leaf_conf = f"{result['cnn_leaf_diagnosis']['confidence']}%" if result["cnn_leaf_diagnosis"] else "N/A"
            m2.metric("CNN Confidence", leaf_conf)
        with m3:
            soil_stat = result["soil_health_diagnosis"]["Status"] if result["soil_health_diagnosis"] else "N/A"
            m3.metric("Soil Status", soil_stat)
        with m4:
            soil_conf = f"{result['soil_health_diagnosis']['Confidence']}%" if result["soil_health_diagnosis"] else "N/A"
            m4.metric("Soil Confidence", soil_conf)

    # TAB 2: LEAF VISION & GRAD-CAM
    with tab_vision:
        st.subheader("👁️ CNN Leaf Disease Diagnosis & Explainable AI (Grad-CAM)")
        if result["cnn_leaf_diagnosis"]:
            cnn_data = result["cnn_leaf_diagnosis"]
            v1, v2 = st.columns([1, 1])

            with v1:
                st.write("**Original Leaf Image & Confidence Breakdown:**")
                st.image(uploaded_image, use_container_width=True)
                st.markdown(f"**Primary Detection:** `{cnn_data['predicted_class']}` ({cnn_data['confidence']}%)")

                st.write("**Top 3 Class Candidates:**")
                for c in cnn_data.get("top_3_predictions", []):
                    st.progress(c["confidence"] / 100.0, text=f"{c['class']}: {c['confidence']}%")

            with v2:
                st.write("**Grad-CAM Saliency Attention Map:**")
                if cnn_data.get("gradcam", {}).get("success") and cnn_data["gradcam"].get("overlaid_array") is not None:
                    st.image(cnn_data["gradcam"]["overlaid_array"], caption="Grad-CAM Activation Heatmap (Red = High Saliency)", use_container_width=True)
                    st.caption("Warm red/yellow regions indicate specific leaf lesion structures that drove the neural network's decision.")
                else:
                    st.info("Grad-CAM visualization not available for this run.")
        else:
            st.info("Leaf vision diagnosis was not executed in this mode.")

    # TAB 3: SOIL HEALTH & SHAP
    with tab_soil:
        st.subheader("🌱 XGBoost Soil Health Modeling & Local SHAP Attributions")
        if result["soil_health_diagnosis"]:
            soil_data = result["soil_health_diagnosis"]
            s1, s2 = st.columns([1, 1])

            with s1:
                st.write(f"**Soil Classification:** `{soil_data['Status']}`")
                st.write(f"**Probability P(Healthy):** `{soil_data['P_Healthy']}`")
                st.write(f"**Model Confidence:** `{soil_data['Confidence']}%`")

                if soil_data['Status'] == "Unhealthy":
                    st.error("Soil health is compromised. Salinity or nutrient imbalance restricts optimal crop yield.")
                else:
                    st.success("Soil chemical matrix is fertile and suitable for cultivation.")

            with s2:
                st.write("**Local SHAP Feature Importance (TreeExplainer Attribution):**")
                shap_feats = soil_data.get("Top_SHAP_Features", [])
                if shap_feats and "error" not in shap_feats[0]:
                    df_shap = pd.DataFrame(shap_feats)
                    fig, ax = plt.subplots(figsize=(7, 4))
                    colors = ['#4caf50' if x > 0 else '#f44336' for x in df_shap['shap_value']]
                    ax.barh(df_shap['feature'], df_shap['shap_value'], color=colors)
                    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
                    ax.set_xlabel("SHAP Impact on Soil Health (Log-Odds)")
                    ax.set_title("Key Drivers (Green = Towards Healthy, Red = Towards Unhealthy)")
                    st.pyplot(fig)
        else:
            st.info("Soil health analysis was not executed in this mode.")

    # TAB 4: ACTIONABLE PRESCRIPTIONS
    with tab_actions:
        st.subheader("💊 Actionable Agronomic Prescriptions & Farmer Recovery Protocols")

        # 1. Plant Protection (Biotic)
        if verdict["plant_protection_advisory"]:
            st.markdown("### 🛡️ 1. Foliar Plant Protection Measures (Biotic Pathogens)")
            for p in verdict["plant_protection_advisory"]:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="action-header">Target Condition:</span> {p['Target_Condition']}<br>
                    <span class="action-header">Recommended Formulation:</span> <strong>{p['Recommended_Protection']}</strong><br>
                    <span class="action-header">Prescribed Dosage:</span> {p['Prescribed_Dosage']}<br>
                    <span class="action-header">Farmer Recovery Advisory:</span> {p['Farmer_Advisory']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No biotic plant disease detected. No fungicide or pesticide application required.")

        # 2. Fertilizer Prescriptions
        if verdict["fertilizer_prescriptions"]:
            st.markdown("### 🌾 2. Precision Fertilizer Prescriptions (Deficiencies)")
            for f in verdict["fertilizer_prescriptions"]:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="action-header">Nutrient Target:</span> {f['Nutrient']} ({f['Diagnosis_Type']})<br>
                    <span class="action-header">Prescribed Fertilizer:</span> <strong>{f['Recommended_Fertilizer']}</strong><br>
                    <span class="action-header">Agronomic Dosage:</span> {f['Prescribed_Dosage']}<br>
                    <span class="action-header">Application Guidelines:</span> {f['Farmer_Advisory']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No soil chemical deficiencies detected. No supplementary chemical fertilizer needed.")

        # 3. Soil Amendments and Recovery
        if verdict["soil_amendments_and_recovery"]:
            st.markdown("### 💧 3. Soil Amendments & Salinity / Toxicity Protocols")
            for a in verdict["soil_amendments_and_recovery"]:
                st.markdown(f"""
                <div class="alert-card-warning">
                    <span class="action-header">Intervention:</span> <strong>{a['Intervention']}</strong> ({a['Target']})<br>
                    <span class="action-header">Prescription:</span> {a['Prescription']}<br>
                    <span class="action-header">Advisory:</span> {a['Advisory']}
                </div>
                """, unsafe_allow_html=True)

        # Download Report Button
        st.markdown("---")
        report_json = json.dumps({
            "sample_id": result["sample_id"],
            "crop": result["crop"],
            "risk_level": verdict["overall_risk_level"],
            "summary": verdict["agronomic_summary"],
            "leaf_diagnosis": result["cnn_leaf_diagnosis"]["predicted_class"] if result["cnn_leaf_diagnosis"] else None,
            "soil_status": result["soil_health_diagnosis"]["Status"] if result["soil_health_diagnosis"] else None,
            "plant_protection": verdict["plant_protection_advisory"],
            "fertilizer_prescriptions": verdict["fertilizer_prescriptions"],
            "soil_amendments": verdict["soil_amendments_and_recovery"]
        }, indent=4)

        st.download_button(
            label="📥 Download Multimodal Diagnostic Report (JSON)",
            data=report_json,
            file_name=f"agronomic_report_{result['sample_id']}_{selected_crop}.json",
            mime="application/json"
        )

