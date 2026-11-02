"""
Command-Line Interface (CLI) for Multimodal Agricultural Decision-Support System
Allows terminal evaluation of leaf images, soil parameters, and crop identities.

Usage:
    python cli.py --crop Guava --image Final_images_phd/Guava/Healthy_Guava/Guava_Healthy_001.jpg
    python cli.py --crop Brinjal --soil-json '{"pH": 7.2, "Conductivity": 1200.0, "Nitrogen": 180.0}'
"""

import sys
import argparse
import json
import os

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from integration import MultimodalIntegrationSystem


def main():
    parser = argparse.ArgumentParser(description="Multimodal Agricultural Decision-Support System CLI")
    parser.add_argument("--crop", type=str, required=True, choices=["Guava", "Papaya", "Brinjal", "Cumin", "Castor"], help="Crop species")
    parser.add_argument("--image", type=str, default=None, help="Path to leaf photograph")
    parser.add_argument("--soil-json", type=str, default=None, help="JSON string or file path containing soil measurements")
    parser.add_argument("--unpaired", action="store_true", help="Flag if leaf and soil are unverified / independent samples")
    parser.add_argument("--sample-id", type=str, default="CLI_SAMPLE_001", help="Sample tracking identifier")
    parser.add_argument("--out", type=str, default=None, help="Optional output JSON path for the report")

    args = parser.parse_args()

    # Parse soil measurements
    soil_readings = None
    if args.soil_json:
        if os.path.exists(args.soil_json):
            with open(args.soil_json, "r", encoding="utf-8") as f:
                soil_readings = json.load(f)
        else:
            try:
                soil_readings = json.loads(args.soil_json)
            except Exception as e:
                print(f"[ERROR] Failed to parse --soil-json: {e}")
                sys.exit(1)

    system = MultimodalIntegrationSystem()
    result = system.process_sample(
        crop=args.crop,
        image_input=args.image,
        soil_readings=soil_readings,
        is_paired_sample=not args.unpaired,
        sample_id=args.sample_id
    )

    verdict = result["synthesized_decision"]

    print("\n" + "=" * 70)
    print("MULTIMODAL AGRICULTURAL DECISION-SUPPORT REPORT")
    print("=" * 70)
    print(f"Sample ID:             {result['sample_id']}")
    print(f"Target Crop:           {result['crop']}")
    print(f"Paired Field Sample:   {result['is_paired_sample']}")
    print(f"Overall Risk Level:    {verdict['overall_risk_level']}")
    print(f"Agronomic Summary:     {verdict['agronomic_summary']}\n")

    if result["cnn_leaf_diagnosis"] and "error" not in result["cnn_leaf_diagnosis"]:
        print("--- 1. Leaf Disease Vision Diagnosis ---")
        cnn = result["cnn_leaf_diagnosis"]
        print(f"Predicted Condition:   {cnn['predicted_class']}")
        print(f"CNN Confidence:        {cnn['confidence']}%")
        print("Top Candidates:        " + ", ".join([f"{c['class']} ({c['confidence']}%)" for c in cnn.get('top_3_predictions', [])]))
        print(f"Grad-CAM Heatmap:      {'Generated' if cnn.get('gradcam', {}).get('success') else 'Unavailable'}\n")

    if result["soil_health_diagnosis"] and "error" not in result["soil_health_diagnosis"]:
        print("--- 2. Soil Health Classification ---")
        soil = result["soil_health_diagnosis"]
        print(f"Soil Status:           {soil['Status']} (P(Healthy) = {soil['P_Healthy']})")
        print(f"Model Confidence:      {soil['Confidence']}%")
        shap_feats = soil.get("Top_SHAP_Features", [])
        if shap_feats and "error" not in shap_feats[0]:
            print("Top SHAP Drivers:      " + ", ".join([f"{f['feature']} ({round(f['shap_value'], 2)})" for f in shap_feats[:4]]))
        print()

    if verdict["warnings_and_conflicts"]:
        print("--- ⚠️ Safety Warnings & Conflict Resolutions ---")
        for w in verdict["warnings_and_conflicts"]:
            print(f"[{w['severity']}] {w['type']}: {w['message']}")
        print()

    if verdict["plant_protection_advisory"]:
        print("--- 🛡️ Plant Protection Measures (Biotic Pathogens) ---")
        for p in verdict["plant_protection_advisory"]:
            print(f"- Target:      {p['Target_Condition']}")
            print(f"  Chemical:    {p['Recommended_Protection']}")
            print(f"  Dosage:      {p['Prescribed_Dosage']}")
            print(f"  Advisory:    {p['Farmer_Advisory']}\n")

    if verdict["fertilizer_prescriptions"]:
        print("--- 🌾 Precision Fertilizer Prescriptions (Deficiencies) ---")
        for f in verdict["fertilizer_prescriptions"]:
            print(f"- Target:      {f['Nutrient']} ({f['Diagnosis_Type']})")
            print(f"  Fertilizer:  {f['Recommended_Fertilizer']}")
            print(f"  Dosage:      {f['Prescribed_Dosage']}")
            print(f"  Advisory:    {f['Farmer_Advisory']}\n")

    if verdict["soil_amendments_and_recovery"]:
        print("--- 💧 Soil Amendments & Recovery Protocols ---")
        for a in verdict["soil_amendments_and_recovery"]:
            print(f"- Action:      {a['Intervention']} ({a['Target']})")
            print(f"  Protocol:    {a['Prescription']}")
            print(f"  Advisory:    {a['Advisory']}\n")

    print("=" * 70)

    if args.out:
        # Filter non-serializable fields (like numpy arrays)
        clean_result = {
            "sample_id": result["sample_id"],
            "crop": result["crop"],
            "risk_level": verdict["overall_risk_level"],
            "summary": verdict["agronomic_summary"],
            "warnings_and_conflicts": verdict["warnings_and_conflicts"],
            "plant_protection": verdict["plant_protection_advisory"],
            "fertilizer_prescriptions": verdict["fertilizer_prescriptions"],
            "soil_amendments": verdict["soil_amendments_and_recovery"]
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(clean_result, f, indent=4)
        print(f"[OK] Report saved to: {args.out}")


if __name__ == "__main__":
    main()

