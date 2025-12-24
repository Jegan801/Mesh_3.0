import os
import pandas as pd

from core.mesh_loader import load_mesh
from analysis.mesh_validation import validate_mesh_changes


# -----------------------------
# CONFIG
# -----------------------------
RAW_ROOT = "PART_02"
TEST_VEHICLES = ["31_", "32_"]

EXP_NAME = "exp_001"
EXP_ROOT = f"experiments/{EXP_NAME}"
OUT_DIR = f"{EXP_ROOT}/outputs"


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for vehicle in TEST_VEHICLES:
        print(f"\n🚗 ENGINEERING VALIDATION — VEHICLE {vehicle}")

        vehicle_path = os.path.join(RAW_ROOT, vehicle)

        # -----------------------------
        # Load meshes
        # -----------------------------
        mesh_init = load_mesh(
            f"{vehicle_path}/first_mesh_1_NODE.csv",
            f"{vehicle_path}/first_mesh_1_ELEMENT.csv"
        )

        mesh_final = load_mesh(
            f"{vehicle_path}/final_mesh_NODE.csv",
            f"{vehicle_path}/final_mesh_ELEMENT.csv"
        )

        cad = load_mesh(
            f"{vehicle_path}/cad_NODE.csv",
            f"{vehicle_path}/cad_ELEMENT.csv"
        )

        # -----------------------------
        # Load recommendations
        # -----------------------------
        rec_csv = f"{OUT_DIR}/vehicle_{vehicle}_recommendations.csv"
        if not os.path.exists(rec_csv):
            print(f"❌ Recommendations not found for {vehicle}")
            continue

        df = pd.read_csv(rec_csv)
        recommendations = df.to_dict(orient="records")

        # -----------------------------
        # Use ONLY MEDIUM + HIGH
        # -----------------------------
        actionable_recs = [
            r for r in recommendations
            if r["ai_severity"] in ["MEDIUM", "HIGH"]
        ]

        print(f"🔍 Actionable elements: {len(actionable_recs)}")

        if not actionable_recs:
            print("⚠️ No actionable elements found — skipping")
            continue

        # -----------------------------
        # Engineering validation
        # -----------------------------
        metrics = validate_mesh_changes(
            mesh_init,
            mesh_final,
            actionable_recs,
            cad,
            cad
        )

        print("\n📊 VALIDATION METRICS (ENGINEERING-DRIVEN)")
        print("----------------------------------------")
        print(f"Change hit rate          : {metrics['change_hit_rate']:.3f}")
        print(f"Quality improvement rate : {metrics['quality_improvement_rate']:.3f}")
        print(f"Avg change magnitude     : {metrics['avg_change_magnitude']:.3f}")


if __name__ == "__main__":
    main()
