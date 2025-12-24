import os
import pickle

from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from quality.intrinsic_rules import detect_intrinsic_errors
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from cad_analysis.cad_rules import get_cad_errors
from analysis.recommendations import generate_recommendations_csv


RAW_ROOT = "PART_02"
TEST_VEHICLES = ["31_", "32_"]

EXP_NAME = "exp_001"
OUT_DIR = f"experiments/{EXP_NAME}/outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for v in TEST_VEHICLES:
        v_path = os.path.join(RAW_ROOT, v)

        cad = load_mesh(
            f"{v_path}/cad_NODE.csv",
            f"{v_path}/cad_ELEMENT.csv"
        )
        mesh = load_mesh(
            f"{v_path}/first_mesh_1_NODE.csv",
            f"{v_path}/first_mesh_1_ELEMENT.csv"
        )

        mesh.element_neighbors = build_element_neighbors(mesh)
        intr = compute_intrinsic_metrics(mesh)
        intr_err = detect_intrinsic_errors(mesh, intr, mesh.element_neighbors)
        cad_d = compute_mesh_to_cad_distances(mesh, cad)
        cad_err = get_cad_errors(mesh, cad_d)

        with open(f"{OUT_DIR}/vehicle_{v}_anomaly_scores.pkl", "rb") as f:
            scores = pickle.load(f)

        out_csv = f"{OUT_DIR}/vehicle_{v}_recommendations.csv"
        generate_recommendations_csv(
            mesh, scores, intr_err, cad_err, out_csv
        )

        print(f"✅ Recommendations generated for {v}")


if __name__ == "__main__":
    main()
