import os
import pickle
import numpy as np

from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from quality.intrinsic_rules import detect_intrinsic_errors
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances


# Safe CAD rules import (your project-compatible way)
try:
    from cad_analysis.cad_rules import get_cad_errors
except ImportError:
    get_cad_errors = None


VEHICLE = "PART_02/31_"


def main():
    # ---------------- Load meshes ----------------
    cad = load_mesh(
        os.path.join(VEHICLE, "cad_NODE.csv"),
        os.path.join(VEHICLE, "cad_ELEMENT.csv")
    )

    mesh = load_mesh(
        os.path.join(VEHICLE, "first_mesh_1_NODE.csv"),
        os.path.join(VEHICLE, "first_mesh_1_ELEMENT.csv")
    )

    # ---------------- Build neighbors ----------------
    neighbors = build_element_neighbors(mesh)
    mesh.element_neighbors = neighbors

    # ---------------- Intrinsic errors ----------------
    intrinsic_metrics = compute_intrinsic_metrics(mesh)
    intrinsic_errors = detect_intrinsic_errors(
        mesh,
        intrinsic_metrics,
        neighbors
    )

    # ---------------- CAD errors ----------------
    cad_dist = compute_mesh_to_cad_distances(mesh, cad)

    if get_cad_errors:
        cad_errors = get_cad_errors(mesh, cad_dist)
    else:
        cad_errors = {}

    rule_flagged = set(intrinsic_errors.keys()) | set(cad_errors.keys())

    # ---------------- AI anomaly scores ----------------
    with open("model_2/vehicle_08_anomaly_scores.pkl", "rb") as f:
        anomaly_scores = pickle.load(f)

    scores = np.array(list(anomaly_scores.values()))
    threshold = np.percentile(scores, 85)

    ai_flagged = {
        eid for eid, score in anomaly_scores.items()
        if score >= threshold
    }

    overlap = rule_flagged & ai_flagged

    # ---------------- Report ----------------
    print("\n📊 AI–RULE AGREEMENT REPORT (Vehicle 08)")
    print("--------------------------------------")
    print(f"Rule-flagged elements : {len(rule_flagged)}")
    print(f"AI high-risk elements : {len(ai_flagged)}")
    print(f"Overlap (agreement)   : {len(overlap)}")

    if rule_flagged:
        print(f"Agreement rate        : {len(overlap) / len(rule_flagged):.2%}")
    else:
        print("Agreement rate        : N/A")

    if ai_flagged:
        print(f"Precision@AI          : {len(overlap) / len(ai_flagged):.2%}")
    else:
        print("Precision@AI          : N/A")


if __name__ == "__main__":
    main()
