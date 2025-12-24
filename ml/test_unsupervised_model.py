import os
import pickle
import numpy as np

from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from ai.feature_builder import build_feature_vector


RAW_ROOT = "PART_02"
TEST_VEHICLES = ["31_", "32_"]

EXP_NAME = "exp_001"
MODEL_DIR = f"experiments/exp_001/models"
OUT_DIR = f"experiments/{EXP_NAME}/outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{MODEL_DIR}/unsupervised_iforest.pkl", "rb") as f:
        model = pickle.load(f)

    with open(f"{MODEL_DIR}/feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    for v in TEST_VEHICLES:
        print(f"\n🚗 Testing vehicle {v}")

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
        cad_d = compute_mesh_to_cad_distances(mesh, cad)

        X, eids = [], []
        for eid in mesh.elements:
            X.append(
                build_feature_vector(
                    eid, mesh, intr, cad_d
                )
            )
            eids.append(eid)

        Xs = scaler.transform(np.array(X))
        scores = -model.decision_function(Xs)

        out = dict(zip(eids, scores))
        with open(f"{OUT_DIR}/vehicle_{v}_anomaly_scores.pkl", "wb") as f:
            pickle.dump(out, f)

        print(f"📦 Saved anomaly scores for {v}")


if __name__ == "__main__":
    main()
