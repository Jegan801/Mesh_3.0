import os
import pickle
import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from ai.feature_builder import build_feature_vector



RAW_ROOT = "PART_02"
TRAIN_VEHICLES = [f"{i:02d}_" for i in range(1, 31)]

EXP_NAME = "exp_001"
MODEL_DIR = f"experiments/{EXP_NAME}/models"



def sanitize_features(X):
    """
    Make feature matrix numerically safe for ML.
    """
    X = np.asarray(X, dtype=np.float64)

    # Replace NaN, +inf, -inf
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6
    )

    # Clip extreme values
    X = np.clip(X, -1e6, 1e6)

    return X


# -----------------------------
# COLLECT TRAINING DATA
# -----------------------------
def collect_training_features():
    X = []

    print("🔍 Collecting training features...")

    for v in TRAIN_VEHICLES:
        v_path = os.path.join(RAW_ROOT, v)

        cad_n = os.path.join(v_path, "cad_NODE.csv")
        cad_e = os.path.join(v_path, "cad_ELEMENT.csv")

        if not os.path.exists(cad_n):
            continue

        cad = load_mesh(cad_n, cad_e)

        for f in os.listdir(v_path):
            if not f.startswith("first_mesh_") or not f.endswith("_NODE.csv"):
                continue

            base = f.replace("_NODE.csv", "")
            n_file = os.path.join(v_path, f"{base}_NODE.csv")
            e_file = os.path.join(v_path, f"{base}_ELEMENT.csv")

            if not os.path.exists(e_file):
                continue

            mesh = load_mesh(n_file, e_file)
            mesh.element_neighbors = build_element_neighbors(mesh)

            intrinsic = compute_intrinsic_metrics(mesh)
            cad_dist = compute_mesh_to_cad_distances(mesh, cad)

            for eid in mesh.elements:
                vec = build_feature_vector(
                    eid=eid,
                    mesh=mesh,
                    intrinsic_metrics=intrinsic,
                    cad_distances=cad_dist
                )
                X.append(vec)

    return np.array(X)


# -----------------------------
# MAIN
# -----------------------------
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X = collect_training_features()
    print(f"✅ Training samples: {X.shape[0]}")

    # 🔑 THIS LINE SAVES YOUR LIFE
    X = sanitize_features(X)

    # Optional debug (once)
    bad = ~np.isfinite(X).all(axis=1)
    print(f"⚠️ Invalid rows after sanitize: {bad.sum()}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )
    model.fit(Xs)

    with open(f"{MODEL_DIR}/unsupervised_iforest.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(f"{MODEL_DIR}/feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(f"{MODEL_DIR}/feature_stats.json", "w") as f:
        json.dump(
            {
                "mean": scaler.mean_.tolist(),
                "std": scaler.scale_.tolist()
            },
            f,
            indent=2
        )

    print("🎯 Unsupervised model training COMPLETE")


if __name__ == "__main__":
    main()
