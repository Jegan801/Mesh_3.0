import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from ml.feature_builder import build_feature_vector
from core.mesh_loader import load_mesh
from core.mesh_neighbors import build_element_neighbors
from quality.intrinsic_metrics import compute_intrinsic_metrics
from quality.intrinsic_rules import detect_intrinsic_errors
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances
from cad_analysis.cad_rules import detect_cad_related_errors

# Load model
model = joblib.load("model_2/severity_model.pkl")

# Load data
mesh = load_mesh(
    "data/first_mesh/first_mesh_2_NODE.csv",
    "data/first_mesh/first_mesh_2_ELEMENT.csv"
)
cad = load_mesh(
    "data/cad/cad_NODE.csv",
    "data/cad/cad_ELEMENT.csv"
)

# Compute features
mesh.element_neighbors = build_element_neighbors(mesh)
metrics = compute_intrinsic_metrics(mesh)
intrinsic_errors = detect_intrinsic_errors(mesh, metrics, mesh.element_neighbors)
distances = compute_mesh_to_cad_distances(mesh, cad)
cad_errors = detect_cad_related_errors(mesh, distances)

X = []
y_true = []

# Rule-based severity = ground truth
for eid in mesh.elements:
    features = build_feature_vector(
        eid, mesh, metrics, intrinsic_errors, cad_errors
    )
    X.append(features)

    # Rule-based label (teacher)
    if "BAD_ASPECT_RATIO" in intrinsic_errors.get(eid, []) or \
       "HIGH_SKEWNESS" in intrinsic_errors.get(eid, []) or \
       "CAD_DEVIATION_HIGH" in cad_errors.get(eid, []):
        y_true.append(2)   # HIGH
    elif "BAD_TRANSITION" in intrinsic_errors.get(eid, []):
        y_true.append(1)   # MEDIUM
    else:
        y_true.append(0)   # LOW

X = np.array(X)
y_true = np.array(y_true)

# Predict
y_pred = model.predict(X)

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
