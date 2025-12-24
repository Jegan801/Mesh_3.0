import numpy as np


def build_feature_vector(eid, mesh, intrinsic_metrics, cad_distances):
    """
    Builds a pure, unsupervised feature vector for ONE mesh element.

    NO rule errors
    NO severity
    NO labels

    Returns: List[float]
    """

    m = intrinsic_metrics[eid]

    # --- Geometry features ---
    area = m["area"]
    aspect_ratio = m["aspect_ratio"]
    skewness = m.get("skewness_proxy", m["aspect_ratio"])

    edges = m.get("edges", [])
    min_edge = min(edges) if edges else 0.0
    max_edge = max(edges) if edges else 0.0

    # --- Topology ---
    neighbors = mesh.element_neighbors.get(eid, [])
    num_neighbors = len(neighbors)

    # --- CAD relation ---
    cad_info = cad_distances.get(eid, 0.0)

# Handle both float and dict formats
    if isinstance(cad_info, dict):
        cad_avg = cad_info.get("mean", 0.0)
        cad_max = cad_info.get("max", cad_avg)
        cad_cov = cad_info.get("coverage", 0.0)
    else:
        cad_avg = float(cad_info)
        cad_max = float(cad_info)
        cad_cov = 0.0


    return [
        area,
        aspect_ratio,
        skewness,
        min_edge,
        max_edge,
        num_neighbors,
        cad_avg,
        cad_max,
        cad_cov
    ]


def build_feature_matrix(mesh, intrinsic_metrics, cad_distances):
    """
    Builds feature matrix for ALL elements in a mesh.
    Used by unsupervised training.
    """

    X = []

    for eid in mesh.elements:
        vec = build_feature_vector(
            eid=eid,
            mesh=mesh,
            intrinsic_metrics=intrinsic_metrics,
            cad_distances=cad_distances
        )
        X.append(vec)

    return np.array(X)
