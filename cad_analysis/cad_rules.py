# =====================================================
# CAD-BASED ERROR RULES
# Converts NODE-level CAD distances into ELEMENT errors
# =====================================================

# -----------------------------
# Thresholds (tunable)
# -----------------------------
MAX_MEAN_CAD_DISTANCE = 2.5
MAX_MAX_CAD_DISTANCE = 4.0
MIN_CAD_COVERAGE = 0.7


def get_cad_errors(mesh, node_cad_distances):
    """
    Convert node-level CAD distances into element-level CAD errors.

    Args:
        mesh: Mesh object
        node_cad_distances: dict[node_id -> distance]

    Returns:
        dict: elem_id -> list of CAD error strings
    """

    cad_errors = {}

    for elem_id, elem in mesh.elements.items():
        # collect node distances for this element
        distances = [
            node_cad_distances.get(nid, 0.0)
            for nid in elem.node_ids
            if nid in node_cad_distances
        ]

        if not distances:
            continue

        mean_dist = sum(distances) / len(distances)
        max_dist = max(distances)

        # coverage = fraction of nodes within tolerance
        covered = [d for d in distances if d <= MAX_MEAN_CAD_DISTANCE]
        coverage = len(covered) / len(distances)

        elem_errors = []

        if mean_dist > MAX_MEAN_CAD_DISTANCE:
            elem_errors.append("CAD_DEVIATION_HIGH")

        if max_dist > MAX_MAX_CAD_DISTANCE:
            elem_errors.append("CAD_OUTLIER_NODE")

        if coverage < MIN_CAD_COVERAGE:
            elem_errors.append("CAD_COVERAGE_WEAK")

        if elem_errors:
            cad_errors[elem_id] = elem_errors

    print("CAD-related mesh errors detected")

    return cad_errors
