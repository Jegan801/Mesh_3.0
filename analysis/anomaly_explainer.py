import numpy as np


def classify_risk(anomaly_score, percentile):
    if percentile >= 95:
        return "HIGH"
    elif percentile >= 85:
        return "MEDIUM"
    else:
        return "LOW"


def explain_element(eid, anomaly_score, intrinsic, cad_info, neighbors):
    reasons = []
    actions = []

    # CAD deviation
    if cad_info > 1.5:
        reasons.append("Extreme CAD deviation")
        actions.append("MOVE NODES TO CAD")

    # Geometry
    if intrinsic["aspect_ratio"] > 3.0 or intrinsic.get("skewness_proxy", 0) > 3.0:
        reasons.append("Poor element shape")
        actions.append("DELETE & REMESH")

    # Topology
    if len(neighbors) < 2:
        reasons.append("Missing connectivity")
        actions.append("ADD CONNECTIVITY")

    if not reasons:
        reasons.append("Within learned distribution")
        actions.append("NO ACTION")

    return reasons, actions


def suggest_fix(action):
    mapping = {
        "MOVE NODES TO CAD": "Project nodes onto CAD surface",
        "DELETE & REMESH": "Delete element and remesh region",
        "ADD CONNECTIVITY": "Fix holes or reconnect mesh",
        "NO ACTION": "Accept mesh as-is"
    }
    return mapping.get(action, "Manual review required")
