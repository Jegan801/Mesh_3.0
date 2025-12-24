def map_actions(
    intrinsic_errors,
    cad_errors,
    anomaly_score,
    risk_level
):
    """
    AI-aware action mapper.
    Produces one clear engineering action.
    """

    actions = []
    reasons = []

    all_errors = set(intrinsic_errors) | set(cad_errors)

    # -----------------------------
    # HIGH RISK
    # -----------------------------
    if risk_level == "HIGH":
        if (
            "BAD_ASPECT_RATIO" in all_errors
            or "HIGH_SKEWNESS" in all_errors
            or anomaly_score > 0.15
        ):
            actions.append("DELETE & REMESH")
            reasons.append("Severely distorted element")

        elif "CAD_DEVIATION_HIGH" in all_errors:
            actions.append("MOVE NODES TO CAD")
            reasons.append("Large CAD deviation")

    # -----------------------------
    # MEDIUM RISK
    # -----------------------------
    elif risk_level == "MEDIUM":
        if "BAD_TRANSITION" in all_errors or "SMALL_AREA" in all_errors:
            actions.append("REFINE LOCALLY")
            reasons.append("Poor mesh transition")

        if "CAD_DEVIATION_HIGH" in all_errors:
            actions.append("MOVE NODES TO CAD")
            reasons.append("CAD mismatch")

    # -----------------------------
    # LOW RISK
    # -----------------------------
    else:
        if all_errors:
            actions.append("MONITOR")
            reasons.append("Minor deviations detected")
        else:
            actions.append("NO ACTION")
            reasons.append("Mesh within learned distribution")

    # -----------------------------
    # Connectivity overrides (highest priority)
    # -----------------------------
    if "MISSING_NEIGHBOR" in all_errors or "ORPHAN_NODE" in all_errors:
        actions = ["ADD CONNECTIVITY"]
        reasons = ["Mesh connectivity issue detected"]

    # -----------------------------
    # Fallback safety
    # -----------------------------
    confidence = min(0.95, 0.55 + anomaly_score)

    if not actions:
        actions.append("REVIEW MANUALLY")
        reasons.append("Unclassified anomaly pattern detected")
        confidence = 0.6

    return {
        "primary_action": actions[0],
        "reasons": reasons,
        "confidence": round(confidence, 2)
    }
