import pandas as pd
from analysis.action_mapper import map_actions


def generate_recommendations_csv(
    mesh,
    anomaly_scores,
    intrinsic_errors,
    cad_errors,
    out_csv,
    high_pct=95,
    med_pct=85
):
    """
    Generate clean element-level recommendation CSV
    driven by unsupervised AI severity.
    """

    rows = []

    scores = list(anomaly_scores.values())
    high_thr = percentile(scores, high_pct)
    med_thr = percentile(scores, med_pct)

    for elem_id, elem in mesh.elements.items():
        score = anomaly_scores.get(elem_id, 0.0)

        # -----------------------------
        # AI-driven severity
        # -----------------------------
        if score >= high_thr:
            risk = "HIGH"
        elif score >= med_thr:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        intr_errs = intrinsic_errors.get(elem_id, [])
        cad_errs = cad_errors.get(elem_id, [])

        action = map_actions(
            intrinsic_errors=intr_errs,
            cad_errors=cad_errs,
            anomaly_score=score,
            risk_level=risk
        )

        # Skip non-actionable elements
        if action["primary_action"] == "NO ACTION":
            continue

        rows.append({
            "element_id": elem_id,
            "node_ids": ",".join(map(str, elem.node_ids)),
            "ai_severity": risk,
            "primary_action": action["primary_action"],
            "confidence": action["confidence"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print(f"✅ Clean recommendations CSV saved → {out_csv}")


def percentile(data, p):
    if not data:
        return 0.0
    data = sorted(data)
    k = int(len(data) * p / 100)
    return data[min(k, len(data) - 1)]