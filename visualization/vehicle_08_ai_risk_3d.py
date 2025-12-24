import os
import pickle
import numpy as np
import plotly.graph_objects as go

from core.mesh_loader import load_mesh


VEHICLE = "raw/08_"
OUT_HTML = "html/vehicle_08_ai_mesh_with_index.html"


def risk_color(score, high_thr, med_thr):
    if score >= high_thr:
        return "red"
    elif score >= med_thr:
        return "orange"
    else:
        return "green"


def main():
    # -----------------------------
    # Load mesh & anomaly scores
    # -----------------------------
    mesh = load_mesh(
        os.path.join(VEHICLE, "first_mesh_1_NODE.csv"),
        os.path.join(VEHICLE, "first_mesh_1_ELEMENT.csv")
    )

    with open("models/vehicle_08_anomaly_scores.pkl", "rb") as f:
        anomaly_scores = pickle.load(f)

    scores = np.array(list(anomaly_scores.values()))
    high_thr = np.percentile(scores, 95)
    med_thr = np.percentile(scores, 85)

    # -----------------------------
    # Prepare node arrays
    # -----------------------------
    node_index = {}
    xs, ys, zs = [], [], []

    for idx, (nid, node) in enumerate(mesh.nodes.items()):
        node_index[nid] = idx
        xs.append(node.x)
        ys.append(node.y)
        zs.append(node.z)

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)

    # -----------------------------
    # Build faces
    # -----------------------------
    I, J, K = [], [], []
    face_colors = []

    for eid, elem in mesh.elements.items():
        nids = elem.node_ids
        score = anomaly_scores.get(eid, 0.0)
        color = risk_color(score, high_thr, med_thr)

        if len(nids) == 3:  # TRI
            i, j, k = [node_index[n] for n in nids]
            I.append(i); J.append(j); K.append(k)
            face_colors.append(color)

        elif len(nids) == 4:  # QUAD → 2 TRIs
            i0, i1, i2, i3 = [node_index[n] for n in nids]

            I += [i0, i0]
            J += [i1, i2]
            K += [i2, i3]

            face_colors += [color, color]

    # -----------------------------
    # Mesh plot
    # -----------------------------
    mesh_plot = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=I,
        j=J,
        k=K,
        facecolor=face_colors,
        opacity=0.95,
        name="AI Mesh Risk"
    )

    # -----------------------------
    # Legend (Index)
    # -----------------------------
    legend_traces = [
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="HIGH Risk (AI)"
        ),
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color="orange"),
            name="MEDIUM Risk (AI)"
        ),
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=10, color="green"),
            name="LOW Risk (AI)"
        )
    ]

    fig = go.Figure(data=[mesh_plot] + legend_traces)

    fig.update_layout(
        title="Vehicle 08 – First Mesh AI Risk Visualization",
        scene=dict(aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    os.makedirs("html", exist_ok=True)
    fig.write_html(OUT_HTML)

    print(f"✅ Mesh visualization with index saved → {OUT_HTML}")


if __name__ == "__main__":
    main()
