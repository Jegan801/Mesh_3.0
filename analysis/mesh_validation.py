import numpy as np
from quality.intrinsic_metrics import compute_intrinsic_metrics
from cad_analysis.cad_mesh_distance import compute_mesh_to_cad_distances


def element_centroid(mesh, elem):
    """Element centroid."""
    pts = [mesh.nodes[n].coords() for n in elem.node_ids]
    return np.mean(pts, axis=0)


def find_region_elements(mesh, center, radius=5.0):
    """Elements near a point."""
    region = []
    for eid, elem in mesh.elements.items():
        try:
            c = element_centroid(mesh, elem)
            if np.linalg.norm(c - center) <= radius:
                region.append(eid)
        except KeyError:
            continue
    return region


def average_quality(elem_ids, quality_map, keys=("aspect_ratio",)):
    """Average quality in region."""
    vals = {k: [] for k in keys}

    for eid in elem_ids:
        if eid not in quality_map:
            continue
        for k in keys:
            if k in quality_map[eid]:
                vals[k].append(quality_map[eid][k])

    return {
        k: np.mean(v) if v else None
        for k, v in vals.items()
    }


def element_changed(elem_id, mesh_init, mesh_final, tol=1e-3):
    """Detect topology or geometry change."""

    if elem_id not in mesh_final.elements:
        return True

    elem_i = mesh_init.elements[elem_id]
    elem_f = mesh_final.elements[elem_id]

    if set(elem_i.node_ids) != set(elem_f.node_ids):
        return True

    for nid in elem_i.node_ids:
        if nid not in mesh_final.nodes:
            return True

        p_i = np.array(mesh_init.nodes[nid].coords())
        p_f = np.array(mesh_final.nodes[nid].coords())

        if np.linalg.norm(p_i - p_f) > tol:
            return True

    return False


def validate_mesh_changes(
    mesh_init,
    mesh_final,
    recommendations,
    cad_init,
    cad_final,
    region_radius=5.0
):
    """Region-based validation for actionable elements."""

    q_init = compute_intrinsic_metrics(mesh_init)
    q_final = compute_intrinsic_metrics(mesh_final)

    compute_mesh_to_cad_distances(mesh_init, cad_init)
    compute_mesh_to_cad_distances(mesh_final, cad_final)

    total = 0
    changed = 0
    improved = 0
    change_magnitudes = []

    for rec in recommendations:
        elem_id = rec["element_id"]
        total += 1

        if element_changed(elem_id, mesh_init, mesh_final):
            changed += 1

            elem = mesh_init.elements.get(elem_id)
            if elem:
                coords_i = []
                coords_f = []
                remeshed = False

                for nid in elem.node_ids:
                    if nid not in mesh_final.nodes:
                        remeshed = True
                        break
                    coords_i.append(mesh_init.nodes[nid].coords())
                    coords_f.append(mesh_final.nodes[nid].coords())

                if not remeshed:
                    c0 = np.mean(coords_i, axis=0)
                    c1 = np.mean(coords_f, axis=0)
                    change_magnitudes.append(
                        np.linalg.norm(c1 - c0)
                    )

        # Region-based quality improvement
        elem = mesh_init.elements.get(elem_id)
        if elem:
            center = element_centroid(mesh_init, elem)

            region_i = find_region_elements(
                mesh_init, center, region_radius
            )
            region_f = find_region_elements(
                mesh_final, center, region_radius
            )

            qi = average_quality(region_i, q_init)
            qf = average_quality(region_f, q_final)

            if (
                qi["aspect_ratio"] is not None
                and qf["aspect_ratio"] is not None
                and qf["aspect_ratio"] < qi["aspect_ratio"]
            ):
                improved += 1

    return {
        "change_hit_rate": changed / max(total, 1),
        "quality_improvement_rate": improved / max(total, 1),
        "avg_change_magnitude": (
            float(np.mean(change_magnitudes))
            if change_magnitudes else 0.0
        ),
    }
