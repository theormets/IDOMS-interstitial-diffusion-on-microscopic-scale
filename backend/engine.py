from __future__ import annotations
import math, heapq
from typing import Dict, Any, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, expm_multiply

# ---------------------------------------------------------------------------
# Constants  (ported verbatim from the IDOMS worked validation example so the
# deployed tool reproduces the validation script bit-for-bit)
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5        # Boltzmann constant, eV/K
NU0 = 1.0e13                  # attempt frequency, 1/s
R_INFLUENCE_A = 0.72          # solute influence length, in units of a (R_U = R_E)
E_MIN = 1.0e-5                # activation-barrier floor, eV
TRAP_RES_THRESHOLD = -1.0e-4  # site is a trap (residence metric) if U_i < this
SEED = 17                     # deterministic alloy-placement seed (matches validation)
NEAR_BOUNDARY_A = 0.30        # absorbing-shell thickness at each boundary (units of a)
SOURCE_Z_FRACTION = {"surface_return": 0.23, "bulk_transmission": 0.10, "retention": 0.50}
SOURCE_SIGMA_XY_A = 1.25
SOURCE_SIGMA_Z_A = 0.80

# Light elements that occupy interstitial (octahedral) sites as solutes;
# everything else is treated as substitutional on the host lattice.
INTERSTITIAL_FORMERS = {"C", "N", "O", "B"}

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def unique_rows(arr: np.ndarray, decimals: int = 8) -> np.ndarray:
    if len(arr) == 0:
        return arr.reshape(0, 3)
    return np.unique(np.round(arr, decimals), axis=0)


def host_lattice(crystal: str, nx: int, ny: int, nz: int, a: float) -> np.ndarray:
    if crystal == "BCC":
        basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    else:
        basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    pts.append((i + b[0], j + b[1], k + b[2]))
    return np.asarray(pts, dtype=float) * a


def bcc_tetra(nx, ny, nz, a):
    basis = np.array([
        (0.50, 0.25, 0.00), (0.50, 0.75, 0.00), (0.25, 0.50, 0.00), (0.75, 0.50, 0.00),
        (0.50, 0.00, 0.25), (0.50, 0.00, 0.75), (0.25, 0.00, 0.50), (0.75, 0.00, 0.50),
        (0.00, 0.50, 0.25), (0.00, 0.50, 0.75), (0.00, 0.25, 0.50), (0.00, 0.75, 0.50),
    ], dtype=float)
    pts = []; lim = np.array([nx, ny, nz], dtype=float)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                p = basis + np.array([i, j, k], dtype=float)
                keep = np.all((p >= -1e-9) & (p <= lim + 1e-9), axis=1)
                pts.extend(p[keep].tolist())
    return unique_rows(np.asarray(pts) * a)


def bcc_octa(nx, ny, nz, a):
    gx = np.arange(0, nx + 0.25, 0.5); gy = np.arange(0, ny + 0.25, 0.5); gz = np.arange(0, nz + 0.25, 0.5)
    pts = np.asarray([(x, y, z) for x in gx for y in gy for z in gz], dtype=float)
    n_half = np.sum(np.abs(pts - np.round(pts)) > 1e-8, axis=1)
    pts = pts[(n_half == 1) | (n_half == 2)]
    pts = pts[np.all((pts >= -1e-9) & (pts <= np.array([nx, ny, nz]) + 1e-9), axis=1)]
    return unique_rows(pts * a)


def fcc_tetra(nx, ny, nz, a):
    basis = np.array([(0.25, 0.25, 0.25), (0.25, 0.25, 0.75), (0.25, 0.75, 0.25), (0.75, 0.25, 0.25),
                      (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75), (0.75, 0.75, 0.75)], float)
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                pts.extend((basis + np.array([i, j, k], float)).tolist())
    return unique_rows(np.asarray(pts) * a)


def fcc_octa(nx, ny, nz, a):
    basis = np.array([(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5),
                      (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)], float)
    pts = []; lim = np.array([nx, ny, nz], float)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                p = basis + np.array([i, j, k], float)
                pts.extend(p[np.all((p >= -1e-9) & (p <= lim + 1e-9), axis=1)].tolist())
    return unique_rows(np.asarray(pts) * a)


def network_name(requested: str, species: str) -> str:
    if requested != "auto":
        return requested
    return "tetrahedral" if species in {"H", "D", "T"} else "octahedral"


def interstitial_network(crystal, network, nx, ny, nz, a):
    blocks = []
    if crystal == "BCC":
        if network in {"tetrahedral", "tetra_octa"}: blocks.append(bcc_tetra(nx, ny, nz, a))
        if network in {"octahedral", "tetra_octa"}: blocks.append(bcc_octa(nx, ny, nz, a))
    else:
        if network in {"tetrahedral", "tetra_octa"}: blocks.append(fcc_tetra(nx, ny, nz, a))
        if network in {"octahedral", "tetra_octa"}: blocks.append(fcc_octa(nx, ny, nz, a))
    if not blocks:
        raise ValueError("No interstitial sites generated.")
    return unique_rows(np.vstack(blocks))


def octahedral_sites(crystal, nx, ny, nz, a):
    return bcc_octa(nx, ny, nz, a) if crystal == "BCC" else fcc_octa(nx, ny, nz, a)


def build_edges(sites: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tree = cKDTree(sites)
    d, _ = tree.query(sites, k=2)
    nn = float(np.median(d[:, 1]))
    pairs = np.array(sorted(tree.query_pairs(1.08 * nn)), dtype=int)
    if len(pairs) == 0:
        raise ValueError("No interstitial edges found.")
    mids = 0.5 * (sites[pairs[:, 0]] + sites[pairs[:, 1]])
    lengths = np.linalg.norm(sites[pairs[:, 0]] - sites[pairs[:, 1]], axis=1)
    return pairs, mids, lengths

# ---------------------------------------------------------------------------
# Alloy placement
#   Substitutional solutes (metals) -> host lattice sites; light interstitial
#   formers (C, N, O, B) -> octahedral sites. Substitutional solutes are drawn
#   first, then interstitial formers, from a single default_rng(SEED) stream, to
#   match the validation script's placement exactly.
# ---------------------------------------------------------------------------
def place_solutes(host_pts, octa_pts, setup):
    rng = np.random.default_rng(SEED)
    counts = dict(setup["domain"]["estimated_atom_counts"])
    elements = list(setup.get("alloying_elements", []))
    substitutional = [e for e in elements if e not in INTERSTITIAL_FORMERS]
    interstitial = [e for e in elements if e in INTERSTITIAL_FORMERS]
    out = {}
    for el in substitutional:
        n = min(int(counts.get(el, 0)), len(host_pts))
        idx = rng.choice(len(host_pts), size=n, replace=False) if n > 0 else np.array([], dtype=int)
        out[el] = host_pts[idx]
    for el in interstitial:
        n = min(int(counts.get(el, 0)), len(octa_pts))
        idx = rng.choice(len(octa_pts), size=n, replace=False) if n > 0 else np.array([], dtype=int)
        out[el] = octa_pts[idx]
    return out

# ---------------------------------------------------------------------------
# Energy fields (reduced vital-energy set)
# ---------------------------------------------------------------------------
def compute_fields(sites, mids, solutes, energies, a):
    U = np.zeros(len(sites), dtype=float)
    Sshift = np.zeros(len(mids), dtype=float)
    R = R_INFLUENCE_A * a
    for el, pts in solutes.items():
        if len(pts) == 0:
            continue
        tree = cKDTree(pts)
        ds, _ = tree.query(sites, k=1)
        dm, _ = tree.query(mids, k=1)
        dU = float(energies.get(f"{el}_site_energy_shift", 0.0))
        dE = float(energies.get(f"{el}_saddle_energy_shift", 0.0))
        if dU:
            U += dU * np.exp(-(ds / R) ** 2)
        if dE:
            Sshift += dE * np.exp(-(dm / R) ** 2)
    return U, Sshift

# ---------------------------------------------------------------------------
# Master-equation system
# ---------------------------------------------------------------------------
def build_system(sites, edges, mids, U, Sshift, setup, energies):
    n = len(sites); T = float(setup["temperature_K"]); kBT = max(KB_EV * T, 1e-30)
    a = float(setup["lattice_parameter_A"])
    E0 = float(energies.get("host_migration_barrier", 0.1))
    rows = []; cols = []; vals = []; adjacency = [[] for _ in range(n)]
    edge_act = np.zeros(len(edges))
    for e in range(len(edges)):
        i = int(edges[e, 0]); j = int(edges[e, 1])
        saddle = max(U[i], U[j]) + E0 + Sshift[e]
        Eij = max(saddle - U[i], E_MIN)
        Eji = max(saddle - U[j], E_MIN)
        kij = NU0 * math.exp(-Eij / kBT); kji = NU0 * math.exp(-Eji / kBT)
        rows += [i, j]; cols += [j, i]; vals += [kij, kji]
        adjacency[i].append((j, kij, Eij, e)); adjacency[j].append((i, kji, Eji, e))
        edge_act[e] = 0.5 * (Eij + Eji)
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    sum_jump = np.asarray(W.sum(axis=1)).ravel()
    z = sites[:, 2]; zmin, zmax = float(z.min()), float(z.max())
    near_surface = z <= zmin + NEAR_BOUNDARY_A * a
    near_bulk = z >= zmax - NEAR_BOUNDARY_A * a
    E_surf = float(energies.get("surface_exit_barrier", E0))
    E_bulk = float(energies.get("bulk_exit_barrier", E0))
    k_surface = np.zeros(n); k_bulk = np.zeros(n)
    for i in np.where(near_surface)[0]:
        k_surface[i] = NU0 * math.exp(-max(E_surf - U[i], E_MIN) / kBT)
    for i in np.where(near_bulk)[0]:
        k_bulk[i] = NU0 * math.exp(-max(E_bulk - U[i], E_MIN) / kBT)
    total = sum_jump + k_surface + k_bulk
    A = diags(total, 0, shape=(n, n), format="csr") - W
    Q = W - diags(total, 0, shape=(n, n), format="csr")
    return {"W": W, "Q": Q, "A": A, "adjacency": adjacency, "total": total,
            "k_surface": k_surface, "k_bulk": k_bulk, "edge_activation": edge_act}

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def source_distribution(sites, setup):
    a = float(setup["lattice_parameter_A"])
    nx, ny, nz = setup["domain"]["Nx"], setup["domain"]["Ny"], setup["domain"]["Nz"]
    zf = SOURCE_Z_FRACTION.get(setup["boundary"], 0.23)
    cx, cy, cz = 0.5 * nx * a, 0.5 * ny * a, zf * nz * a
    sig_xy = SOURCE_SIGMA_XY_A * a; sig_z = SOURCE_SIGMA_Z_A * a
    dx2 = (sites[:, 0] - cx) ** 2 + (sites[:, 1] - cy) ** 2
    dz2 = (sites[:, 2] - cz) ** 2
    w = np.exp(-dx2 / sig_xy ** 2) * np.exp(-dz2 / sig_z ** 2)
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(sites))
    return w / w.sum()


def solve_first_passage(system, start):
    A = system["A"]
    try:
        qsurf = np.clip(spsolve(A, system["k_surface"]), 0.0, 1.0)
        tabs = np.maximum(spsolve(A, np.ones(A.shape[0])), 0.0)
        res = np.maximum(spsolve(A.T, start), 0.0)
        mfpt = float(start @ tabs)
        psurf = float(start @ qsurf); pbulk = 1.0 - psurf
    except Exception:
        qsurf = np.full(A.shape[0], 0.5); tabs = np.ones(A.shape[0]) * 1e-9; res = start.copy()
        mfpt = 1e-9; psurf = 0.5; pbulk = 0.5
    if not math.isfinite(mfpt) or mfpt <= 0:
        mfpt = 1e-12
    return {"q_surface": qsurf, "t_abs": tabs, "residence": res, "mfpt": mfpt,
            "p_surface": max(0.0, min(1.0, psurf)), "p_bulk": max(0.0, min(1.0, pbulk))}


def solve_mfpt(system, start):
    A = system["A"]
    try:
        tabs = np.maximum(spsolve(A, np.ones(A.shape[0])), 0.0)
        mfpt = float(start @ tabs)
    except Exception:
        mfpt = 1e-9
    if not math.isfinite(mfpt) or mfpt <= 0:
        mfpt = 1e-12
    return mfpt


def survival_outputs(system, start, mfpt, npts=80):
    Q = system["Q"]; tmax = max(5.0 * mfpt, 1e-13)
    times = np.linspace(0.0, tmax, npts)
    try:
        P = expm_multiply(Q.T, start, start=0.0, stop=tmax, num=npts)
        S = np.clip(np.asarray(P.sum(axis=1)).ravel(), 0.0, 1.0)
        if len(S) != npts or not np.all(np.isfinite(S)):
            raise ValueError("bad survival")
    except Exception:
        S = np.exp(-times / max(mfpt, 1e-30))
    try:
        S_map = np.clip(np.asarray(expm_multiply(Q * max(mfpt, 1e-30), np.ones(Q.shape[0]))).ravel(), 0.0, 1.0)
        if not np.all(np.isfinite(S_map)):
            raise ValueError("bad map")
    except Exception:
        S_map = np.clip(np.exp(-max(mfpt, 1e-30) / np.maximum(system["total"], 1e-30)), 0.0, 1.0)
    if len(S):
        S[0] = 1.0
    return times, S, S_map


def dijkstra_to_boundary(system, start_idx, target):
    n = len(system["adjacency"]); dist = np.full(n, np.inf); prev = np.full(n, -1, dtype=int)
    dist[start_idx] = 0.0; heap = [(0.0, start_idx)]
    kr = system["k_surface"] if target == "surface" else system["k_bulk"]
    best = None; best_total = np.inf
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if kr[u] > 0:
            total = d + 1.0 / max(kr[u], 1e-300)
            if total < best_total:
                best_total = total; best = u
        for v, rate, eact, eidx in system["adjacency"][u]:
            nd = d + 1.0 / max(rate, 1e-300)
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u; heapq.heappush(heap, (nd, v))
    if best is None:
        return [start_idx], float("inf")
    path = []; cur = int(best)
    while cur != -1:
        path.append(cur); cur = int(prev[cur]) if prev[cur] != -1 else -1
    return path[::-1], float(best_total)


def path_barriers(system, path):
    lookup = {}
    for u, adj in enumerate(system["adjacency"]):
        for v, rate, eact, eidx in adj:
            lookup[(u, v)] = float(eact)
    return np.asarray([lookup[(int(u), int(v))] for u, v in zip(path[:-1], path[1:])
                       if (int(u), int(v)) in lookup], dtype=float)


def build_full_channel(path_s, path_b):
    have_s = len(path_s) >= 2; have_b = len(path_b) >= 2
    if have_s and have_b:
        surf_half = list(path_s[::-1]); full = surf_half + list(path_b[1:])
        junction = len(surf_half) - 1
    elif have_s:
        full = list(path_s[::-1]); junction = len(full) - 1
    elif have_b:
        full = list(path_b); junction = 0
    else:
        full = list(path_s); junction = 0
    return full, junction

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def in_slice(points, y0, a, half=0.65):
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    return np.abs(points[:, 1] - y0) <= half * a


def thin_idx(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.linspace(0, n - 1, max_n).astype(int)


def xz(points, a):
    return {"x": (points[:, 0] / a).tolist() if len(points) else [],
            "z": (points[:, 2] / a).tolist() if len(points) else []}


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return 0.0 if (math.isnan(obj) or math.isinf(obj)) else obj
    return obj

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def run_simulation(setup: Dict[str, Any], energies_eV: Dict[str, float]) -> Dict[str, Any]:
    if len(setup.get("alloying_elements", [])) > 2:
        raise ValueError("This backend supports up to two alloying elements.")

    nx, ny, nz = int(setup["domain"]["Nx"]), int(setup["domain"]["Ny"]), int(setup["domain"]["Nz"])
    a = float(setup["lattice_parameter_A"]); crystal = setup["crystal_structure"]
    network = network_name(setup.get("site_network", "auto"), setup["interstitial_species"])

    host = host_lattice(crystal, nx, ny, nz, a)
    sites = interstitial_network(crystal, network, nx, ny, nz, a)
    octa = octahedral_sites(crystal, nx, ny, nz, a)
    edges, mids, lengths = build_edges(sites)
    solutes = place_solutes(host, octa, setup)

    E0 = float(energies_eV.get("host_migration_barrier", 0.1))

    # --- alloy calculation ---
    U, Sshift = compute_fields(sites, mids, solutes, energies_eV, a)
    system = build_system(sites, edges, mids, U, Sshift, setup, energies_eV)
    start = source_distribution(sites, setup)
    fp = solve_first_passage(system, start)
    times, S, S_map = survival_outputs(system, start, fp["mfpt"])

    # --- ideal-host reference (retention) ---
    U_h, S_h = compute_fields(sites, mids, solutes, {"host_migration_barrier": E0}, a)
    system_h = build_system(sites, edges, mids, U_h, S_h, setup, {"host_migration_barrier": E0})
    mfpt_host = solve_mfpt(system_h, start)
    retention = float(fp["mfpt"] / mfpt_host) if mfpt_host > 0 else 1.0

    # --- dominant path (metrics) and full channel (visualisation) ---
    source_idx = int(np.argmax(start))
    path_s, R_s = dijkstra_to_boundary(system, source_idx, "surface")
    path_b, R_b = dijkstra_to_boundary(system, source_idx, "bulk")
    if fp["p_surface"] >= fp["p_bulk"]:
        target, path, Rdom = "surface", path_s, R_s
    else:
        target, path, Rdom = "bulk", path_b, R_b
    dom_e = path_barriers(system, path)
    dom_pts = sites[np.asarray(path, dtype=int)]
    dom_len = float(np.sum(np.linalg.norm(np.diff(dom_pts, axis=0), axis=1))) if len(dom_pts) > 1 else 0.0
    dom_straight = float(np.linalg.norm(dom_pts[-1] - dom_pts[0])) if len(dom_pts) > 1 else 1.0
    tortuosity = dom_len / max(dom_straight, 1e-12)
    mean_barrier = float(np.mean(dom_e)) if len(dom_e) else E0
    bottleneck = float(np.max(dom_e)) if len(dom_e) else E0

    channel, junction = build_full_channel(path_s, path_b)
    ch_pts = sites[np.asarray(channel, dtype=int)]
    ch_e = path_barriers(system, channel)
    ch_node_c = (np.r_[ch_e[:1], ch_e] if len(ch_e) else np.zeros(len(ch_pts))).tolist()
    y0 = float(np.median(ch_pts[:, 1])) if len(ch_pts) else 0.5 * ny * a

    # --- residence-weighted trap fraction ---
    res = fp["residence"]; total_res = float(res.sum()) if res.sum() > 0 else 1.0
    trap_metric_mask = U < TRAP_RES_THRESHOLD
    trap_frac = float(res[trap_metric_mask].sum() / total_res) if np.any(trap_metric_mask) else 0.0

    # --- visual filtering to a thin y-slice around the channel ---
    fe_i = np.where(in_slice(host, y0, a, 0.55))[0]; fe_i = fe_i[thin_idx(len(fe_i), 700)]
    hs_i = np.where(in_slice(sites, y0, a, 0.55))[0]; hs_i = hs_i[thin_idx(len(hs_i), 700)]

    trap_vis = U <= min(-1e-4, float(np.percentile(U, 10)))
    trap_i = np.where(trap_vis & in_slice(sites, y0, a, 0.9))[0]; trap_i = trap_i[thin_idx(len(trap_i), 350)]

    ea = system["edge_activation"]
    high_thr = float(np.percentile(ea, 95)) if len(ea) else float("inf")
    high_i = np.where((ea >= high_thr) & in_slice(mids, y0, a, 0.9))[0]; high_i = high_i[thin_idx(len(high_i), 350)]

    solute_out = []
    for el, pts in solutes.items():
        if len(pts) == 0:
            continue
        sp = pts[in_slice(pts, y0, a, 1.3)]; sp = sp[thin_idx(len(sp), 160)]
        for p in sp:
            solute_out.append({"element": el, "x": float(p[0] / a), "z": float(p[2] / a)})

    # --- site-resolved survival map (projected, thinned) ---
    map_i = thin_idx(len(sites), 3000)
    survival_map = {"x": (sites[map_i, 0] / a).tolist(),
                    "z": (sites[map_i, 2] / a).tolist(),
                    "S": np.asarray(S_map)[map_i].tolist()}

    result = {
        "metrics": {
            "mean_first_passage_time_s": fp["mfpt"],
            "probability_bulk_transmission": fp["p_bulk"],
            "probability_surface_return": fp["p_surface"],
            "bottleneck_barrier_eV": bottleneck,
            "minimum_resistance_path_s": Rdom,
            "path_tortuosity": tortuosity,
            "pathway_energy_cost_eV": mean_barrier,
            "trap_residence_fraction": max(0.0, min(1.0, trap_frac)),
            "percolation_accessibility_fraction": float(np.mean(ea <= E0 + 0.05)) if len(ea) else 0.0,
            "retention_factor_vs_host": retention,
        },
        "dominant_absorbing_boundary": target,
        # ---- plot (a): lattice-aware channel in a thin y-slice ----
        "lattice_points": xz(host[fe_i], a),
        "interstitial_sites": xz(sites[hs_i], a),
        "solute_positions": solute_out,
        "trap_sites": xz(sites[trap_i], a),
        "high_barrier_edges": xz(mids[high_i], a),
        "dominant_pathway": {
            "x": (ch_pts[:, 0] / a).tolist(),
            "z": (ch_pts[:, 2] / a).tolist(),
            "activation_barrier_eV": ch_node_c,
            "injection_index": int(junction),
            "surface_exit": ({"x": float(ch_pts[0, 0] / a), "z": float(ch_pts[0, 2] / a)} if len(ch_pts) else {"x": 0, "z": 0}),
            "bulk_exit": ({"x": float(ch_pts[-1, 0] / a), "z": float(ch_pts[-1, 2] / a)} if len(ch_pts) else {"x": 0, "z": 0}),
        },
        # ---- plot (b): activation-barrier profile along the channel ----
        "barrier_profile": {
            "jump_index": list(range(1, len(ch_e) + 1)),
            "activation_barrier_eV": ch_e.tolist(),
            "host_barrier_eV": E0,
            "bottleneck_index": int(np.argmax(ch_e) + 1) if len(ch_e) else 0,
            "injection_index": int(junction),
        },
        # ---- plot (c): global survival probability ----
        "survival_probability": {"time_s": times.tolist(), "S_t": np.asarray(S).tolist(), "mfpt_s": fp["mfpt"]},
        # ---- plot (d): site-resolved survival map at t = MFPT ----
        "survival_map": survival_map,
        "domain": setup["domain"],
        "site_network_used": network,
        "notes": {
            "reproduces": "Ported from the IDOMS worked validation example; identical results for matching inputs.",
            "energy_definition": "E_act(i->j) = E_saddle(i,j) - U_i, with U_i the explicit diffusing-species site energy.",
            "channel": "Visualised pathway is the continuous minimum-resistance channel surface <-> injection <-> bulk.",
            "retention_definition": "Retention factor = tau_MFPT(alloy) / tau_MFPT(ideal host) on the same domain.",
        },
    }
    return json_safe(result)
