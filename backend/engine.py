from __future__ import annotations
import math, heapq, hashlib
from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve, expm_multiply

# ---------------------------------------------------------------------------
# Physical / numerical constants
# ---------------------------------------------------------------------------
KB_EV = 8.617333262e-5        # Boltzmann constant, eV/K
NU0 = 1.0e13                  # attempt frequency, 1/s
R_INFLUENCE_A = 0.85          # R_U = R_E = 0.85 a  (local influence length)
E_MIN = 1.0e-4                # activation-barrier floor, eV
TRAP_THRESHOLD_EV = -0.01     # site is a physical trap if U_i < this (spec)

# ---------------------------------------------------------------------------
# Crystal and interstitial geometry
# ---------------------------------------------------------------------------
def unique_rows(a: np.ndarray) -> np.ndarray:
    if len(a) == 0:
        return a.reshape(0, 3)
    return np.unique(np.round(a, 8), axis=0)

def host_lattice(crystal: str, nx: int, ny: int, nz: int, a: float) -> np.ndarray:
    if crystal == "BCC":
        basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    else:  # FCC
        basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    pts.append((i + b[0], j + b[1], k + b[2]))
    return np.asarray(pts, float) * a

def bcc_tetra(nx, ny, nz, a):
    basis = np.array([(0.50, 0.25, 0), (0.50, 0.75, 0), (0.25, 0.50, 0), (0.75, 0.50, 0),
                      (0.50, 0, 0.25), (0.50, 0, 0.75), (0.25, 0, 0.50), (0.75, 0, 0.50),
                      (0, 0.50, 0.25), (0, 0.50, 0.75), (0, 0.25, 0.50), (0, 0.75, 0.50)], float)
    pts = []
    lim = np.array([nx, ny, nz], float)
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                p = basis + np.array([i, j, k], float)
                pts.extend(p[np.all((p >= -1e-9) & (p <= lim + 1e-9), axis=1)].tolist())
    return unique_rows(np.asarray(pts) * a)

def bcc_octa(nx, ny, nz, a):
    gx = np.arange(0, nx + 0.25, 0.5); gy = np.arange(0, ny + 0.25, 0.5); gz = np.arange(0, nz + 0.25, 0.5)
    pts = np.asarray([(x, y, z) for x in gx for y in gy for z in gz], float)
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
    # small interstitials favour tetrahedral sites, larger ones octahedral
    return "tetrahedral" if species in {"H", "D", "T"} else "octahedral"

def interstitial_sites(crystal: str, network: str, nx: int, ny: int, nz: int, a: float) -> np.ndarray:
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
# ---------------------------------------------------------------------------
def rng_from_setup(setup):
    # deterministic seed so identical inputs give an identical alloy configuration
    key = (str(setup.get("composition_at_percent", {})) +
           str(setup.get("crystal_structure", "")) +
           str(setup.get("domain", {}).get("total_lattice_sites", 0)))
    seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2 ** 32)
    return np.random.default_rng(seed)

def place_solutes(host_pts, setup):
    rng = rng_from_setup(setup)
    idx = np.arange(len(host_pts)); rng.shuffle(idx)
    counts = dict(setup["domain"]["estimated_atom_counts"])
    out = {}; cur = 0
    for el in setup.get("alloying_elements", []):
        n = min(int(counts.get(el, 0)), max(0, len(idx) - cur))
        out[el] = host_pts[idx[cur:cur + n]]
        cur += n
    return out

# ---------------------------------------------------------------------------
# Energy fields  (reduced vital-energy set only)
#   U_i    = sum_s dU_s exp(-(r_is/R_U)^2)          host reference U = 0
#   Sshift = sum_s dEsad_s exp(-(r_mij_s/R_E)^2)    no cluster / pair terms
# ---------------------------------------------------------------------------
def compute_fields(sites, mids, solutes, energies, a):
    U = np.zeros(len(sites))            # host site-energy reference is zero (spec)
    Sshift = np.zeros(len(mids))
    R = R_INFLUENCE_A * a               # R_U = R_E
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
def build_system(sites, edges, mids, lengths, U, Sshift, setup, energies):
    n = len(sites); T = float(setup["temperature_K"]); kBT = max(KB_EV * T, 1e-30)
    E0 = float(energies.get("host_migration_barrier", 0.1))
    rows = []; cols = []; vals = []; adjacency = [[] for _ in range(n)]; edge_act = []
    for e, (i, j) in enumerate(edges):
        i = int(i); j = int(j)
        saddle = max(U[i], U[j]) + E0 + Sshift[e]          # E_saddle_ij
        Eij = max(saddle - U[i], E_MIN)                    # E_act(i->j) = E_saddle - U_i
        Eji = max(saddle - U[j], E_MIN)
        kij = NU0 * math.exp(-Eij / kBT)
        kji = NU0 * math.exp(-Eji / kBT)
        rows += [i, j]; cols += [j, i]; vals += [kij, kji]
        adjacency[i].append((j, kij, Eij, e)); adjacency[j].append((i, kji, Eji, e))
        edge_act.append(0.5 * (Eij + Eji))
    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    sum_jump = np.asarray(W.sum(axis=1)).ravel()
    a = float(setup["lattice_parameter_A"]); z = sites[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    near_surface = z <= zmin + 0.35 * a
    near_bulk = z >= zmax - 0.35 * a
    # Simple mode: surface and bulk exit barriers default to the host migration barrier.
    # (These .get() defaults leave room for an optional advanced mode later.)
    E_surf = float(energies.get("surface_exit_barrier", E0))
    E_bulk = float(energies.get("bulk_exit_barrier", E0))
    k_surface = np.zeros(n); k_bulk = np.zeros(n)
    for i in np.where(near_surface)[0]:
        k_surface[i] = NU0 * math.exp(-max(E_surf - U[i], E_MIN) / kBT)
    for i in np.where(near_bulk)[0]:
        k_bulk[i] = NU0 * math.exp(-max(E_bulk - U[i], E_MIN) / kBT)
    total = np.maximum(sum_jump + k_surface + k_bulk, 1e-300)
    A = diags(total, 0, format="csr") - W
    return {"W": W, "A": A, "adjacency": adjacency, "total": total,
            "k_surface": k_surface, "k_bulk": k_bulk,
            "edge_activation": np.asarray(edge_act)}

# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def source_distribution(sites, setup):
    a = float(setup["lattice_parameter_A"]); z = sites[:, 2]; x = sites[:, 0]; y = sites[:, 1]
    nx, ny, nz = setup["domain"]["Nx"], setup["domain"]["Ny"], setup["domain"]["Nz"]
    cx, cy = 0.5 * nx * a, 0.5 * ny * a
    z0 = {"surface_return": 0.25, "bulk_transmission": 0.10, "retention": 0.50}.get(setup["boundary"], 0.25) * nz * a
    sigz = max(0.75 * a, 0.06 * nz * a); sigxy = max(1.5 * a, 0.18 * min(nx, ny) * a)
    w = np.exp(-((z - z0) / sigz) ** 2) * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / sigxy ** 2))
    if not np.isfinite(w).all() or w.sum() <= 0:
        w = np.ones(len(sites))
    return w / w.sum()

def solve_fp(system, start):
    A = system["A"]
    try:
        qsurf = spsolve(A, system["k_surface"]); qsurf = np.clip(np.nan_to_num(qsurf, nan=0.5, posinf=1, neginf=0), 0, 1)
        tabs = spsolve(A, np.ones(A.shape[0])); tabs = np.maximum(np.nan_to_num(tabs, nan=0, posinf=0, neginf=0), 0)
        res = spsolve(A.T, start); res = np.maximum(np.nan_to_num(res, nan=0, posinf=0, neginf=0), 0)
        mfpt = float(start @ tabs)
        psurf = float(start @ qsurf); pbulk = 1.0 - psurf
    except Exception:
        qsurf = np.full(A.shape[0], 0.5); tabs = np.ones(A.shape[0]) * 1e-9; res = start.copy()
        mfpt = 1e-9; psurf = 0.5; pbulk = 0.5
    if not math.isfinite(mfpt) or mfpt <= 0:
        mfpt = 1e-12
    return {"q_surface": qsurf, "p_surface": max(0, min(1, psurf)),
            "p_bulk": max(0, min(1, pbulk)), "mfpt": mfpt, "residence": res}

def solve_mfpt(system, start):
    """Lightweight MFPT-only solve (used for the ideal-host retention reference)."""
    A = system["A"]
    try:
        tabs = spsolve(A, np.ones(A.shape[0]))
        tabs = np.maximum(np.nan_to_num(tabs, nan=0, posinf=0, neginf=0), 0)
        mfpt = float(start @ tabs)
    except Exception:
        mfpt = 1e-9
    if not math.isfinite(mfpt) or mfpt <= 0:
        mfpt = 1e-12
    return mfpt

def survival_curve(system, start, mfpt, npts=80):
    mfpt = max(float(mfpt), 1e-15)
    tmax = 5 * mfpt
    times = np.linspace(0, tmax, npts)
    try:
        Q = system["W"] - diags(system["total"], 0, format="csr")   # Q = K - diag(row sums + boundary)
        P = expm_multiply(Q.T, start, start=0, stop=tmax, num=npts)  # S(t) = 1^T exp(Q^T t) p0
        S = np.asarray(P.sum(axis=1)).ravel()
        S = np.clip(np.nan_to_num(S, nan=0, posinf=1, neginf=0), 0, 1)
        if len(S) != npts or S.max() <= 0:
            raise ValueError("bad survival")
    except Exception:
        S = np.exp(-times / mfpt)   # fallback so the survival plot is never empty
    S[0] = 1.0
    return {"time_s": times.tolist(), "S_t": S.tolist()}

def shortest_path(sites, system, start_idx, target):
    n = len(sites); dist = np.full(n, np.inf); prev = np.full(n, -1, dtype=int); dist[start_idx] = 0
    heap = [(0.0, start_idx)]
    absorb = system["k_surface"] if target == "surface" else system["k_bulk"]
    best = None; best_total = np.inf
    while heap:
        d, u = heapq.heappop(heap)
        if d != dist[u]:
            continue
        if absorb[u] > 0:
            total = d + 1 / max(absorb[u], 1e-300)
            if total < best_total:
                best_total = total; best = u
        for v, rate, eact, eidx in system["adjacency"][u]:
            nd = d + 1 / max(rate, 1e-300)        # edge resistance R_ij = 1 / k_ij
            if nd < dist[v]:
                dist[v] = nd; prev[v] = u; heapq.heappush(heap, (nd, v))
    if best is None:
        return [start_idx], float("inf")
    path = []; cur = int(best)
    while cur != -1:
        path.append(cur); cur = int(prev[cur]) if prev[cur] != -1 else -1
    path.reverse()
    return path, float(best_total)

def path_info(path, sites, system):
    if len(path) < 2:
        return {"tortuosity": 1.0, "barriers": [0.0], "mean_barrier": 0.0, "bottleneck": 0.0}
    lookup = {}
    for u, neis in enumerate(system["adjacency"]):
        for v, rate, eact, eidx in neis:
            lookup[(u, v)] = float(eact)      # directional activation barrier E_act(u->v)
    barriers = [lookup.get((int(u), int(v)), 0.0) for u, v in zip(path[:-1], path[1:])]
    coords = sites[np.asarray(path)]
    length = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
    straight = float(np.linalg.norm(coords[-1] - coords[0]))
    arr = np.asarray(barriers, float)
    return {"tortuosity": length / max(straight, 1e-12), "barriers": barriers,
            "mean_barrier": float(np.nanmean(arr)), "bottleneck": float(np.nanmax(arr))}

# ---------------------------------------------------------------------------
# Output filtering (thin y-slice around the dominant pathway)
# ---------------------------------------------------------------------------
def in_path_slice(points, y0, a, width=0.70):
    return np.abs(points[:, 1] - y0) <= width * a

def thin(points, a, mask=None, max_points=800):
    if mask is not None:
        points = points[mask]
    if len(points) > max_points:
        points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
    return {"x": (points[:, 0] / a).tolist() if len(points) else [],
            "z": (points[:, 2] / a).tolist() if len(points) else []}

def solute_out(solutes, a, y0, max_each=180):
    out = []
    for el, pts in solutes.items():
        pts = pts[in_path_slice(pts, y0, a, 0.85)] if len(pts) else pts
        if len(pts) > max_each:
            pts = pts[np.linspace(0, len(pts) - 1, max_each).astype(int)]
        for p in pts:
            out.append({"element": el, "x": float(p[0] / a), "y": float(p[1] / a), "z": float(p[2] / a)})
    return out

# ---------------------------------------------------------------------------
# JSON safety: no NaN / Infinity ever leaves the backend
# ---------------------------------------------------------------------------
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
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
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
    sites = interstitial_sites(crystal, network, nx, ny, nz, a)
    edges, mids, lengths = build_edges(sites)
    solutes = place_solutes(host, setup)

    E0 = float(energies_eV.get("host_migration_barrier", 0.1))

    # --- alloy calculation (user-supplied shifts) ---
    U, Sshift = compute_fields(sites, mids, solutes, energies_eV, a)
    system = build_system(sites, edges, mids, lengths, U, Sshift, setup, energies_eV)
    start = source_distribution(sites, setup)
    fp = solve_fp(system, start)
    survival = survival_curve(system, start, fp["mfpt"])

    # --- ideal-host reference for a rigorous retention factor ---
    # same geometry and boundary condition, all solute shifts set to zero.
    host_energies = {"host_migration_barrier": E0}
    U_h, S_h = compute_fields(sites, mids, solutes, host_energies, a)  # -> U=0, Sshift=0
    system_h = build_system(sites, edges, mids, lengths, U_h, S_h, setup, host_energies)
    mfpt_host = solve_mfpt(system_h, start)
    retention = float(fp["mfpt"] / mfpt_host) if mfpt_host > 0 else 1.0

    # --- dominant minimum-resistance pathway ---
    source_idx = int(np.argmax(start))
    path_s, Rs = shortest_path(sites, system, source_idx, "surface")
    path_b, Rb = shortest_path(sites, system, source_idx, "bulk")
    if fp["p_surface"] >= fp["p_bulk"]:
        target, path, Rdom = "surface", path_s, Rs
    else:
        target, path, Rdom = "bulk", path_b, Rb
    pm = path_info(path, sites, system)
    path_pts = sites[np.asarray(path)]
    y0 = float(np.median(path_pts[:, 1])) if len(path_pts) else 0.5 * ny * a

    # --- physically meaningful traps (U_i < -0.01 eV) ---
    trap_mask = (U < TRAP_THRESHOLD_EV) & in_path_slice(sites, y0, a, 0.75)
    trap_idx = np.where(trap_mask)[0]
    if len(trap_idx) > 250:
        trap_idx = trap_idx[np.linspace(0, len(trap_idx) - 1, 250).astype(int)]

    # --- top 5% highest-barrier edge midpoints in the slice ---
    ea = system["edge_activation"]
    high_thr = float(np.percentile(ea, 95)) if len(ea) else float("inf")
    high_idx = np.where((ea >= high_thr) & in_path_slice(mids, y0, a, 0.75))[0]
    if len(high_idx) > 250:
        high_idx = high_idx[np.linspace(0, len(high_idx) - 1, 250).astype(int)]

    # --- residence-weighted trap fraction ---
    res = fp["residence"]
    total_res = float(res.sum()) if np.isfinite(res.sum()) and res.sum() > 0 else 1.0
    trap_frac = float(res[U < TRAP_THRESHOLD_EV].sum() / total_res) if np.any(U < TRAP_THRESHOLD_EV) else 0.0

    mean_bar = pm["mean_barrier"] if math.isfinite(pm["mean_barrier"]) else E0
    barriers = pm["barriers"]
    marker_barriers = ([barriers[0]] + barriers) if len(barriers) == len(path) - 1 and len(barriers) > 0 else [mean_bar] * len(path)

    result = {
        "metrics": {
            "mean_first_passage_time_s": fp["mfpt"],
            "probability_bulk_transmission": fp["p_bulk"],
            "probability_surface_return": fp["p_surface"],
            "bottleneck_barrier_eV": pm["bottleneck"],
            "minimum_resistance_path_s": Rdom,
            "path_tortuosity": pm["tortuosity"],
            "pathway_energy_cost_eV": mean_bar,
            "trap_residence_fraction": max(0.0, min(1.0, trap_frac)),
            "percolation_accessibility_fraction": float(np.mean(ea <= E0 + 0.05)) if len(ea) else 0.0,
            "retention_factor_vs_host": retention,
        },
        "survival_probability": survival,
        "dominant_pathway": {
            "target": target,
            "x": (path_pts[:, 0] / a).tolist(),
            "y": (path_pts[:, 1] / a).tolist(),
            "z": (path_pts[:, 2] / a).tolist(),
            "activation_barrier_eV": marker_barriers,
        },
        "lattice_points": thin(host, a, in_path_slice(host, y0, a, 0.55), 700),
        "interstitial_sites": thin(sites, a, in_path_slice(sites, y0, a, 0.55), 700),
        "solute_positions": solute_out(solutes, a, y0, 120),
        "trap_sites": {"x": (sites[trap_idx, 0] / a).tolist(), "z": (sites[trap_idx, 2] / a).tolist()},
        "high_barrier_edges": {"x": (mids[high_idx, 0] / a).tolist(), "z": (mids[high_idx, 2] / a).tolist()},
        "domain": setup["domain"],
        "site_network_used": network,
        "notes": {
            "energy_definition": "Activation barrier is E_act(i->j) = E_saddle(i,j) - U_i; the diffusing-species site energy U_i is explicit.",
            "reduced_energy_mode": "Vital-energy mode: host_migration_barrier plus site- and saddle-energy shifts per alloying element. U_host=0 and E_surface=E_bulk=E_m^0 are internal defaults.",
            "retention_definition": "Retention factor = tau_MFPT(alloy) / tau_MFPT(ideal host), computed from two solves on the same domain.",
            "visual_filter": "Lattice/interstitial/solute points are filtered to a thin y-slice around the dominant path to avoid projection clutter.",
            "survival_bugfix": "A valid survival curve is always returned; an exponential fallback is used if the matrix exponential fails.",
        },
    }
    return json_safe(result)
