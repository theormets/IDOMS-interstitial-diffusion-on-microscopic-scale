import math
import random
from typing import Dict, Any

KB_EV = 8.617333262e-5

def run_simulation(setup: Dict[str, Any], energies_eV: Dict[str, float]) -> Dict[str, Any]:
    """
    Trial backend engine.
    This uses user-supplied energies only. Replace with the full graph/KMC engine when ready.
    """
    domain = setup["domain"]
    Nx, Nz = domain["Nx"], domain["Nz"]
    temperature = setup["temperature_K"]
    kBT = KB_EV * temperature

    host_barrier = float(energies_eV.get("host_migration_barrier", 0.1))
    surface_barrier = float(energies_eV.get("surface_exit_barrier", host_barrier))
    bulk_barrier = float(energies_eV.get("bulk_exit_barrier", host_barrier))

    seed = abs(hash(str(setup["composition_at_percent"]))) % (2**32)
    rng = random.Random(seed)

    solutes = []
    for el, count in domain["estimated_atom_counts"].items():
        if el == "host":
            continue
        for _ in range(min(int(count), 250)):
            solutes.append({"element": el, "x": rng.random() * Nx, "z": rng.random() * Nz})

    x, z, barriers = [], [], []
    xcur = 0.5 * Nx
    npts = 90
    for i in range(npts):
        t = i / (npts - 1)
        zcur = t * Nz
        push = 0.0
        local_barrier = host_barrier
        for s in solutes:
            dx = xcur - s["x"]
            dz = zcur - s["z"]
            r2 = dx * dx + dz * dz + 0.25
            jump = float(energies_eV.get(f"{s['element']}_modified_jump_barrier", host_barrier))
            trap = float(energies_eV.get(f"{s['element']}_trap_depth", 0.0))
            bind = float(energies_eV.get(f"{s['element']}_interstitial_binding", 0.0))
            penalty = jump - host_barrier
            local_barrier += max(0.0, penalty) * math.exp(-r2 / 3.0)
            local_barrier += max(0.0, trap + bind) * 0.35 * math.exp(-r2 / 2.0)
            push += (dx / r2) * max(0.0, penalty) * 0.55
            push -= (dx / r2) * max(0.0, trap + bind) * 0.10
        xcur += push + (rng.random() - 0.5) * 0.08
        xcur = max(0.2, min(Nx - 0.2, xcur))
        x.append(xcur)
        z.append(zcur)
        barriers.append(local_barrier)

    length = sum(math.sqrt((x[i] - x[i-1])**2 + (z[i] - z[i-1])**2) for i in range(1, npts))
    straight = math.sqrt((x[-1] - x[0])**2 + (z[-1] - z[0])**2)
    tortuosity = length / max(straight, 1e-12)

    mean_barrier = sum(barriers) / len(barriers)
    bottleneck = max(barriers)
    rate_scale = 1e13 * math.exp(-mean_barrier / max(kBT, 1e-30))
    mfpt = tortuosity * npts / max(rate_scale, 1e-30)

    k_surface = math.exp(-surface_barrier / max(kBT, 1e-30))
    k_bulk = math.exp(-bulk_barrier / max(kBT, 1e-30))
    alloy_resistance = math.exp((mean_barrier - host_barrier) / max(kBT, 1e-30)) * tortuosity
    p_surface = k_surface / max(k_surface + k_bulk * alloy_resistance, 1e-30)
    p_surface = max(0.0, min(1.0, p_surface))
    p_bulk = 1.0 - p_surface

    trap_score = 0.0
    for i in range(npts):
        local = 0.0
        for s in solutes:
            trap = float(energies_eV.get(f"{s['element']}_trap_depth", 0.0))
            bind = float(energies_eV.get(f"{s['element']}_interstitial_binding", 0.0))
            r2 = (x[i] - s["x"])**2 + (z[i] - s["z"])**2
            local += max(0.0, trap + bind) * math.exp(-r2 / 1.5)
        trap_score += local
    trap_score /= npts
    trap_fraction = max(0.0, min(0.95, trap_score / (trap_score + 0.25)))

    times, S_t = [], []
    for i in range(80):
        tt = (i / 79) * mfpt * 5
        times.append(tt)
        S_t.append(math.exp(-tt / max(mfpt, 1e-30)))

    return {
        "metrics": {
            "mean_first_passage_time_s": mfpt,
            "probability_bulk_transmission": p_bulk,
            "probability_surface_return": p_surface,
            "bottleneck_barrier_eV": bottleneck,
            "mean_path_barrier_eV": mean_barrier,
            "trap_residence_fraction": trap_fraction,
            "path_tortuosity": tortuosity,
        },
        "survival_probability": {"time_s": times, "S_t": S_t},
        "dominant_pathway": {"x": x, "z": z, "barrier_eV": barriers},
        "solute_positions": solutes,
        "domain": domain,
        "note": "Trial backend engine using user-supplied energies only. Replace with the full graph/KMC engine for publication-quality runs.",
    }
