/*
IDOMS trial frontend.
No database. No energy estimation. User supplies all energies in eV.
API_BASE is empty by default, so this GitHub Pages trial runs fully in-browser.
Later, set API_BASE to your Render/FastAPI URL.
*/

const API_BASE = "https://idoms-backend.onrender.com"; // Example later: "https://idoms-backend.onrender.com"
const MAX_ALLOY_ELEMENTS = 5;
const MIN_ATOMS_PER_ALLOY = 3;
const MIN_NX_NY = 6;
const MIN_NZ = 8;

let latestSetup = null;
let latestEnergyTemplate = [];
let latestResult = null;

document.addEventListener("DOMContentLoaded", () => {
  addCompositionRow("C", 1.0);
  addCompositionRow("Cr", 9.0);
  updateCompositionSummary();
});

function addCompositionRow(element = "", atpct = "") {
  const tbody = document.getElementById("compositionRows");
  if (tbody.children.length >= MAX_ALLOY_ELEMENTS) {
    alert(`Trial version allows up to ${MAX_ALLOY_ELEMENTS} alloying elements.`);
    return;
  }
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="el-symbol" value="${element}" placeholder="e.g. Cr" oninput="updateCompositionSummary()" /></td>
    <td><input class="el-atpct" type="number" min="0" max="100" step="0.001" value="${atpct}" oninput="updateCompositionSummary()" /></td>
    <td><button class="remove-btn" onclick="removeRow(this)">Remove</button></td>
  `;
  tbody.appendChild(tr);
  updateCompositionSummary();
}

function removeRow(btn) {
  btn.closest("tr").remove();
  updateCompositionSummary();
}

function getComposition() {
  const rows = [...document.querySelectorAll("#compositionRows tr")];
  const comp = {};
  for (const row of rows) {
    const el = row.querySelector(".el-symbol").value.trim();
    const val = Number(row.querySelector(".el-atpct").value);
    if (el && val > 0) comp[el] = val;
  }
  return comp;
}

function updateCompositionSummary() {
  const comp = getComposition();
  const totalAlloy = Object.values(comp).reduce((a, b) => a + b, 0);
  const host = document.getElementById("hostElement")?.value?.trim() || "Host";
  const hostPct = 100 - totalAlloy;
  const box = document.getElementById("compositionSummary");
  if (!box) return;
  if (hostPct < 0) {
    box.textContent = "Error: total alloying content exceeds 100 at.%.";
    box.classList.add("error");
    return;
  }
  box.classList.remove("error");
  box.textContent = [
    `Alloying elements: ${Object.keys(comp).length}`,
    `Total alloying content: ${totalAlloy.toFixed(3)} at.%`,
    `${host} content: ${hostPct.toFixed(3)} at.%`
  ].join("\n");
}

function collectSetup() {
  const host = document.getElementById("hostElement").value.trim();
  const composition = getComposition();
  const alloyingElements = Object.keys(composition);
  const totalAlloy = Object.values(composition).reduce((a, b) => a + b, 0);
  if (!host) throw new Error("Please enter a host element.");
  if (alloyingElements.length > MAX_ALLOY_ELEMENTS) throw new Error("Maximum 5 alloying elements allowed.");
  if (totalAlloy >= 100) throw new Error("Total alloying content must be less than 100 at.%.");

  const aAng = Number(document.getElementById("latticeA").value);
  if (!(aAng > 0)) throw new Error("Please enter a valid lattice parameter in Å.");
  const temperatureK = Number(document.getElementById("temperatureK").value);
  if (!(temperatureK > 0)) throw new Error("Please enter a valid temperature in K.");
  const slabThicknessNm = Number(document.getElementById("slabThicknessNm").value);
  if (!(slabThicknessNm > 0)) throw new Error("Please enter a valid slab thickness in nm.");

  const crystal = document.getElementById("crystalStructure").value;
  const sitePerCell = crystal === "BCC" ? 2 : 4;
  const domain = calculateEfficientDomain(composition, crystal, aAng, slabThicknessNm);

  return {
    host,
    alloying_elements: alloyingElements,
    composition_at_percent: composition,
    host_at_percent: 100 - totalAlloy,
    crystal_structure: crystal,
    lattice_parameter_A: aAng,
    interstitial_species: document.getElementById("interstitialSpecies").value,
    temperature_K: temperatureK,
    requested_slab_thickness_nm: slabThicknessNm,
    boundary: document.getElementById("boundaryType").value,
    site_per_unit_cell: sitePerCell,
    domain
  };
}

function calculateEfficientDomain(composition, crystal, aAng, slabThicknessNm) {
  const sitePerCell = crystal === "BCC" ? 2 : 4;
  const fractions = Object.values(composition).filter(v => v > 0).map(v => v / 100);
  let requiredLatticeSites = 120;
  for (const f of fractions) {
    requiredLatticeSites = Math.max(requiredLatticeSites, Math.ceil(MIN_ATOMS_PER_ALLOY / f));
  }
  const aNm = aAng * 0.1;
  let Nz = Math.max(MIN_NZ, Math.ceil(slabThicknessNm / aNm));
  const requiredCells = Math.ceil(requiredLatticeSites / sitePerCell);
  let Nx = Math.max(MIN_NX_NY, Math.ceil(Math.sqrt(requiredCells / Nz)));
  let Ny = Nx;
  while (sitePerCell * Nx * Ny * Nz < requiredLatticeSites) {
    if (Nx <= Ny) Nx++;
    else Ny++;
  }
  const totalSites = sitePerCell * Nx * Ny * Nz;
  const atomCounts = {};
  for (const [el, atpct] of Object.entries(composition)) {
    atomCounts[el] = Math.max(1, Math.round(totalSites * atpct / 100));
  }
  const totalAlloyAtoms = Object.values(atomCounts).reduce((a, b) => a + b, 0);
  atomCounts["host"] = Math.max(0, totalSites - totalAlloyAtoms);
  return {
    Nx, Ny, Nz,
    unit: "unit cells",
    total_lattice_sites: totalSites,
    required_lattice_sites_minimum: requiredLatticeSites,
    physical_size_nm: { Lx: Nx * aNm, Ly: Ny * aNm, Lz: Nz * aNm },
    estimated_atom_counts: atomCounts
  };
}

function buildEnergyTemplate(setup) {
  const I = setup.interstitial_species;
  const host = setup.host;
  const template = [
    { key: "host_migration_barrier", label: `Baseline ${I} migration barrier in ${host}`, unit: "eV", default: 0.088 },
    { key: "surface_exit_barrier", label: "Surface exit / return barrier", unit: "eV", default: 0.088 },
    { key: "bulk_exit_barrier", label: "Bulk transmission barrier", unit: "eV", default: 0.088 }
  ];
  for (const el of setup.alloying_elements) {
    template.push(
      { key: `${el}_interstitial_binding`, label: `${I} binding energy near ${el}`, unit: "eV", default: 0.000 },
      { key: `${el}_modified_jump_barrier`, label: `${el}-modified local jump barrier for ${I}`, unit: "eV", default: 0.088 },
      { key: `${el}_trap_depth`, label: `Trap depth near ${el}-rich local region`, unit: "eV", default: 0.000 }
    );
  }
  return template;
}

async function prepareEnergyInputs() {
  try {
    latestSetup = collectSetup();
    latestEnergyTemplate = buildEnergyTemplate(latestSetup);
    const d = latestSetup.domain;
    const L = d.physical_size_nm;
    const counts = Object.entries(d.estimated_atom_counts).map(([k, v]) => `${k}: ${v}`).join(", ");
    document.getElementById("domainSummary").textContent =
      `Calculated domain: ${d.Nx} × ${d.Ny} × ${d.Nz} ${d.unit}\n` +
      `Physical size: ${L.Lx.toFixed(3)} × ${L.Ly.toFixed(3)} × ${L.Lz.toFixed(3)} nm\n` +
      `Total lattice sites: ${d.total_lattice_sites}\n` +
      `Estimated lattice-site counts: ${counts}\n\n` +
      `Note: Nx, Ny and Nz are numbers of unit cells and have no physical unit. Physical length is N × a.`;
    renderEnergyFields(latestEnergyTemplate);
  } catch (err) {
    alert(err.message);
  }
}

function renderEnergyFields(template) {
  const container = document.getElementById("energyFields");
  container.innerHTML = "";
  template.forEach(item => {
    const row = document.createElement("div");
    row.className = "energy-row";
    row.innerHTML = `
      <div><label>${item.label}</label><div class="energy-key">key: ${item.key}; unit: ${item.unit}</div></div>
      <input class="energy-input" id="energy_${item.key}" type="number" step="0.001" value="${item.default}" />`;
    container.appendChild(row);
  });
}

function collectEnergies() {
  const energies = {};
  for (const item of latestEnergyTemplate) {
    const el = document.getElementById(`energy_${item.key}`);
    if (!el) throw new Error(`Missing energy input: ${item.key}`);
    const val = Number(el.value);
    if (!Number.isFinite(val)) throw new Error(`Invalid energy value for ${item.key}`);
    energies[item.key] = val;
  }
  return energies;
}

async function runSimulation() {
  try {
    if (!latestSetup) await prepareEnergyInputs();
    const energies = collectEnergies();
    const payload = { setup: latestSetup, energies_eV: energies };
    document.getElementById("runStatus").textContent = "Running trial simulation...";
    if (API_BASE) {
      const response = await fetch(`${API_BASE}/simulate`, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
      latestResult = await response.json();
    } else {
      latestResult = runBrowserTrialSimulation(payload);
    }
    renderResults(latestResult);
    document.getElementById("runStatus").textContent = "Simulation complete. Results generated using the IDOMS backend.";
  } catch (err) {
    alert(err.message);
    document.getElementById("runStatus").textContent = "Simulation failed.";
  }
}

function seededRandom(seed) {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return function() { s = s * 16807 % 2147483647; return (s - 1) / 2147483646; };
}
function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h += (h << 1) + (h << 4) + (h << 7) + (h << 8) + (h << 24); }
  return Math.abs(h >>> 0);
}

function runBrowserTrialSimulation(payload) {
  const setup = payload.setup;
  const energies = payload.energies_eV;
  const d = setup.domain;
  const rng = seededRandom(hashString(JSON.stringify(setup.composition_at_percent)));
  const hostBarrier = energies.host_migration_barrier;
  const surfaceBarrier = energies.surface_exit_barrier;
  const bulkBarrier = energies.bulk_exit_barrier;
  const kBT = 8.617333262e-5 * setup.temperature_K;

  const solutes = [];
  for (const [el, count] of Object.entries(d.estimated_atom_counts)) {
    if (el === "host") continue;
    const nPlot = Math.min(count, 250);
    for (let i = 0; i < nPlot; i++) solutes.push({ element: el, x: rng() * d.Nx, z: rng() * d.Nz });
  }

  const points = 90, x = [], z = [], barrierAlongPath = [];
  let xcur = d.Nx * 0.5;
  for (let i = 0; i < points; i++) {
    const t = i / (points - 1);
    const zcur = t * d.Nz;
    let push = 0, localBarrier = hostBarrier;
    for (const s of solutes) {
      const dx = xcur - s.x, dz = zcur - s.z;
      const r2 = dx * dx + dz * dz + 0.25;
      const jumpKey = `${s.element}_modified_jump_barrier`;
      const trapKey = `${s.element}_trap_depth`;
      const bindKey = `${s.element}_interstitial_binding`;
      const penalty = (energies[jumpKey] || hostBarrier) - hostBarrier;
      const trap = energies[trapKey] || 0;
      const bind = energies[bindKey] || 0;
      localBarrier += Math.max(0, penalty) * Math.exp(-r2 / 3.0);
      localBarrier += Math.max(0, trap + bind) * 0.35 * Math.exp(-r2 / 2.0);
      push += (dx / r2) * Math.max(0, penalty) * 0.55;
      push -= (dx / r2) * Math.max(0, trap + bind) * 0.10;
    }
    xcur += push + (rng() - 0.5) * 0.08;
    xcur = Math.max(0.2, Math.min(d.Nx - 0.2, xcur));
    x.push(xcur); z.push(zcur); barrierAlongPath.push(localBarrier);
  }

  const bottleneck = Math.max(...barrierAlongPath);
  const meanBarrier = barrierAlongPath.reduce((a, b) => a + b, 0) / barrierAlongPath.length;
  const rateScale = 1e13 * Math.exp(-meanBarrier / kBT);
  const tortuosity = computePathTortuosity(x, z);
  const mfpt = tortuosity * points / Math.max(rateScale, 1e-30);
  const kSurface = Math.exp(-surfaceBarrier / kBT);
  const kBulk = Math.exp(-bulkBarrier / kBT);
  const alloyResistance = Math.exp((meanBarrier - hostBarrier) / Math.max(kBT, 1e-30)) * tortuosity;
  let pSurface = kSurface / (kSurface + kBulk * alloyResistance);
  pSurface = Math.max(0, Math.min(1, pSurface));
  const pBulk = 1 - pSurface;
  const trapResidence = estimateTrapResidenceFraction(energies, solutes, x, z);
  const times = [], S = [];
  for (let i = 0; i < 80; i++) { const tt = (i / 79) * mfpt * 5; times.push(tt); S.push(Math.exp(-tt / Math.max(mfpt, 1e-30))); }

  return {
    metrics: { mean_first_passage_time_s: mfpt, probability_bulk_transmission: pBulk, probability_surface_return: pSurface, bottleneck_barrier_eV: bottleneck, mean_path_barrier_eV: meanBarrier, trap_residence_fraction: trapResidence, path_tortuosity: tortuosity },
    survival_probability: { time_s: times, S_t: S },
    dominant_pathway: { x, z, barrier_eV: barrierAlongPath },
    solute_positions: solutes,
    domain: d,
    note: "Browser-only trial calculation using user-supplied energies. Replace with FastAPI backend for the full graph/KMC engine."
  };
}

function computePathTortuosity(x, z) {
  let length = 0;
  for (let i = 1; i < x.length; i++) length += Math.sqrt((x[i] - x[i-1]) ** 2 + (z[i] - z[i-1]) ** 2);
  const straight = Math.sqrt((x[x.length-1] - x[0]) ** 2 + (z[z.length-1] - z[0]) ** 2);
  return length / Math.max(straight, 1e-12);
}
function estimateTrapResidenceFraction(energies, solutes, x, z) {
  if (solutes.length === 0) return 0;
  let trapScore = 0;
  for (let i = 0; i < x.length; i++) {
    let local = 0;
    for (const s of solutes) {
      const trap = energies[`${s.element}_trap_depth`] || 0;
      const bind = energies[`${s.element}_interstitial_binding`] || 0;
      const r2 = (x[i] - s.x) ** 2 + (z[i] - s.z) ** 2;
      local += Math.max(0, trap + bind) * Math.exp(-r2 / 1.5);
    }
    trapScore += local;
  }
  trapScore /= x.length;
  return Math.max(0, Math.min(0.95, trapScore / (trapScore + 0.25)));
}

function renderResults(result) { renderMetricCards(result.metrics); renderPathPlot(result); renderSurvivalPlot(result); }
function formatValue(v) { return Math.abs(v) > 0 && (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) ? v.toExponential(3) : Number(v).toFixed(4); }
function renderMetricCards(metrics) {
  const container = document.getElementById("metricsCards");
  container.innerHTML = "";
  const labels = { mean_first_passage_time_s: "Mean first-passage time (s)", probability_bulk_transmission: "Probability of bulk transmission", probability_surface_return: "Probability of surface return", bottleneck_barrier_eV: "Bottleneck barrier (eV)", mean_path_barrier_eV: "Mean path barrier (eV)", trap_residence_fraction: "Trap residence fraction", path_tortuosity: "Path tortuosity" };
  for (const [key, label] of Object.entries(labels)) {
    if (!(key in metrics)) continue;
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<div class="value">${formatValue(metrics[key])}</div><div class="label">${label}</div>`;
    container.appendChild(div);
  }
}
function renderPathPlot(result) {
  const path = result.dominant_pathway, d = result.domain;
  const traces = [];
  const byEl = {};
  for (const s of result.solute_positions || []) {
    if (!byEl[s.element]) byEl[s.element] = {x: [], z: []};
    byEl[s.element].x.push(s.x); byEl[s.element].z.push(s.z);
  }
  for (const [el, pts] of Object.entries(byEl)) traces.push({ x: pts.x, y: pts.z, mode: "markers", type: "scatter", name: `${el} positions`, marker: {size: 7, opacity: 0.55} });
  traces.push({ x: path.x, y: path.z, mode: "lines+markers", type: "scatter", name: "dominant pathway", line: {width: 4}, marker: { size: 6, color: path.barrier_eV, colorscale: "Inferno", colorbar: {title: "local barrier<br>(eV)"} } });
  Plotly.newPlot("pathPlot", traces, { title: "Dominant interstitial pathway in calculated domain", xaxis: {title: "x / a", range: [0, d.Nx]}, yaxis: {title: "depth z / a", range: [d.Nz, 0]}, legend: {orientation: "h"}, margin: {l: 60, r: 30, t: 55, b: 70} }, {responsive: true});
}
function renderSurvivalPlot(result) {
  Plotly.newPlot("survivalPlot", [{ x: result.survival_probability.time_s, y: result.survival_probability.S_t, mode: "lines", type: "scatter", name: "S(t)", line: {width: 4} }], { title: "Survival probability", xaxis: {title: "time (s)"}, yaxis: {title: "S(t)", range: [0, 1.05]}, margin: {l: 60, r: 30, t: 55, b: 70} }, {responsive: true});
}
function downloadInputJson() {
  if (!latestSetup) { alert("Generate energy inputs first."); return; }
  const energies = latestEnergyTemplate.length ? collectEnergies() : {};
  downloadJson("idoms_input.json", {setup: latestSetup, energies_eV: energies});
}
function downloadResultJson() {
  if (!latestResult) { alert("Run a simulation first."); return; }
  downloadJson("idoms_result.json", latestResult);
}
function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url);
}
