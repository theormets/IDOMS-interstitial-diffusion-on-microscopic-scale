const DEFAULT_BACKEND_URL = "https://idoms-backend.onrender.com";
const MAX_ALLOY_ELEMENTS = 2;
const MIN_ATOMS_PER_ALLOY = 4;
const MIN_NX_NY = 5;
const MIN_NZ = 8;

// Literature-anchored default energy shifts (eV) used by the validation case,
// so the tool's defaults reproduce the worked example exactly.
const DEFAULT_SHIFTS = {
  C:  {site: -0.090, saddle: 0.070},
  Cr: {site:  0.025, saddle: 0.007}
};

const INFERNO = "Inferno";
const TEAL = "#0f766e";

let latestSetup = null;
let latestEnergyTemplate = [];
let latestResult = null;

document.addEventListener("DOMContentLoaded", () => {
  const urlInput = document.getElementById("backendUrl");
  if (urlInput && !urlInput.value.trim()) urlInput.value = DEFAULT_BACKEND_URL;
  addCompositionRow("C", 1.0);
  addCompositionRow("Cr", 9.0);
  updateCompositionSummary();
  updateAddButtonState();
});

function apiBase() {
  const v = document.getElementById("backendUrl").value.trim().replace(/\/$/, "");
  return v || DEFAULT_BACKEND_URL;
}

async function checkBackend() {
  const status = document.getElementById("backendStatus");
  try {
    const r = await fetch(`${apiBase()}/health`);
    const j = await r.json();
    status.textContent = `Backend status: ${j.status}. Version: ${j.version || "unknown"}`;
  } catch (err) {
    status.textContent = `Backend check failed: ${err.message}`;
  }
}

function updateAddButtonState() {
  const btn = document.getElementById("addAlloyBtn");
  if (!btn) return;
  const rows = document.querySelectorAll("#compositionRows tr").length;
  btn.style.display = rows >= MAX_ALLOY_ELEMENTS ? "none" : "";
}

function addCompositionRow(element = "", atpct = "") {
  const tbody = document.getElementById("compositionRows");
  if (tbody.children.length >= MAX_ALLOY_ELEMENTS) {
    updateAddButtonState();
    return;
  }
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="el-symbol" value="${element}" placeholder="e.g. Cr" oninput="updateCompositionSummary()" /></td>
    <td><input class="el-atpct" type="number" min="0" max="100" step="0.001" value="${atpct}" oninput="updateCompositionSummary()" /></td>
    <td><button class="remove-btn" onclick="removeRow(this)">Remove</button></td>`;
  tbody.appendChild(tr);
  updateCompositionSummary();
  updateAddButtonState();
}

function removeRow(btn) {
  btn.closest("tr").remove();
  updateCompositionSummary();
  updateAddButtonState();
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
    return;
  }
  box.textContent =
    `Alloying elements: ${Object.keys(comp).length} / ${MAX_ALLOY_ELEMENTS}\n` +
    `Total alloying content: ${totalAlloy.toFixed(3)} at.%\n` +
    `${host} content: ${hostPct.toFixed(3)} at.%`;
}

function collectSetup() {
  const host = document.getElementById("hostElement").value.trim();
  const composition = getComposition();
  const alloyingElements = Object.keys(composition);
  const totalAlloy = Object.values(composition).reduce((a, b) => a + b, 0);
  if (!host) throw new Error("Please enter a host element.");
  if (alloyingElements.length > MAX_ALLOY_ELEMENTS) throw new Error("Maximum 2 alloying elements allowed.");
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
    site_network: document.getElementById("siteNetwork").value,
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
  let requiredLatticeSites = 160;
  for (const f of fractions) requiredLatticeSites = Math.max(requiredLatticeSites, Math.ceil(MIN_ATOMS_PER_ALLOY / f));
  const aNm = aAng * 0.1;
  let Nz = Math.max(MIN_NZ, Math.ceil(slabThicknessNm / aNm));
  const requiredCells = Math.ceil(requiredLatticeSites / sitePerCell);
  let Nx = Math.max(MIN_NX_NY, Math.ceil(Math.sqrt(requiredCells / Nz)));
  let Ny = Nx;
  while (sitePerCell * Nx * Ny * Nz < requiredLatticeSites) {
    if (Nx <= Ny) Nx++; else Ny++;
  }
  const totalSites = sitePerCell * Nx * Ny * Nz;
  const atomCounts = {};
  for (const [el, atpct] of Object.entries(composition)) atomCounts[el] = Math.max(1, Math.round(totalSites * atpct / 100));
  const totalAlloyAtoms = Object.values(atomCounts).reduce((a, b) => a + b, 0);
  atomCounts["host"] = Math.max(0, totalSites - totalAlloyAtoms);
  return {
    Nx, Ny, Nz,
    unit: "unit cells",
    total_lattice_sites: totalSites,
    required_lattice_sites_minimum: requiredLatticeSites,
    physical_size_nm: {Lx: Nx * aNm, Ly: Ny * aNm, Lz: Nz * aNm},
    estimated_atom_counts: atomCounts
  };
}

// Reduced vital-energy set: host_migration_barrier + per-alloy {site,saddle} shifts.
// Per-element defaults reproduce the worked validation example.
function buildEnergyTemplate(setup) {
  const I = setup.interstitial_species;
  const host = setup.host;
  const defaultBarrier = setup.crystal_structure === "BCC" && host === "Fe" && I === "H" ? 0.088 : 0.100;
  const template = [
    {key: "host_migration_barrier", label: `Baseline ${I} migration barrier in pure ${host}`, kind: "activation barrier", unit: "eV", default: defaultBarrier}
  ];
  for (const el of setup.alloying_elements) {
    const d = DEFAULT_SHIFTS[el] || {site: 0.000, saddle: 0.000};
    template.push(
      {key: `${el}_site_energy_shift`, label: `Site-energy shift ΔU for ${I} near ${el} (negative = attractive/trapping)`, kind: "site energy", unit: "eV", default: d.site},
      {key: `${el}_saddle_energy_shift`, label: `Saddle-energy shift ΔE‡ for ${I} jumps near ${el} (positive = local barrier)`, kind: "saddle energy", unit: "eV", default: d.saddle}
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
      `Nx, Ny and Nz are numbers of unit cells. Physical length is N × a.`;
    renderEnergyFields(latestEnergyTemplate);
  } catch (err) { alert(err.message); }
}

function renderEnergyFields(template) {
  const container = document.getElementById("energyFields");
  container.innerHTML = "";
  template.forEach(item => {
    const row = document.createElement("div");
    row.className = "energy-row";
    row.innerHTML = `
      <div><label>${item.label}</label><span class="energy-type">${item.kind}</span><div class="energy-key">key: ${item.key}; unit: ${item.unit}</div></div>
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
    const payload = {setup: latestSetup, energies_eV: energies};
    document.getElementById("runStatus").textContent = "Running backend graph/KMC calculation...";
    const response = await fetch(`${apiBase()}/simulate`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
    if (!response.ok) throw new Error(`Backend error: ${await response.text()}`);
    latestResult = await response.json();
    if (latestResult && latestResult.error) throw new Error(latestResult.error);
    renderResults(latestResult);
    document.getElementById("runStatus").textContent = "Simulation complete. Results are from backend graph/master-equation calculation using user-supplied energies.";
  } catch (err) {
    alert(err.message);
    document.getElementById("runStatus").textContent = "Simulation failed.";
  }
}

function renderResults(result) {
  renderMetricCards(result.metrics || {});
  renderPathPlot(result);
  renderBarrierPlot(result);
  renderSurvivalPlot(result);
  renderSurvivalMapPlot(result);
}

function isFiniteArray(arr) { return Array.isArray(arr) && arr.length > 0 && arr.every(v => Number.isFinite(v)); }

function formatValue(v) { if (v === null || v === undefined || Number.isNaN(v)) return "—"; if (Math.abs(v) > 0 && (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4)) return Number(v).toExponential(3); return Number(v).toFixed(4); }

function renderMetricCards(metrics) {
  const container = document.getElementById("metricsCards");
  container.innerHTML = "";
  const labels = {
    mean_first_passage_time_s: "Mean first-passage time (s)", probability_bulk_transmission: "Probability of bulk transmission", probability_surface_return: "Probability of surface return", bottleneck_barrier_eV: "Critical bottleneck barrier (eV)", minimum_resistance_path_s: "Minimum path resistance (s)", path_tortuosity: "Pathway tortuosity", pathway_energy_cost_eV: "Mean activation barrier on dominant path (eV)", trap_residence_fraction: "Trap residence fraction", percolation_accessibility_fraction: "Accessible interstitial fraction", retention_factor_vs_host: "Retention factor vs ideal host"};
  for (const [key, label] of Object.entries(labels)) {
    if (!(key in metrics)) continue;
    const div = document.createElement("div");
    div.className = "metric";
    div.innerHTML = `<div class="value">${formatValue(metrics[key])}</div><div class="label">${label}</div>`;
    container.appendChild(div);
  }
}

// ---- Plot (a): lattice-aware minimum-resistance channel -------------------
function soluteStyle(el) {
  if (["C", "N", "O", "B"].includes(el)) return {symbol: "diamond", color: "#ef4444"};
  return {symbol: "triangle-up", color: "#22c55e"};
}

function renderPathPlot(result) {
  const d = result.domain || latestSetup.domain;
  const traces = [];
  if (result.lattice_points?.x?.length) traces.push({x: result.lattice_points.x, y: result.lattice_points.z, mode: "markers", type: "scatter", name: "host lattice", marker: {size: 4, color: "rgba(100,116,139,0.30)"}});
  if (result.interstitial_sites?.x?.length) traces.push({x: result.interstitial_sites.x, y: result.interstitial_sites.z, mode: "markers", type: "scatter", name: "interstitial sites", marker: {size: 3, color: "rgba(125,211,252,0.35)"}});
  if (result.trap_sites?.x?.length) traces.push({x: result.trap_sites.x, y: result.trap_sites.z, mode: "markers", type: "scatter", name: "trap sites (U<0)", marker: {size: 7, color: "rgba(168,85,247,0.50)", symbol: "circle"}});
  if (result.high_barrier_edges?.x?.length) traces.push({x: result.high_barrier_edges.x, y: result.high_barrier_edges.z, mode: "markers", type: "scatter", name: "high-barrier edges (top 5%)", marker: {size: 7, color: "rgba(220,38,38,0.65)", symbol: "x"}});
  const byEl = {};
  for (const s of result.solute_positions || []) {
    if (!byEl[s.element]) byEl[s.element] = {x: [], z: []};
    byEl[s.element].x.push(s.x); byEl[s.element].z.push(s.z);
  }
  for (const [el, pts] of Object.entries(byEl)) {
    const st = soluteStyle(el);
    traces.push({x: pts.x, y: pts.z, mode: "markers", type: "scatter", name: `${el} solute`, marker: {size: 10, symbol: st.symbol, color: st.color, line: {width: 0.5, color: "#111"}, opacity: 0.9}});
  }
  const path = result.dominant_pathway || {x: [], z: [], activation_barrier_eV: []};
  traces.push({x: path.x || [], y: path.z || [], mode: "lines+markers", type: "scatter", name: "minimum-resistance channel", line: {width: 4, color: TEAL}, marker: {size: 7, color: path.activation_barrier_eV || [], colorscale: INFERNO, colorbar: {title: "activation<br>barrier (eV)"}, line: {width: 0.3, color: "#111"}}});
  if (path.x && path.x.length && Number.isInteger(path.injection_index)) {
    const jx = path.x[path.injection_index], jz = path.z[path.injection_index];
    traces.push({x: [jx], y: [jz], mode: "markers", type: "scatter", name: "injection site", marker: {size: 18, symbol: "star", color: "#fde047", line: {width: 0.8, color: "#111"}}});
  }
  if (path.surface_exit) traces.push({x: [path.surface_exit.x], y: [path.surface_exit.z], mode: "markers", type: "scatter", name: "surface exit", marker: {size: 11, symbol: "square", color: "#0ea5e9", line: {width: 0.6, color: "#111"}}});
  if (path.bulk_exit) traces.push({x: [path.bulk_exit.x], y: [path.bulk_exit.z], mode: "markers", type: "scatter", name: "bulk exit", marker: {size: 11, symbol: "square", color: "#7c3aed", line: {width: 0.6, color: "#111"}}});
  Plotly.newPlot("pathPlot", traces, {
    title: "Minimum-resistance channel through injection site (lattice y-slice)",
    xaxis: {title: "x / a", range: [-0.25, d.Nx + 0.25]},
    yaxis: {title: "depth z / a", range: [d.Nz + 0.25, -0.25]},
    legend: {orientation: "h"}, margin: {l: 65, r: 30, t: 55, b: 90}
  }, {responsive: true});
}

// ---- Plot (b): activation-barrier profile along the channel ---------------
function renderBarrierPlot(result) {
  const bp = result.barrier_profile || {jump_index: [], activation_barrier_eV: []};
  const x = bp.jump_index || [], y = bp.activation_barrier_eV || [];
  const traces = [{x, y, mode: "lines+markers", type: "scatter", name: "activation barrier", line: {width: 2.4, color: TEAL}, marker: {size: 6, color: y, colorscale: INFERNO, line: {width: 0.3, color: "#111"}}}];
  if (Number.isInteger(bp.bottleneck_index) && bp.bottleneck_index > 0) {
    const bi = bp.bottleneck_index - 1;
    traces.push({x: [x[bi]], y: [y[bi]], mode: "markers", type: "scatter", name: "bottleneck", marker: {size: 15, symbol: "circle-open", color: "#dc2626", line: {width: 2.2}}});
  }
  const shapes = [];
  const anns = [];
  if (typeof bp.host_barrier_eV === "number") {
    shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: bp.host_barrier_eV, y1: bp.host_barrier_eV, line: {dash: "dash", width: 1.3, color: "#64748b"}});
    anns.push({xref: "paper", x: 0.01, y: bp.host_barrier_eV, yanchor: "bottom", showarrow: false, text: `host barrier Eₘ⁰ = ${bp.host_barrier_eV.toFixed(3)} eV`, font: {size: 11, color: "#475569"}});
  }
  if (Number.isInteger(bp.injection_index) && bp.injection_index > 0 && bp.injection_index < x.length) {
    const jx = bp.injection_index + 0.5;
    shapes.push({type: "line", x0: jx, x1: jx, yref: "paper", y0: 0, y1: 1, line: {dash: "dot", width: 1.2, color: "#f59e0b"}});
    anns.push({x: jx, yref: "paper", y: 0.98, yanchor: "top", showarrow: false, text: "injection", textangle: 90, font: {size: 10, color: "#b45309"}});
  }
  Plotly.newPlot("barrierPlot", traces, {
    title: "Activation-barrier profile along channel (surface → bulk)",
    xaxis: {title: "jump index along channel"},
    yaxis: {title: "activation barrier (eV)"},
    shapes, annotations: anns, showlegend: false, margin: {l: 70, r: 30, t: 55, b: 60}
  }, {responsive: true});
}

// ---- Plot (c): global survival probability --------------------------------
function renderSurvivalPlot(result) {
  let t = result.survival_probability?.time_s || [];
  let s = result.survival_probability?.S_t || [];
  const mfpt = result.survival_probability?.mfpt_s;
  if (!isFiniteArray(t) || !isFiniteArray(s) || t.length !== s.length) { t = [0, 1]; s = [1, 0]; }
  const shapes = [], anns = [];
  if (typeof mfpt === "number" && mfpt > 0) {
    shapes.push({type: "line", x0: mfpt, x1: mfpt, yref: "paper", y0: 0, y1: 1, line: {dash: "dot", width: 1.4, color: "#dc2626"}});
    anns.push({x: mfpt, y: 0.5, xanchor: "left", showarrow: false, text: ` MFPT = ${Number(mfpt).toExponential(2)} s`, font: {size: 11, color: "#991b1b"}});
  }
  Plotly.newPlot("survivalPlot", [{x: t, y: s, mode: "lines", type: "scatter", name: "S(t)", fill: "tozeroy", fillcolor: "rgba(15,118,110,0.08)", line: {width: 3, color: TEAL}}], {
    title: "Global survival probability S(t)",
    xaxis: {title: "time (s)"}, yaxis: {title: "S(t)", range: [0, 1.05]},
    shapes, annotations: anns, showlegend: false, margin: {l: 65, r: 30, t: 55, b: 60}
  }, {responsive: true});
}

// ---- Plot (d): site-resolved survival map at t = MFPT ---------------------
function renderSurvivalMapPlot(result) {
  const d = result.domain || latestSetup.domain;
  const m = result.survival_map || {x: [], z: [], S: []};
  const traces = [{
    x: m.x || [], y: m.z || [], mode: "markers", type: "scattergl", name: "S_i(MFPT)",
    marker: {size: 6, color: m.S || [], colorscale: "Viridis", cmin: 0, cmax: 1, colorbar: {title: "P(survive<br>to MFPT)"}, symbol: "square"}
  }];
  const path = result.dominant_pathway;
  if (path?.x?.length) {
    traces.push({x: path.x, y: path.z, mode: "lines", type: "scatter", name: "channel", line: {width: 2.4, color: "#ffffff"}});
    if (Number.isInteger(path.injection_index)) traces.push({x: [path.x[path.injection_index]], y: [path.z[path.injection_index]], mode: "markers", type: "scatter", name: "injection", marker: {size: 15, symbol: "star", color: "#fde047", line: {width: 0.7, color: "#111"}}});
  }
  Plotly.newPlot("survivalMapPlot", traces, {
    title: "Site-resolved survival probability at t = MFPT",
    xaxis: {title: "x / a", range: [0, d.Nx]},
    yaxis: {title: "depth z / a", range: [d.Nz, 0]},
    legend: {orientation: "h"}, margin: {l: 65, r: 30, t: 55, b: 70}
  }, {responsive: true});
}

function downloadInputJson() { if (!latestSetup) { alert("Generate energy inputs first."); return; } downloadJson("idoms_input.json", {setup: latestSetup, energies_eV: latestEnergyTemplate.length ? collectEnergies() : {}}); }
function downloadResultJson() { if (!latestResult) { alert("Run a simulation first."); return; } downloadJson("idoms_result.json", latestResult); }
function downloadJson(filename, obj) { const blob = new Blob([JSON.stringify(obj, null, 2)], {type: "application/json"}); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url); }
