const MAX_ALLOY_ELEMENTS = 2;
const MIN_ATOMS_PER_ALLOY = 4;
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

function apiBase() {
  return document.getElementById("backendUrl").value.trim().replace(/\/$/, "");
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

function addCompositionRow(element = "", atpct = "") {
  const tbody = document.getElementById("compositionRows");
  if (tbody.children.length >= MAX_ALLOY_ELEMENTS) {
    alert(`This version allows up to ${MAX_ALLOY_ELEMENTS} alloying elements.`);
    return;
  }
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td><input class="el-symbol" value="${element}" placeholder="e.g. Cr" oninput="updateCompositionSummary()" /></td>
    <td><input class="el-atpct" type="number" min="0" max="100" step="0.001" value="${atpct}" oninput="updateCompositionSummary()" /></td>
    <td><button class="remove-btn" onclick="removeRow(this)">Remove</button></td>`;
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

function buildEnergyTemplate(setup) {
  const I = setup.interstitial_species;
  const host = setup.host;
  const defaultBarrier = setup.crystal_structure === "BCC" && host === "Fe" && I === "H" ? 0.088 : 0.100;
  const template = [
    {key: "host_site_energy", label: `Reference site energy U for ${I} in ${host} interstitial site`, kind: "site energy", unit: "eV", default: 0.000},
    {key: "host_migration_barrier", label: `Baseline ${I} migration barrier in ${host}`, kind: "activation barrier", unit: "eV", default: defaultBarrier},
    {key: "surface_exit_barrier", label: "Surface exit / return barrier", kind: "boundary barrier", unit: "eV", default: defaultBarrier},
    {key: "bulk_exit_barrier", label: "Bulk transmission barrier", kind: "boundary barrier", unit: "eV", default: defaultBarrier}
  ];
  for (const el of setup.alloying_elements) {
    template.push(
      {key: `${el}_site_energy_shift`, label: `Site-energy shift ΔU for ${I} near ${el} (negative = attractive/trapping)`, kind: "site energy", unit: "eV", default: 0.000},
      {key: `${el}_saddle_energy_shift`, label: `Saddle-energy shift ΔE‡ for ${I} jumps near ${el}`, kind: "saddle energy", unit: "eV", default: 0.000},
      {key: `${el}_cluster_trap_depth`, label: `Additional trap depth near ${el}-rich local region`, kind: "cluster site energy", unit: "eV", default: 0.000}
    );
  }
  if (setup.alloying_elements.length === 2) {
    const [a, b] = setup.alloying_elements;
    template.push({key: `pair_${a}_${b}_saddle_shift`, label: `Pair correction to saddle energy near ${a}–${b} local environment`, kind: "pair saddle energy", unit: "eV", default: 0.000});
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
    renderResults(latestResult);
    document.getElementById("runStatus").textContent = "Simulation complete. Results are from backend graph/master-equation calculation using user-supplied energies.";
  } catch (err) {
    alert(err.message);
    document.getElementById("runStatus").textContent = "Simulation failed.";
  }
}

function renderResults(result) { renderMetricCards(result.metrics || {}); renderPathPlot(result); renderSurvivalPlot(result); }
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

function renderPathPlot(result) {
  const d = result.domain || latestSetup.domain;
  const traces = [];
  if (result.lattice_points?.x?.length) traces.push({x: result.lattice_points.x, y: result.lattice_points.z, mode: "markers", type: "scatter", name: "host lattice points near path", marker: {size: 4, color: "rgba(100,116,139,0.28)"}});
  if (result.interstitial_sites?.x?.length) traces.push({x: result.interstitial_sites.x, y: result.interstitial_sites.z, mode: "markers", type: "scatter", name: "interstitial sites near path", marker: {size: 3, color: "rgba(14,165,233,0.25)"}});
  const byEl = {};
  for (const s of result.solute_positions || []) {
    if (!byEl[s.element]) byEl[s.element] = {x: [], z: []};
    byEl[s.element].x.push(s.x); byEl[s.element].z.push(s.z);
  }
  for (const [el, pts] of Object.entries(byEl)) traces.push({x: pts.x, y: pts.z, mode: "markers", type: "scatter", name: `${el} solute near path`, marker: {size: 9, symbol: "diamond", opacity: 0.78}});
  if (result.trap_sites?.x?.length) traces.push({x: result.trap_sites.x, y: result.trap_sites.z, mode: "markers", type: "scatter", name: "low-site-energy trap sites", marker: {size: 6, color: "rgba(168,85,247,0.55)", symbol: "circle"}});
  if (result.high_barrier_edges?.x?.length) traces.push({x: result.high_barrier_edges.x, y: result.high_barrier_edges.z, mode: "markers", type: "scatter", name: "highest-barrier edge midpoints", marker: {size: 6, color: "rgba(239,68,68,0.62)", symbol: "x"}});
  const path = result.dominant_pathway || {x: [], z: [], activation_barrier_eV: []};
  traces.push({x: path.x || [], y: path.z || [], mode: "lines+markers", type: "scatter", name: "dominant minimum-resistance pathway", line: {width: 5, color: "#0f766e"}, marker: {size: 8, color: path.activation_barrier_eV || [], colorscale: "Inferno", colorbar: {title: "activation<br>barrier (eV)"}}});
  Plotly.newPlot("pathPlot", traces, {title: "Lattice-aware interstitial pathway (filtered near dominant path)", xaxis: {title: "x / a", range: [0, d.Nx]}, yaxis: {title: "depth z / a", range: [d.Nz, 0]}, legend: {orientation: "h"}, margin: {l: 65, r: 30, t: 55, b: 90}}, {responsive: true});
}

function renderSurvivalPlot(result) {
  let t = result.survival_probability?.time_s || [];
  let s = result.survival_probability?.S_t || [];
  if (!t.length || !s.length) { t = [0, 1]; s = [1, 0]; }
  Plotly.newPlot("survivalPlot", [{x: t, y: s, mode: "lines", type: "scatter", name: "S(t)", line: {width: 4, color: "#0f766e"}}], {title: "Survival probability", xaxis: {title: "time (s)"}, yaxis: {title: "S(t)", range: [0, 1.05]}, margin: {l: 65, r: 30, t: 55, b: 70}}, {responsive: true});
}

function downloadInputJson() { if (!latestSetup) { alert("Generate energy inputs first."); return; } downloadJson("idoms_input.json", {setup: latestSetup, energies_eV: latestEnergyTemplate.length ? collectEnergies() : {}}); }
function downloadResultJson() { if (!latestResult) { alert("Run a simulation first."); return; } downloadJson("idoms_result.json", latestResult); }
function downloadJson(filename, obj) { const blob = new Blob([JSON.stringify(obj, null, 2)], {type: "application/json"}); const url = URL.createObjectURL(blob); const a = document.createElement("a"); a.href = url; a.download = filename; a.click(); URL.revokeObjectURL(url); }
