const API_BASE = "http://127.0.0.1:8000"; 
// later replace with your Render backend URL

let latestSetup = null;
let requiredEnergyKeys = [];

function collectSetup() {
  const elements = document.getElementById("elements").value
    .split(",")
    .map(x => x.trim())
    .filter(x => x.length > 0);

  const domain = document.getElementById("domain").value
    .split(",")
    .map(Number);

  return {
    host: document.getElementById("host").value,
    alloying_elements: elements,
    composition: {},
    crystal_structure: document.getElementById("crystal").value,
    interstitial_species: document.getElementById("interstitial").value,
    temperature_K: Number(document.getElementById("temperature").value),
    domain_cells: domain,
    boundary: document.getElementById("boundary").value
  };
}

async function getEnergyTemplate() {
  latestSetup = collectSetup();

  const response = await fetch(`${API_BASE}/energy-template`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(latestSetup)
  });

  const data = await response.json();
  requiredEnergyKeys = data.required_energy_values.map(x => x.key);

  const container = document.getElementById("energyFields");
  container.innerHTML = "";

  data.required_energy_values.forEach(item => {
    container.innerHTML += `
      <label>${item.label} (${item.unit})</label>
      <input id="${item.key}" type="number" step="0.001" value="0.000">
      <br>
    `;
  });
}

async function runSimulation() {
  const energies = {};
  requiredEnergyKeys.forEach(key => {
    energies[key] = Number(document.getElementById(key).value);
  });

  const payload = {
    setup: latestSetup,
    energies_eV: energies
  };

  const response = await fetch(`${API_BASE}/simulate`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  const result = await response.json();

  document.getElementById("metrics").textContent =
    JSON.stringify(result.metrics, null, 2);

  Plotly.newPlot("pathPlot", [{
    x: result.dominant_pathway.x,
    y: result.dominant_pathway.z,
    mode: "lines+markers",
    name: "Dominant pathway"
  }], {
    title: "Dominant Interstitial Pathway",
    xaxis: {title: "x / a"},
    yaxis: {title: "z / a"}
  });

  Plotly.newPlot("survivalPlot", [{
    x: result.survival_probability.time_s,
    y: result.survival_probability.S_t,
    mode: "lines+markers",
    name: "S(t)"
  }], {
    title: "Survival Probability",
    xaxis: {title: "time / s"},
    yaxis: {title: "S(t)"}
  });
}
