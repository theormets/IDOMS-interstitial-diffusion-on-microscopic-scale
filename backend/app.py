from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Dict
from engine import run_simulation

app = FastAPI(title="Interstitial Pathway Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later replace with your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ToolSetup(BaseModel):
    host: str
    alloying_elements: List[str]
    composition: Dict[str, float]
    crystal_structure: Literal["BCC", "FCC"]
    interstitial_species: Literal["H", "D", "T", "C", "N", "O"]
    temperature_K: float
    domain_cells: List[int]
    boundary: Literal["surface_return", "bulk_transmission", "retention"]

class SimulationInput(BaseModel):
    setup: ToolSetup
    energies_eV: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/energy-template")
def energy_template(setup: ToolSetup):
    n = len(setup.alloying_elements)

    required = [
        {
            "key": "host_migration_barrier",
            "label": f"Baseline {setup.interstitial_species} migration barrier in {setup.host}",
            "unit": "eV",
        },
        {
            "key": "surface_exit_barrier",
            "label": "Surface exit / return barrier",
            "unit": "eV",
        },
        {
            "key": "bulk_exit_barrier",
            "label": "Bulk transmission barrier",
            "unit": "eV",
        },
    ]

    for el in setup.alloying_elements:
        required.extend([
            {
                "key": f"{el}_interstitial_binding",
                "label": f"{setup.interstitial_species} binding energy near {el}",
                "unit": "eV",
            },
            {
                "key": f"{el}_modified_jump_barrier",
                "label": f"{el}-modified local jump barrier",
                "unit": "eV",
            },
            {
                "key": f"{el}_trap_depth",
                "label": f"Trap depth near {el}-rich local region",
                "unit": "eV",
            },
        ])

    return {
        "number_of_alloying_elements": n,
        "number_of_required_energy_values": len(required),
        "required_energy_values": required,
        "note": "All energies must be supplied by the user. No database or estimation is used in this trial version."
    }

@app.post("/simulate")
def simulate(inp: SimulationInput):
    result = run_simulation(inp.setup.model_dump(), inp.energies_eV)
    return result 
