from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Literal
from engine import run_simulation

app = FastAPI(title="IDOMS Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your GitHub Pages URL later.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Domain(BaseModel):
    Nx: int
    Ny: int
    Nz: int
    unit: str
    total_lattice_sites: int
    required_lattice_sites_minimum: int
    physical_size_nm: Dict[str, float]
    estimated_atom_counts: Dict[str, int]

class ToolSetup(BaseModel):
    host: str
    alloying_elements: List[str] = Field(default_factory=list)
    composition_at_percent: Dict[str, float] = Field(default_factory=dict)
    host_at_percent: float
    crystal_structure: Literal["BCC", "FCC"]
    lattice_parameter_A: float
    interstitial_species: Literal["H", "D", "T", "C", "N", "O"]
    temperature_K: float
    requested_slab_thickness_nm: float
    boundary: Literal["surface_return", "bulk_transmission", "retention"]
    site_per_unit_cell: int
    domain: Domain

class SimulationInput(BaseModel):
    setup: ToolSetup
    energies_eV: Dict[str, float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/simulate")
def simulate(inp: SimulationInput):
    return run_simulation(inp.setup.model_dump(), inp.energies_eV)
