from __future__ import annotations

from dataclasses import asdict, dataclass, fields, field
import math


@dataclass(frozen=True)
class Config:
    dt: float = 0.5
    max_nodes: int = 800
    max_tips: int = 48
    radius: float = 0.45
    wall_compliance: float = 0.35
    osmotic_scale: float = 0.9
    membrane_rate: float = 0.24
    uptake_rate: float = 0.16
    metabolic_rate: float = 0.035
    maintenance_rate: float = 0.006
    energy_yield: float = 5.0
    growth_speed: float = 0.65
    compartment_length: float = 0.5
    growth_carbon: float = 0.16
    growth_energy: float = 0.12
    branch_interval: float = 8.0
    branch_rate: float = 0.28
    fusion_radius: float = 0.65
    diffusion: float = 0.04
    solute_diffusion: float = 0.055
    hydraulic_scale: float = 0.12
    streaming_rate: float = 0.012
    reinforcement_rate: float = 0.025
    starvation_time: float = 18.0
    # Substrate level at which a living, dormant growth front may resume.
    resprout_threshold: float = 0.03
    event_limit: int = 3000
    # Candidate mechanism: uptake-associated local polarity consolidation.
    # Opt-in; none of these rates is experimentally calibrated.
    foraging_memory_rate: float = 0.0
    foraging_memory_gain: float = 0.0
    foraging_memory_decay: float = 0.001
    foraging_memory_cost: float = 0.01
    foraging_memory_inheritance: float = 0.85

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")
        for name in ("dt", "radius", "wall_compliance", "max_nodes", "max_tips", "event_limit", "branch_interval", "compartment_length"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("max_nodes", "max_tips", "event_limit"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")

        if self.foraging_memory_inheritance>1:
            raise ValueError("foraging_memory_inheritance must be <=1")
        if self.foraging_memory_rate>0 and self.foraging_memory_cost<=0:
            raise ValueError("foraging memory synthesis requires a positive carbon cost")

    def as_dict(self):
        return asdict(self)


@dataclass
class Compartment:
    id: int
    x: float
    y: float
    length: float
    radius: float
    water: float
    nutrient: float
    energy: float
    osmolyte: float
    reserve: float
    biomass: float
    pressure: float = 0.0
    age: float = 0.0
    damage: float = 0.0
    alive: bool = True

    # Phenomenological chemical imprint state; not a measured fungal receptor model.
    receptor_trace: list[float] = field(default_factory=list)
    receptor_variance: list[float] = field(default_factory=list)
    memory_material: float = 0.0
    memory_exposures: int = 0
    receptor_fast_trace: list[float] = field(default_factory=list)
    fast_memory_material: float = 0.0
    fast_memory_exposures: int = 0
    polarity_trace_x: float = 0.0
    polarity_trace_y: float = 0.0
    polarity_material: float = 0.0

    @property
    def volume(self):
        return math.pi * self.radius ** 2 * self.length

    @property
    def concentration(self):
        return self.nutrient / max(self.water, 1e-12)


@dataclass
class Segment:
    id: int
    a: int
    b: int
    length: float
    radius: float
    pore_open: float = 1.0
    flow: float = 0.0
    activity: float = 0.0
    material: float = 0.0
    age: float = 0.0
    inactive_time: float = 0.0
    alive: bool = True
    fusion: bool = False

    route_trace: float = 0.0
    route_fast_trace: float = 0.0
    route_memory_material: float = 0.0
    route_fast_material: float = 0.0

    def conductance(self, scale: float):
        if not self.alive:
            return 0.0
        return scale * self.pore_open * math.pi * self.radius ** 4 / (8 * self.length)


@dataclass
class Tip:
    id: int
    node: int
    dx: float
    dy: float
    branch_age: float = 0.0
    starvation: float = 0.0
    vesicles: float = 0.5
    pending_length: float = 0.0
    pending_water: float = 0.0
    pending_biomass: float = 0.0
    active: bool = True
