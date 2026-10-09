"""Dimensional state for the mechanistic hypha milestone; no prototype imports."""
from dataclasses import dataclass, fields
from math import isfinite, pi


@dataclass(frozen=True)
class Parameters:
    temperature_K: float = 298.15
    radius_um: float = 2.5
    segment_length_um: float = 12.0
    growth_zone_um: float = 4.0
    bulk_modulus_MPa: float = 4.0
    membrane_permeability: float = 0.0005  # pL/(um^2 MPa min)
    uptake_density: float = 0.00002  # pmol/(um^2 min)
    uptake_Km_M: float = 0.01
    metabolism_per_min: float = 0.2
    ATP_yield: float = 5.0  # ATP equivalents / respired carbon equivalent
    maintenance_density: float = 0.0002  # pmol ATP/(pL min)
    reserve_release_per_min: float = 0.03
    vesicle_synthesis_per_min: float = 0.5
    vesicle_synthesis_ATP: float = 0.5
    motor_delivery_per_min: float = 2.0
    motor_delivery_ATP: float = 0.2
    exocytosis_per_min: float = 4.0
    wall_carbon_density: float = 0.0001  # pmol / um^2
    wall_insertion_ATP: float = 0.5
    wall_extensibility: float = 0.8  # 1/(MPa min), local growth zone
    wall_yield_MPa: float = 0.15
    pore_radius_um: float = 0.15
    septum_length_um: float = 0.5
    effective_viscosity_Pa_s: float = 0.1
    diffusion_um2_min: float = 20.0
    max_substep_min: float = 0.01
    max_compartments: int = 128

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value < 0:
                raise ValueError(f'{f.name} must be finite and nonnegative')
        for name in ('temperature_K', 'radius_um', 'segment_length_um', 'growth_zone_um',
                     'bulk_modulus_MPa', 'uptake_Km_M', 'wall_carbon_density',
                     'septum_length_um', 'effective_viscosity_Pa_s', 'max_substep_min'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.growth_zone_um > self.segment_length_um:
            raise ValueError('growth zone must fit within an initial segment')
        if self.max_substep_min > 0.02:
            raise ValueError('max_substep_min must be <= 0.02; refinement is required for larger steps')
        if not isinstance(self.max_compartments, int) or self.max_compartments < 2:
            raise ValueError('max_compartments must be an integer >= 2')

    @property
    def RT(self):
        # R = 0.008314462618 MPa L/(mol K); pmol/pL = mol/L.
        return 0.008314462618 * self.temperature_K

    @property
    def cross_section_pL_um(self):
        return pi * self.radius_um ** 2 * 0.001

    @property
    def wall_carbon_pmol_pL(self):
        return 2000 * self.wall_carbon_density / self.radius_um


@dataclass
class Bath:
    water_pL: float = 100.0
    nutrient_pmol: float = 2.0
    inert_osmolyte_pmol: float = 0.0

    def osmotic_pressure(self, p):
        return p.RT * (self.nutrient_pmol + self.inert_osmolyte_pmol) / self.water_pL


@dataclass
class Cell:
    id: int
    bath: int
    rest_volume_pL: float
    water_pL: float
    nutrient_pmol: float
    osmolyte_pmol: float
    reserve_pmol: float
    wall_pmol: float
    ATP_pmol: float
    cytoplasmic_cargo_pmol: float = 0.0
    apical_cargo_pmol: float = 0.0
    tip: bool = False

    def pressure(self, p):
        return p.bulk_modulus_MPa * max(0.0, self.water_pL / self.rest_volume_pL - 1)

    def osmotic_pressure(self, p):
        return p.RT * (self.nutrient_pmol + self.osmolyte_pmol) / self.water_pL

    def rest_length(self, p):
        return self.rest_volume_pL / p.cross_section_pL_um

    def membrane_area(self, p):
        # Lateral cylinder area; caps/septal faces are not external uptake surfaces.
        return 2 * pi * p.radius_um * self.rest_length(p)


@dataclass
class Septum:
    a: int
    b: int
    opening: float = 1.0  # effective fraction of conductance, prescribed in this milestone
    last_flow_pL_min: float = 0.0

    def conductance(self, p):
        # Poiseuille unit conversion: um, Pa*s -> pL/(MPa min).
        return self.opening * 60000 * pi * p.pore_radius_um ** 4 / (8 * p.effective_viscosity_Pa_s * p.septum_length_um)
