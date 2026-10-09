"""A conservative dimensional hypha, separate from the legacy graph prototype."""
from copy import deepcopy
from dataclasses import asdict, fields
import json
from math import ceil, isfinite
from pathlib import Path
from .state import Parameters, Bath, Cell, Septum
from .mechanisms import nutrient_uptake, reaction, water_transport, solute_diffusion, wall_expansion


class Hypha:
    schema = 'mycelia.mechanistic-hypha.v1'

    def __init__(self, parameters=None, *, initial_pressure_MPa=0.4,
                 initial_ATP_pmol=0.015, initial_cargo_pmol=0.0002,
                 tip_bath_nutrient=True):
        self.parameters = parameters or Parameters()
        p = self.parameters
        for value in (initial_pressure_MPa, initial_ATP_pmol, initial_cargo_pmol):
            if not isfinite(value) or value < 0:
                raise ValueError('initial pressure, ATP and cargo must be finite and nonnegative')
        self.baths = [Bath(), Bath(nutrient_pmol=2.0 if tip_bath_nutrient else 0.0)]
        # Match external osmotic concentration between source-only and rich baths.
        if not tip_bath_nutrient:
            self.baths[1].inert_osmolyte_pmol = 2.0
        self.cells = []
        volume = p.cross_section_pL_um * p.segment_length_um
        water = volume * (1 + initial_pressure_MPa / p.bulk_modulus_MPa)
        for i in range(2):
            total_solute = (initial_pressure_MPa + self.baths[i].osmotic_pressure(p)) * water / p.RT
            nutrient = min(0.01, total_solute * 0.2)
            self.cells.append(Cell(i, i, volume, water, nutrient, total_solute - nutrient,
                                   0.03, volume * p.wall_carbon_pmol_pL, initial_ATP_pmol,
                                   initial_cargo_pmol if i == 1 else 0.0,
                                   initial_cargo_pmol if i == 1 else 0.0, i == 1))
        self.septa = [Septum(0, 1)]
        self.time_min = 0.0
        self.steps = 0
        self.limiter_events = 0
        self._substep = 0.0
        self.ledger = dict(respired_carbon_pmol=0.0, ATP_spent_pmol=0.0,
                           inert_osmolyte_added_pmol=0.0)
        self.initial = self._totals()
        self.initial_rest_length_um = sum(c.rest_length(p) for c in self.cells)
        self.history = []
        self.validate()
        self._record()

    def _totals(self):
        carbon_fields = ('nutrient_pmol', 'osmolyte_pmol', 'reserve_pmol', 'wall_pmol',
                         'cytoplasmic_cargo_pmol', 'apical_cargo_pmol')
        return dict(carbon_pmol=sum(b.nutrient_pmol for b in self.baths) + sum(getattr(c, f) for c in self.cells for f in carbon_fields),
                    water_pL=sum(b.water_pL for b in self.baths) + sum(c.water_pL for c in self.cells),
                    ATP_pmol=sum(c.ATP_pmol for c in self.cells),
                    inert_osmolyte_pmol=sum(b.inert_osmolyte_pmol for b in self.baths))

    def budgets(self):
        actual = self._totals()
        expected = dict(self.initial)
        expected['carbon_pmol'] += self.ledger.get('carbon_added_pmol', 0.0) - self.ledger.get('carbon_removed_pmol', 0.0) - self.ledger['respired_carbon_pmol']
        expected['ATP_pmol'] += self.parameters.ATP_yield * self.ledger['respired_carbon_pmol'] - self.ledger['ATP_spent_pmol']
        expected['inert_osmolyte_pmol'] += self.ledger['inert_osmolyte_added_pmol']
        return {name: actual[name] - expected[name] for name in actual}

    def validate(self):
        p = self.parameters
        if not self.cells or len(self.cells) > p.max_compartments or sum(c.tip for c in self.cells) != 1:
            raise ValueError('a hypha must contain exactly one apical compartment within capacity')
        if not isfinite(self.time_min) or self.time_min < 0 or not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError('invalid clock')
        for bath in self.baths:
            for f in fields(bath):
                value = getattr(bath, f.name)
                if not isfinite(value) or value < -1e-13:
                    raise ValueError(f'invalid bath {f.name}')
            if bath.water_pL <= 0:
                raise ValueError('bath must contain water')
        for i, c in enumerate(self.cells):
            if c.id != i or not isinstance(c.bath, int) or not 0 <= c.bath < len(self.baths) or not isinstance(c.tip, bool):
                raise ValueError('invalid cell identity, bath, or tip')
            for f in fields(c):
                if f.name in {'id', 'bath', 'tip'}:
                    continue
                value = getattr(c, f.name)
                if not isfinite(value) or value < -1e-13:
                    raise ValueError(f'invalid cell {i} {f.name}')
            if c.water_pL <= 0 or c.rest_volume_pL <= 0:
                raise ValueError('cell water and rest volume must be positive')
        pairs = set()
        for edge in self.septa:
            if not isinstance(edge.a, int) or not isinstance(edge.b, int) or not 0 <= edge.a < len(self.cells) or not 0 <= edge.b < len(self.cells) or edge.a == edge.b:
                raise ValueError('invalid septal endpoints')
            pair = tuple(sorted((edge.a, edge.b)))
            if pair in pairs:
                raise ValueError('duplicate septum')
            pairs.add(pair)
            if not isfinite(edge.opening) or not 0 <= edge.opening <= 1 or not isfinite(edge.last_flow_pL_min):
                raise ValueError('invalid septal opening or flow')
        for value in [*self.initial.values(), *self.ledger.values()]:
            if not isfinite(value) or value < -1e-13:
                raise ValueError('invalid resource ledger')
        for name, error in self.budgets().items():
            if abs(error) > 1e-10 * max(1.0, self.initial[name]):
                raise ValueError(f'{name} balance failed: {error:.6g}')

    def _septate(self):
        p = self.parameters
        tip = next(c for c in self.cells if c.tip)
        if tip.rest_length(p) < 2 * p.segment_length_um or len(self.cells) >= p.max_compartments:
            return
        new = deepcopy(tip)
        new.id = len(self.cells)
        fraction = 0.5
        for field in ('rest_volume_pL', 'water_pL', 'nutrient_pmol', 'osmolyte_pmol',
                      'reserve_pmol', 'wall_pmol', 'ATP_pmol'):
            setattr(new, field, getattr(tip, field) * fraction)
            setattr(tip, field, getattr(tip, field) * (1 - fraction))
        # Tip-local vesicle pools follow the advancing apex rather than doubling.
        tip.cytoplasmic_cargo_pmol = tip.apical_cargo_pmol = 0.0
        tip.tip = False
        self.cells.append(new)
        self.septa.append(Septum(tip.id, new.id))

    def step(self, dt_min=0.1):
        if not isfinite(dt_min) or dt_min <= 0:
            raise ValueError('dt_min must be finite and positive')
        self.validate()
        snapshot = deepcopy(self.__dict__)
        try:
            count = ceil(dt_min / self.parameters.max_substep_min)
            dt = dt_min / count
            for _ in range(count):
                nutrient_uptake(self, dt)
                for c in self.cells:
                    reaction(c, self.parameters, dt, self.ledger)
                water_transport(self, dt)
                solute_diffusion(self, dt)
                for c in self.cells:
                    wall_expansion(c, self.parameters, dt, self.ledger)
                self._septate()
                self.time_min += dt
                self.validate()
            self.steps += 1
            self._record()
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise
        return self.summary()

    def run(self, duration_min, dt_min=0.1):
        if not isfinite(duration_min) or duration_min < 0 or not isfinite(dt_min) or dt_min <= 0:
            raise ValueError('duration must be finite and nonnegative; dt must be positive')
        # Avoid a spurious microscopic final step from accumulated floating point.
        remaining = duration_min
        while remaining > max(1e-12, duration_min * 1e-12):
            advance = min(dt_min, remaining)
            self.step(advance)
            remaining -= advance
        return self.summary()

    def set_bath_nutrient(self, bath_id, amount_pmol):
        """External finite-substrate intervention, recorded as carbon added/removed."""
        if isinstance(bath_id, bool) or not isinstance(bath_id, int) or not 0 <= bath_id < len(self.baths):
            raise ValueError('bath_id must identify an existing bath')
        if isinstance(amount_pmol, bool) or not isinstance(amount_pmol, (int, float)) or not isfinite(amount_pmol) or amount_pmol < 0:
            raise ValueError('amount_pmol must be finite and nonnegative')
        self.validate()
        difference = amount_pmol - self.baths[bath_id].nutrient_pmol
        field = 'carbon_added_pmol' if difference >= 0 else 'carbon_removed_pmol'
        self.ledger[field] = self.ledger.get(field, 0.0) + abs(difference)
        self.baths[bath_id].nutrient_pmol = float(amount_pmol)
        self.validate()

    def hyperosmotic_shock(self, delta_MPa):
        if not isfinite(delta_MPa) or delta_MPa < 0:
            raise ValueError('osmotic pressure increment must be finite and nonnegative')
        self.validate()
        for bath in self.baths:
            added = delta_MPa * bath.water_pL / self.parameters.RT
            bath.inert_osmolyte_pmol += added
            self.ledger['inert_osmolyte_added_pmol'] += added
        self.validate()
        self._record()

    def summary(self):
        p = self.parameters
        tip = next(c for c in self.cells if c.tip)
        total_length = sum(c.rest_length(p) for c in self.cells)
        return dict(time_min=self.time_min, step=self.steps, compartments=len(self.cells),
                    tip_pressure_MPa=tip.pressure(p), tip_water_pL=tip.water_pL,
                    total_water_pL=sum(c.water_pL for c in self.cells),
                    total_rest_volume_pL=sum(c.rest_volume_pL for c in self.cells),
                    rest_length_um=total_length, permanent_extension_um=total_length-self.initial_rest_length_um,
                    tip_osmotic_pressure_MPa=tip.osmotic_pressure(p),
                    tip_external_osmotic_pressure_MPa=self.baths[tip.bath].osmotic_pressure(p),
                    tip_nutrient_pmol=tip.nutrient_pmol, tip_ATP_pmol=tip.ATP_pmol,
                    tip_cytoplasmic_cargo_pmol=tip.cytoplasmic_cargo_pmol,
                    tip_apical_cargo_pmol=tip.apical_cargo_pmol,
                    wall_carbon_pmol=sum(c.wall_pmol for c in self.cells),
                    flow_toward_tip_pL_min=sum(e.last_flow_pL_min if e.b == tip.id else -e.last_flow_pL_min if e.a == tip.id else 0 for e in self.septa),
                    limiter_events=self.limiter_events, budget_errors=self.budgets(),
                    model='dimensional mechanistic hypha; uncalibrated',
                    empirical_validation='not performed')

    def _record(self):
        self.history.append(self.summary())
        if len(self.history) > 10000:
            self.history = self.history[-10000:]

    def state_dict(self):
        return dict(schema=self.schema, parameters=asdict(self.parameters),
                    baths=[asdict(b) for b in self.baths], cells=[asdict(c) for c in self.cells],
                    septa=[asdict(s) for s in self.septa], time_min=self.time_min, steps=self.steps,
                    limiter_events=self.limiter_events, initial=deepcopy(self.initial), ledger=deepcopy(self.ledger),
                    initial_rest_length_um=self.initial_rest_length_um, history=deepcopy(self.history))

    @classmethod
    def from_state_dict(cls, data):
        if data.get('schema') != cls.schema:
            raise ValueError('unsupported mechanistic schema; prototype states cannot be migrated implicitly')
        obj = cls.__new__(cls)
        obj.parameters = Parameters(**data['parameters'])
        obj.baths = [Bath(**b) for b in data['baths']]
        obj.cells = [Cell(**c) for c in data['cells']]
        obj.septa = [Septum(**s) for s in data['septa']]
        for field in ('time_min', 'steps', 'limiter_events', 'initial', 'ledger', 'initial_rest_length_um', 'history'):
            setattr(obj, field, deepcopy(data[field]))
        if not isfinite(obj.initial_rest_length_um) or obj.initial_rest_length_um <= 0:
            raise ValueError('invalid initial length')
        obj._substep = 0.0
        obj.validate()
        return obj

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.state_dict(), indent=2, allow_nan=False))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        return cls.from_state_dict(json.loads(Path(path).read_text()))
